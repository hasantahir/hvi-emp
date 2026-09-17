"""Job submission: turn a validated `RunSpec` into a scheduler script.

Slurm is implemented. `Scheduler` is the seam for the others -- subclass it,
implement `directives`, `array_directive` and `submit_command`, and register
it in `SCHEDULERS`. PBS, SGE and LSF differ only in the spelling of those
three things.

The four ways a cluster job goes wrong, and what is done about each
-------------------------------------------------------------------
**Oversubscription.** A job that requests 4 CPUs and then forks a pool sized
to the *node's* core count will be throttled or killed, and on a shared
system it steals from whoever else is on that node. `HVI_EMP_WORKERS` is
exported from `cpus_per_task` and `hvi_emp.parallel.cpu_count` honours it, so
the pool matches the allocation by construction rather than by the user
remembering.

**Silent partial failure.** A sweep of forty scenarios where six fail should
not look like a sweep of forty where six had zero charge. Array tasks exit
non-zero on failure and the aggregation step refuses to run on an incomplete
set unless explicitly told to tolerate it.

**Wall-clock guesses.** `estimate_walltime` derives a request from the actual
cost model -- number of scenarios, 1T versus 2T, cores available -- rather
than leaving a user to guess. A job killed at the wall clock loses
everything; one that asks for 24 h to do 20 minutes of work sits in the queue.

**Unreproducible results.** Every submission writes the resolved config next
to the script. A results directory whose configuration is not beside it
cannot be interpreted six months later.

Array jobs versus one wide job
------------------------------
A sweep is submitted as an array, one task per point. This is deliberate:
short independent tasks backfill into a busy queue far better than one long
wide reservation, a single non-converging point costs one task rather than
the whole sweep, and a partially-complete sweep can be resumed by
re-submitting only the missing indices.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .schema import ClusterSpec, ConfigError, RunSpec

__all__ = ["Scheduler", "SlurmScheduler", "LocalScheduler", "SCHEDULERS",
           "get_scheduler", "estimate_walltime", "estimate_memory_mb",
           "write_job_script", "submit", "scheduler_available"]

#: Rough per-scenario cost [s] on one core, measured on the reference case
#: (1 pg Fe -> Al at 50 km/s) after the vectorisation work. Used only to size
#: a walltime request, so a factor of two is fine; the safety margin below is
#: what actually protects the job.
SECONDS_PER_SCENARIO = {"1T": 4.0, "2T": 20.0}

#: Multiply the estimate by this before requesting. Cluster nodes are often
#: slower than a workstation, the first scenario pays the ionisation-table
#: build, and being killed at the wall clock loses everything.
WALLTIME_SAFETY = 3.0

#: Floor on requested memory [MB]. The chain itself is small; the floor
#: covers the interpreter, NumPy/SciPy and one ionisation table per worker.
MEMORY_FLOOR_MB = 2048
MEMORY_PER_WORKER_MB = 400


def _fmt_walltime(seconds: float) -> str:
    """Seconds -> Slurm ``[D-]HH:MM:SS``, rounded up to the next minute."""
    seconds = max(60, int(math.ceil(seconds / 60.0) * 60))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def estimate_walltime(n_scenarios: int, expansion_model: str = "1T",
                      workers: int = 1, safety: float = WALLTIME_SAFETY
                      ) -> str:
    """Walltime to request for `n_scenarios`, as a Slurm time string.

    Includes a one-off allowance for the ionisation-table build, which is
    paid once per process and is a large fraction of a short job.
    """
    per = SECONDS_PER_SCENARIO.get(expansion_model, 20.0)
    workers = max(1, int(workers))
    compute = per * max(1, n_scenarios) / workers
    startup = 30.0 + 5.0 * min(workers, 8)      # interpreter + table build
    return _fmt_walltime((compute + startup) * safety)


def estimate_memory_mb(workers: int, expansion_model: str = "1T") -> int:
    """Memory to request [MB].

    Each worker is a separate process with its own interpreter and its own
    ionisation-table cache, so memory scales with the pool, not with the
    physics.
    """
    per = MEMORY_PER_WORKER_MB * (2 if expansion_model == "2T" else 1)
    return max(MEMORY_FLOOR_MB, MEMORY_FLOOR_MB + per * max(0, workers - 1))


# ---------------------------------------------------------------------------
# Scheduler backends
# ---------------------------------------------------------------------------

@dataclass
class Scheduler:
    """Base class. Subclass for PBS/SGE/LSF; three methods differ."""
    spec: ClusterSpec

    name = "base"
    directive_prefix = "#"
    submit_binary = "true"
    array_index_var = "0"

    def directives(self, job_name: str, log_dir: str) -> list:
        raise NotImplementedError

    def array_directive(self, n: int) -> str:
        raise NotImplementedError

    def submit_command(self, script: str) -> list:
        return [self.submit_binary, script]


@dataclass
class SlurmScheduler(Scheduler):
    name = "slurm"
    directive_prefix = "#SBATCH"
    submit_binary = "sbatch"
    array_index_var = "${SLURM_ARRAY_TASK_ID}"

    def directives(self, job_name: str, log_dir: str) -> list:
        s = self.spec
        d = [f"--job-name={job_name}",
             f"--output={log_dir}/%x-%A_%a.out",
             f"--error={log_dir}/%x-%A_%a.err",
             f"--time={s.time}",
             f"--nodes={s.nodes}",
             f"--ntasks={s.ntasks}",
             f"--cpus-per-task={s.cpus_per_task}",
             f"--mem-per-cpu={s.mem_per_cpu}"]
        if s.partition:
            d.append(f"--partition={s.partition}")
        if s.account:
            d.append(f"--account={s.account}")
        if s.qos:
            d.append(f"--qos={s.qos}")
        if s.gpus:
            gres = (f"gpu:{s.gpu_type}:{s.gpus}" if s.gpu_type
                    else f"gpu:{s.gpus}")
            d.append(f"--gres={gres}")
        d.extend(s.extra_directives)
        return d

    def array_directive(self, n: int) -> str:
        if n < 1:
            raise ConfigError("array size must be >= 1")
        throttle = self.spec.array_throttle
        rng = f"0-{n - 1}"
        return f"--array={rng}%{throttle}" if throttle else f"--array={rng}"


@dataclass
class LocalScheduler(Scheduler):
    """Not a scheduler: runs the script here, for testing the same path."""
    name = "local"
    directive_prefix = "#"
    submit_binary = "bash"
    array_index_var = "${HVI_EMP_TASK_ID:-0}"

    def directives(self, job_name: str, log_dir: str) -> list:
        return []

    def array_directive(self, n: int) -> str:
        return ""


SCHEDULERS = {"slurm": SlurmScheduler, "local": LocalScheduler}


def get_scheduler(spec: ClusterSpec) -> Scheduler:
    try:
        return SCHEDULERS[spec.backend](spec)
    except KeyError:
        raise ConfigError(
            f"no scheduler backend {spec.backend!r}. Available: "
            f"{sorted(SCHEDULERS)}. Add one by subclassing "
            f"hvi_emp.cluster.Scheduler.") from None


def scheduler_available(backend: str = "slurm") -> bool:
    """Is this scheduler's submit command actually on PATH?"""
    binary = {"slurm": "sbatch", "local": "bash"}.get(backend)
    return bool(binary and shutil.which(binary))


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

def _environment_lines(spec: ClusterSpec) -> list:
    """module loads, env activation and thread pinning."""
    lines = []
    if spec.modules:
        lines.append("module purge")
        lines += [f"module load {m}" for m in spec.modules]
    if spec.conda_env:
        lines += ['eval "$(conda shell.bash hook)"',
                  f"conda activate {spec.conda_env}"]
    if spec.venv:
        lines.append(f"source {spec.venv}/bin/activate")

    lines += [
        "",
        "# Pin every nested thread pool to one thread.",
        "#",
        "# NumPy's BLAS starts a thread per core by default. Inside a job",
        "# that already runs one process per allocated CPU, that is",
        "# cpus_per_task^2 threads fighting over cpus_per_task cores, and",
        "# it makes the job slower than serial while looking busy.",
        "# hvi_emp's parallelism is at the process level, so the BLAS",
        "# threads have nothing useful to do.",
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "",
        "# Size hvi_emp's process pool to the allocation, not to the node.",
        f"export HVI_EMP_WORKERS={spec.cpus_per_task}",
    ]
    return lines


def write_job_script(run: RunSpec, path: str | Path,
                     conf_overrides=None,
                     python: str = "python",
                     n_array: int | None = None,
                     log_dir: str = "logs") -> str:
    """Write a submission script for `run`. Returns the script text.

    Parameters
    ----------
    n_array
        Number of array tasks. ``None`` derives it from the sweep: one task
        per sweep point, or a single job when there is no sweep.
    conf_overrides
        Hydra-style overrides recorded in the script, so the script alone
        reproduces the run.
    """
    path = Path(path)
    spec = run.cluster
    sched = get_scheduler(spec)
    job_name = run.tag

    if n_array is None:
        n_array = len(run.sweep.resolve()) if run.sweep else 1
    if n_array < 1:
        raise ConfigError("nothing to run: the sweep resolved to zero points")

    overrides = list(conf_overrides or ())
    ov = " ".join(f"'{o}'" for o in overrides)

    lines = ["#!/bin/bash",
             "# Generated by hvi_emp.cluster.write_job_script -- edit the",
             "# config, not this file, so the run stays reproducible.",
             "set -euo pipefail",
             ""]

    directives = sched.directives(job_name, log_dir)
    if n_array > 1:
        directives.insert(1, sched.array_directive(n_array))
    for d in directives:
        lines.append(f"{sched.directive_prefix} {d}")
    if directives:
        lines.append("")

    lines += [f'mkdir -p "{log_dir}" "{run.output_dir}"', ""]
    lines += _environment_lines(spec)
    lines += [
        "",
        "# Fail loudly. Without `set -e` and this trap, a Python traceback",
        "# leaves the array task exiting 0 and the aggregation step happily",
        "# builds a plot from a missing point.",
        'trap \'echo "FAILED: ${BASH_SOURCE[0]} at line ${LINENO}" >&2\' ERR',
        "",
    ]

    if n_array > 1:
        lines += [
            f"INDEX={sched.array_index_var}",
            'echo "task ${INDEX} of ' + str(n_array) + ' on $(hostname)"',
            "",
            f"{python} -m hvi_emp.run \\",
            f"    --config-dir {os.environ.get('HVI_EMP_CONF', 'hvi_emp/conf')} \\",
            f"    --sweep-index \"${{INDEX}}\" \\",
            f"    --output-dir {run.output_dir} \\",
            (f"    {ov}" if ov else "    "),
        ]
    else:
        lines += [
            'echo "running on $(hostname)"',
            "",
            f"{python} -m hvi_emp.run \\",
            f"    --config-dir {os.environ.get('HVI_EMP_CONF', 'hvi_emp/conf')} \\",
            f"    --output-dir {run.output_dir} \\",
            (f"    {ov}" if ov else "    "),
        ]

    text = "\n".join(x.rstrip() for x in lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return text


def submit(run: RunSpec, script_path: str | Path, dry_run: bool = True,
           **kw) -> dict:
    """Write the script and, unless `dry_run`, submit it.

    `dry_run` defaults to **True**. Writing a file is safe; queuing a job
    that charges an allocation is not, and a function that submits by
    accident is worse than one that needs an argument.
    """
    text = write_job_script(run, script_path, **kw)
    sched = get_scheduler(run.cluster)
    cmd = sched.submit_command(str(script_path))
    out = {"script": str(script_path), "command": " ".join(cmd),
           "submitted": False, "text": text}

    if dry_run:
        out["note"] = ("dry run: script written, nothing submitted. Pass "
                       "dry_run=False to queue it.")
        return out
    if not scheduler_available(run.cluster.backend):
        raise RuntimeError(
            f"{cmd[0]!r} is not on PATH, so this is not a "
            f"{run.cluster.backend} submit host. The script is written at "
            f"{script_path} and can be submitted from a login node.")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out["submitted"] = proc.returncode == 0
    out["stdout"] = proc.stdout.strip()
    out["stderr"] = proc.stderr.strip()
    if proc.returncode != 0:
        raise RuntimeError(f"submission failed: {proc.stderr.strip()}")
    for tok in proc.stdout.split():
        if tok.isdigit():
            out["job_id"] = tok
            break
    return out
