"""The full waterfall: one DAG across every solver, with resume.

What this is
------------
`scripts/run_full_chain.py` is the entry point; this module is the machinery.
It defines the stages, works out which are already done, submits the
expensive ones to Slurm with dependencies, and reports what is blocked on
what.

The waterfall, in the order physics happens:

===========  ===========  ============  ==============  ==================
stage        code         length        time            typical cost
===========  ===========  ============  ==============  ==================
``reduced``  this package  --            --              seconds
``md``       LAMMPS       nm            100 ps          minutes
``hydro``    M2C          mm            us              10-40 h per case
``mhd``      OpenMHD      m             ms              hours
``pic``      PIConGPU     um            ns              hours + a compile
===========  ===========  ============  ==============  ==================

Every stage is optional. A missing solver produces its input deck, records
why it could not run, and does not stop the others -- the common case on a
fresh machine is that only `reduced` and `md` execute and the rest queue.

Resume
------
State lives in ``manifest.json`` in the run directory. A stage is *done* when
its declared outputs exist; re-running skips those and picks up the rest. The
manifest also records the Slurm job id of anything submitted, so a second
invocation can tell "queued" from "never started" -- a distinction that
matters when a campaign spans days.

What this deliberately does not do
----------------------------------
It does not chain the solvers *physically*. Each stage is initialised from
the reduced chain's state, not from the previous solver's output, because the
scales do not meet: LAMMPS runs a 1 nm projectile and M2C a 1 mm one. They
are four independent simulations of the same *scenario*, not four parts of
one simulation, and `viz.composite` is written to say so on screen.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field

#: Stage order. Later stages may depend on earlier ones through the manifest,
#: but never through a physical handoff -- see the module docstring.
STAGE_ORDER = ("reduced", "md", "hydro", "mhd", "pic")

#: Human-facing description, used in reports and in the composite's captions.
STAGE_INFO = {
    "reduced": {"code": "hvi_emp", "scale": "analytic",
                "what": "shock, vaporisation, plume, EMP",
                "cost": "seconds"},
    "md": {"code": "LAMMPS", "scale": "nm",
           "what": "atomistic ejecta and fragments", "cost": "minutes"},
    "hydro": {"code": "M2C", "scale": "mm",
              "what": "crater, shock, coupled ionisation",
              "cost": "10-40 h per case"},
    "mhd": {"code": "OpenMHD", "scale": "m",
            "what": "magnetised plume, diamagnetic cavity",
            "cost": "hours"},
    "pic": {"code": "PIConGPU", "scale": "um",
            "what": "charge separation and the radiated pulse",
            "cost": "hours, after a compile"},
}


@dataclass
class StageResult:
    """What happened to one stage on one invocation."""
    name: str
    status: str                    # done | ran | submitted | blocked | skipped
    reason: str = ""
    outputs: list = field(default_factory=list)
    job_id: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in ("done", "ran", "submitted")

    def line(self) -> str:
        info = STAGE_INFO.get(self.name, {})
        mark = {"done": "[done]", "ran": "[ran ]", "submitted": "[sbmt]",
                "blocked": "[----]", "skipped": "[skip]"}.get(
                    self.status, "[????]")
        tail = f" job {self.job_id}" if self.job_id else ""
        return (f"  {mark} {self.name:<9}{info.get('code', ''):<11}"
                f"{self.reason}{tail}")


class Manifest:
    """Run state on disk, so a campaign can span days and reboots."""

    def __init__(self, path: str):
        self.path = path
        self.data = {"stages": {}, "created": time.time()}
        if os.path.isfile(path):
            try:
                with open(path) as fh:
                    self.data = json.load(fh)
            except (OSError, ValueError):
                # A corrupt manifest must not lose a multi-day campaign, so
                # keep the old file rather than overwriting it silently.
                bad = path + ".corrupt"
                shutil.copyfile(path, bad)
                self.data = {"stages": {}, "created": time.time(),
                             "note": f"previous manifest unreadable, kept "
                                     f"at {os.path.basename(bad)}"}

    def get(self, stage: str) -> dict:
        return self.data["stages"].get(stage, {})

    def set(self, stage: str, **kw) -> None:
        self.data["stages"].setdefault(stage, {}).update(kw)
        self.data["updated"] = time.time()
        self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".",
                    exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.path)          # atomic: never a half-written file


def outputs_exist(paths) -> bool:
    """True when every declared output is present and non-empty.

    Emptiness matters: a Slurm job killed at the wall clock leaves
    zero-length files behind, and treating those as "done" would skip the
    stage forever.
    """
    paths = list(paths)
    if not paths:
        return False
    return all(os.path.exists(p) and
               (os.path.isdir(p) or os.path.getsize(p) > 0) for p in paths)


def slurm_available() -> bool:
    return shutil.which("sbatch") is not None


def job_is_live(job_id: str | None) -> bool:
    """True if `job_id` is still queued or running.

    Without this, a re-run during a multi-hour job resubmits it.
    """
    if not job_id or not shutil.which("squeue"):
        return False
    try:
        out = subprocess.run(["squeue", "-h", "-j", str(job_id)],
                             capture_output=True, text=True, timeout=20)
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def submit_slurm(script: str, depends_on: str | None = None,
                 dry_run: bool = False) -> str | None:
    """sbatch a script, optionally after another job. Returns the job id."""
    cmd = ["sbatch", "--parsable"]
    if depends_on:
        cmd.append(f"--dependency=afterok:{depends_on}")
    cmd.append(script)
    if dry_run:
        return "DRYRUN"
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:      # pragma: no cover
        raise RuntimeError(f"sbatch failed: {exc}") from exc
    if out.returncode != 0:
        raise RuntimeError(f"sbatch rejected the job: {out.stderr.strip()}")
    return out.stdout.strip().split(";")[0]


def write_stage_script(path: str, body: str, job_name: str,
                       cores: int = 64, hours: int = 48,
                       gpus: int = 0, mem_gb: int = 64,
                       log_dir: str = "logs") -> str:
    """A self-contained sbatch script for one stage.

    Deliberately plain: the heavy stages are long-running MPI jobs whose
    resource needs differ by an order of magnitude, and a single templated
    directive block is easier to check by eye than a clever abstraction.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    gres = f"#SBATCH --gres=gpu:{gpus}\n" if gpus else ""
    text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}-%j.out
#SBATCH --error={log_dir}/{job_name}-%j.err
#SBATCH --time={hours:02d}:00:00
#SBATCH --nodes=1
#SBATCH --ntasks={cores}
#SBATCH --mem={mem_gb}G
{gres}
set -euo pipefail
mkdir -p {log_dir}
echo "host $(hostname)  start $(date -Is)"

{body}

echo "done $(date -Is)"
"""
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, 0o755)
    return path


def report(results) -> str:
    """A table a person can read at a glance, and act on."""
    lines = ["", "=" * 70, "full-chain status", "=" * 70]
    for r in results:
        lines.append(r.line())
    lines.append("")
    done = [r for r in results if r.status in ("done", "ran")]
    queued = [r for r in results if r.status == "submitted"]
    blocked = [r for r in results if r.status == "blocked"]
    lines.append(f"  {len(done)} complete, {len(queued)} queued, "
                 f"{len(blocked)} blocked")
    if queued:
        lines.append("")
        lines.append("  re-run this command when the jobs finish; completed "
                     "stages are skipped")
    if blocked:
        lines.append("")
        lines.append("  blocked stages need a solver installed -- see "
                     "docs/INSTALLING_SOLVERS.md")
    return "\n".join(lines)


__all__ = ["STAGE_ORDER", "STAGE_INFO", "StageResult", "Manifest",
           "outputs_exist", "slurm_available", "job_is_live", "submit_slurm",
           "write_stage_script", "report"]
