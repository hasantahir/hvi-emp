"""Config-driven entry point: ``python -m hvi_emp.run``.

This is what a job script and a Snakemake rule both call. It does one thing
per invocation and writes a self-describing result, which is what makes it
composable by an orchestrator.

    python -m hvi_emp.run                                  # one scenario
    python -m hvi_emp.run scenario=fletcher_w_al           # swap a group
    python -m hvi_emp.run scenario.velocity=62e3           # override a field
    python -m hvi_emp.run sweep=velocity                   # whole sweep here
    python -m hvi_emp.run sweep=velocity --sweep-index 3   # one point of it
    python -m hvi_emp.run sweep=velocity --submit          # write an sbatch
    python -m hvi_emp.run --print-config                   # resolve and stop

``--sweep-index`` is the array-job hook: the same command with a different
index runs a different point, so a Slurm array and a Snakemake DAG drive it
identically.

What lands on disk
------------------
Every run writes, next to its results:

* ``config.yaml`` -- the fully resolved configuration, so the run can be
  repeated exactly. A results file without this is an orphan.
* ``result.json`` -- scalars, keyed by name, with units in the key.
* ``manifest.json`` -- code version, hostname, job id, timings, and the
  library versions that produced it.

Nothing here catches exceptions to keep going. A failed scenario must exit
non-zero so the scheduler and the orchestrator both see it; a sweep that
quietly records zero for a failure is indistinguishable from a real result.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

from .config import (CONF_DIR, config_to_yaml, load_config, to_run_spec)
from .schema import ConfigError

__all__ = ["main", "run_one", "run_sweep", "write_outputs"]


def _versions() -> dict:
    import scipy
    from . import __version__ as ver
    from .accel import cupy_available
    from .schema import USING_PYDANTIC
    from .config import USING_HYDRA
    return {"hvi_emp": ver, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "pydantic": USING_PYDANTIC, "hydra": USING_HYDRA,
            "cupy": cupy_available()}


def _job_environment() -> dict:
    """Scheduler identifiers, so a result can be traced back to its job."""
    keys = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
            "SLURM_CPUS_PER_TASK", "SLURM_JOB_PARTITION", "HVI_EMP_WORKERS")
    return {k: os.environ[k] for k in keys if k in os.environ}


def run_one(scenario: dict) -> dict:
    """Run one scenario and reduce it to scalars.

    Returns plain floats only. The full `Scenario` holds megabytes of arrays
    per run; a sweep of forty of those pickled across a process boundary is
    hundreds of megabytes for six numbers.
    """
    from .pipeline import run_scenario

    kw = dict(scenario)
    pm, tm = kw.pop("projectile"), kw.pop("target")
    # `diameter` is an input spelling for `mass`; the schema has already
    # converted it, so it must not be forwarded as a keyword.
    diam = kw.pop("diameter", None)
    mass, vel = kw.pop("mass"), kw.pop("velocity")
    angle = kw.pop("angle_deg", 0.0)
    sc = run_scenario(pm, tm, mass=mass, velocity=vel, angle_deg=angle, **kw)

    band = sc.bands.get(916e6, {}) if sc.bands else {}
    out = {
        "projectile": pm, "target": tm,
        "mass_kg": float(mass), "velocity_m_s": float(vel),
        "diameter_m": float(diam) if diam else None,
        "angle_deg": float(angle),
        "expansion_model": kw.get("expansion_model", "1T"),
        "peak_pressure_GPa": float(sc.impact.P_ic) / 1e9,
        "vapour_mass_kg": float(sc.impact.m_vapour),
        "plasma_mass_kg": float(sc.impact.m_plasma),
        "T_e_formation_eV": float(sc.impact.plasma.T_eV),
        "Zbar_formation": float(sc.impact.plasma.Zbar),
        "Q_formation_C": float(np.atleast_1d(sc.impact.Q_free)[-1]),
        "warnings": list(sc.warnings),
    }
    if sc.expansion is not None:
        out.update({
            "Q_frozen_C": float(sc.expansion.Q_final),
            "v_z_asymptotic_m_s": float(sc.expansion.v_z[-1]),
            "n_e_peak_m3": float(np.max(sc.expansion.n_e)),
        })
    if band.get("reached"):
        out["E_916MHz_V_m"] = float(band["E_peak"])
    return out


def run_sweep(run, index: int | None = None) -> list:
    """Run a sweep, or just the `index`-th point of it.

    A single index is the array-job path: N independent invocations, each
    doing one point, so one failure costs one point.
    """
    from .parallel import cpu_count, parallel_map

    values = run.sweep.resolve()
    param = run.sweep.parameter
    base = dict(run.scenario.model_dump())
    if param not in base:
        raise ConfigError(
            f"sweep parameter {param!r} is not a scenario field. Known: "
            f"{sorted(base)}")

    scenarios = []
    for v in values:
        s = dict(base)
        s[param] = v
        scenarios.append(s)

    if index is not None:
        if not (0 <= index < len(scenarios)):
            raise ConfigError(
                f"--sweep-index {index} is outside 0..{len(scenarios) - 1} "
                f"for this sweep ({len(scenarios)} points). An array sized "
                f"differently from the sweep is the usual cause.")
        return [run_one(scenarios[index])]

    res = parallel_map(run_one, scenarios, workers=cpu_count(),
                       on_error="raise")
    return list(res)


def write_outputs(results: list, cfg: dict, out_dir: Path,
                  elapsed: float) -> dict:
    """Write config.yaml, result.json and manifest.json. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    paths["config"] = out_dir / "config.yaml"
    paths["config"].write_text(config_to_yaml(cfg))

    paths["result"] = out_dir / "result.json"
    payload = results[0] if len(results) == 1 else results
    paths["result"].write_text(json.dumps(payload, indent=2, sort_keys=True))

    paths["manifest"] = out_dir / "manifest.json"
    paths["manifest"].write_text(json.dumps({
        "n_results": len(results),
        "elapsed_s": round(elapsed, 3),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime(time.time() - elapsed)),
        "versions": _versions(),
        "job": _job_environment(),
        "argv": sys.argv,
    }, indent=2, sort_keys=True))
    return {k: str(v) for k, v in paths.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m hvi_emp.run",
        description="Run hvi_emp from the configuration tree.",
        epilog="Any remaining arguments are Hydra-style overrides, e.g. "
               "scenario=fletcher_w_al scenario.velocity=62e3")
    ap.add_argument("--config-dir", default=None,
                    help=f"configuration tree (default: {CONF_DIR})")
    ap.add_argument("--config-name", default="config")
    ap.add_argument("--output-dir", default=None,
                    help="override output_dir from the config")
    ap.add_argument("--sweep-index", type=int, default=None,
                    help="run only this point of the sweep (array-job hook)")
    ap.add_argument("--print-config", action="store_true",
                    help="resolve, validate and print the config, then stop")
    ap.add_argument("--submit", action="store_true",
                    help="write a scheduler script instead of running")
    ap.add_argument("--submit-now", action="store_true",
                    help="with --submit, actually queue the job")
    ap.add_argument("--script", default=None,
                    help="path for the generated job script")
    ap.add_argument("overrides", nargs="*", help=argparse.SUPPRESS)
    # parse_known_args, not parse_args: overrides are free-form `a.b=c`
    # tokens that may appear anywhere on the line, and argparse cannot
    # interleave a `nargs="*"` positional with optionals. Anything left over
    # must look like an override, or it is a typo worth reporting rather
    # than ignoring.
    args, extra = ap.parse_known_args(argv)
    stray = [e for e in extra if "=" not in e]
    if stray:
        ap.error(f"unrecognised argument(s): {stray}")
    args.overrides = list(args.overrides) + [e for e in extra if "=" in e]

    conf_dir = Path(args.config_dir) if args.config_dir else None
    try:
        cfg = load_config(args.overrides, conf_dir=conf_dir,
                          config_name=args.config_name)
    except ConfigError as exc:
        print(f"configuration error:\n  {exc}", file=sys.stderr)
        return 2

    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    run = to_run_spec(cfg)

    if args.print_config:
        print(config_to_yaml(cfg))
        n = len(run.sweep.resolve()) if run.sweep else 1
        print(f"# resolves to {n} scenario(s)")
        return 0

    if args.submit:
        from .cluster import (estimate_memory_mb, estimate_walltime, submit)
        n = len(run.sweep.resolve()) if run.sweep else 1
        if run.cluster.time == "01:00:00":      # the schema default
            run.cluster.time = estimate_walltime(
                1 if run.sweep else n, run.scenario.expansion_model,
                workers=run.cluster.cpus_per_task)
        script = Path(args.script or
                      f"{cfg['output_dir']}/{cfg['tag']}/submit.sh")
        info = submit(run, script, dry_run=not args.submit_now,
                      conf_overrides=args.overrides)
        print(f"script     {info['script']}")
        print(f"array size {n}")
        print(f"walltime   {run.cluster.time}")
        print(f"memory     ~{estimate_memory_mb(run.cluster.cpus_per_task, run.scenario.expansion_model)} MB")
        print(f"command    {info['command']}")
        if not info["submitted"]:
            print(f"\n{info.get('note', '')}")
        else:
            print(f"job id     {info.get('job_id', '?')}")
        return 0

    t0 = time.time()
    try:
        if run.sweep:
            results = run_sweep(run, index=args.sweep_index)
            suffix = (f"/point_{args.sweep_index:04d}"
                      if args.sweep_index is not None else "")
        else:
            if args.sweep_index is not None:
                print("configuration error:\n  --sweep-index given but no "
                      "sweep is configured; add `sweep=<name>`.",
                      file=sys.stderr)
                return 2
            results = [run_one(dict(run.scenario.model_dump()))]
            suffix = ""
    except ConfigError as exc:
        # A misconfigured array is a user error, not a crash. Exit 2 so the
        # scheduler records a failure without a traceback burying the reason.
        print(f"configuration error:\n  {exc}", file=sys.stderr)
        return 2
    elapsed = time.time() - t0

    out_dir = Path(cfg["output_dir"]) / cfg["tag"]
    paths = write_outputs(results, cfg, Path(str(out_dir) + suffix), elapsed)

    print(f"{len(results)} scenario(s) in {elapsed:.1f}s -> {paths['result']}")
    for r in results:
        for w in r.get("warnings", []):
            print(f"  ! {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
