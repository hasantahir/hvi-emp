#!/usr/bin/env python3
"""Run the whole waterfall and render the composite animation.

**This is the one file to run.**

    python scripts/run_full_chain.py                  # what would happen
    python scripts/run_full_chain.py --run            # run what is cheap
    python scripts/run_full_chain.py --submit         # + queue the heavy ones
    python scripts/run_full_chain.py --render         # build the composite

It is safe to run repeatedly. Completed stages are skipped, running Slurm
jobs are left alone, and each invocation renders whatever chapters now have
data. A full campaign is days long, so this is designed to be run again
tomorrow rather than held open.

The stages
----------
    reduced   hvi_emp     analytic     seconds
    md        LAMMPS      nm, 100 ps   minutes
    hydro     M2C         mm, us       10-40 h per case      -> Slurm
    mhd       OpenMHD     m, ms        hours                 -> Slurm
    pic       PIConGPU    um, ns       hours + a compile     -> Slurm

A missing solver blocks only its own stage. On a fresh machine `reduced` and
`md` will run and the rest will report what to install.

About the composite
-------------------
`--render` produces one continuous zoom across all five. It looks like a
single simulation and it is not: LAMMPS runs a 1 nm projectile, M2C a 1 mm
one, so the handovers change *model*, not just magnification. The generated
scene burns that statement into every frame. See `hvi_emp/viz/composite.py`.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp import run_scenario
from hvi_emp.chain import (STAGE_INFO, STAGE_ORDER, Manifest, StageResult,
                           job_is_live, outputs_exist, report,
                           slurm_available, submit_slurm, write_stage_script)
from hvi_emp.viz.composite import build_chapters, write_composite_script


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_reduced(run_dir, sc, args, mani) -> StageResult:
    """The analytic chain, plus the impact- and plume-scale .vti scenes."""
    from hvi_emp.viz.paraview_scene import write_scene_for
    from hvi_emp.viz.vti import write_scene

    out = os.path.join(run_dir, "reduced")
    pvds = [os.path.join(out, s, f"{s}.pvd") for s in ("impact", "plume")]
    if outputs_exist(pvds) and not args.force:
        return StageResult("reduced", "done", "scenes already written",
                           outputs=pvds)
    if not args.run:
        return StageResult("reduced", "skipped", "pass --run")

    t0 = time.perf_counter()
    made = []
    for scene in ("impact", "plume"):
        res = write_scene(sc, os.path.join(out, scene), scene=scene,
                          n_frames=args.frames, quality=args.quality,
                          verbose=False)
        write_scene_for(res, os.path.join(out, scene, f"scene_{scene}.py"),
                        preset="talk")
        made.append(res["pvd"])
    return StageResult("reduced", "ran", f"{len(made)} scenes", outputs=made,
                       seconds=time.perf_counter() - t0)


def _md_config(args, dump):
    """The MD configuration, from a named scale with per-flag overrides."""
    from hvi_emp.solvers.lammps_stage1 import config_for_scale

    over = {"dump_every": args.md_dump_every, "dump_path": dump}
    if args.md_cells:
        n = args.md_cells
        over["target_cells"] = (n, n, int(n * 0.7))
    if args.md_radius:
        over["projectile_radius"] = args.md_radius
    if args.md_ps:
        over["t_production"] = args.md_ps * 1e-12
    return config_for_scale(args.md_scale, material=args.md_material,
                            velocity=args.md_velocity, **over)


def stage_md(run_dir, sc, args, mani) -> StageResult:
    """LAMMPS.

    Small runs go in-process; anything over `--md-local-minutes` is submitted
    instead, because a 10-hour MD run should not be holding a terminal open.
    """
    from hvi_emp.solvers import have_lammps
    from hvi_emp.solvers.lammps_postprocess import convert_dump

    out = os.path.join(run_dir, "md")
    os.makedirs(out, exist_ok=True)
    pvd = os.path.join(out, "md_atoms.pvd")
    dump = os.path.join(out, "impact.dump")
    cfg = _md_config(args, dump)
    est = cfg.estimate_cost(cores=args.cores)
    size = (f"{est['atoms']/1e6:.2f}M atoms, "
            f"~{est['wall_hours']:.2f} h on {args.cores} cores")

    if outputs_exist([pvd]) and not args.force:
        return StageResult("md", "done", f"already converted ({size})",
                           outputs=[pvd])
    if not have_lammps():
        return StageResult("md", "blocked",
                           "lammps not importable -- pip install lammps mpich")

    prev = mani.get("md")
    if job_is_live(prev.get("job_id")):
        return StageResult("md", "submitted", "still running",
                           job_id=prev.get("job_id"))

    # already run under Slurm, just needs converting?
    if os.path.isfile(dump) and os.path.getsize(dump) > 0:
        res = convert_dump(dump, out, material=args.md_material,
                           fragments=True, name="md", verbose=False)
        return StageResult("md", "ran", f"converted {len(res['times'])} "
                           f"frames ({size})", outputs=[res["atoms_pvd"]])

    heavy = est["wall_hours"] * 60.0 > args.md_local_minutes
    if heavy:
        if not args.submit:
            return StageResult("md", "skipped",
                               f"{size}; pass --submit, or use "
                               f"--md-scale smoke to run here")
        if not slurm_available():
            return StageResult("md", "blocked",
                               f"{size} is too big to run in-process and "
                               f"there is no sbatch")
        from hvi_emp.solvers.lammps_stage1 import generate_input_deck
        deck = os.path.join(out, "in.impact")
        with open(deck, "w") as fh:
            fh.write(generate_input_deck(cfg))
        body = (f"cd {out}\n"
                f"mpirun -np {args.cores} lmp -in in.impact")
        script = write_stage_script(
            os.path.join(out, "md.sbatch"), body, job_name="hvi-md",
            cores=args.cores, hours=max(2, int(est["wall_hours"] * 2) + 1),
            mem_gb=max(16, int(est["memory_GB"] * 3) + 8),
            log_dir=os.path.join(run_dir, "logs"))
        job = submit_slurm(script, dry_run=args.dry_run)
        mani.set("md", job_id=job, script=script, submitted=time.time())
        return StageResult("md", "submitted", size, job_id=job)

    if not args.run:
        return StageResult("md", "skipped", f"{size}; pass --run")

    t0 = time.perf_counter()
    from hvi_emp.solvers.lammps_stage1 import run_impact
    run_impact(cfg, screen=False)
    res = convert_dump(dump, out, material=args.md_material, fragments=True,
                       name="md", verbose=False)
    return StageResult("md", "ran",
                       f"{len(res['times'])} frames ({size})",
                       outputs=[res["atoms_pvd"]],
                       seconds=time.perf_counter() - t0)


def _solver_stage(name, run_dir, sc, args, mani, *, deck_fn, found_fn,
                  run_body, cores, hours, gpus=0, mem_gb=64,
                  output_glob=None) -> StageResult:
    """Shared shape for the three expensive, externally-run solvers.

    All three follow the same life: write a deck (always possible), check the
    solver exists, submit to Slurm, and on a later invocation notice the
    output. The differences are the deck writer and the run command, so those
    are passed in rather than duplicated three times.
    """
    out = os.path.join(run_dir, name)
    os.makedirs(out, exist_ok=True)
    deck = deck_fn(out)                       # always write the deck

    produced = output_glob(out) if output_glob else []
    if produced and outputs_exist(produced) and not args.force:
        return StageResult(name, "done", "output present", outputs=produced)

    prev = mani.get(name)
    if job_is_live(prev.get("job_id")):
        return StageResult(name, "submitted", "still running",
                           job_id=prev.get("job_id"))

    found = found_fn()
    if not found:
        info = STAGE_INFO[name]
        return StageResult(name, "blocked",
                           f"{info['code']} not found -- deck written to "
                           f"{os.path.relpath(out, run_dir)}/")
    if not args.submit:
        return StageResult(name, "skipped",
                           f"deck ready; pass --submit ({STAGE_INFO[name]['cost']})")
    if not slurm_available():
        return StageResult(name, "blocked",
                           "no sbatch on PATH; run the deck by hand")

    script = write_stage_script(
        os.path.join(out, f"{name}.sbatch"), run_body(out, deck),
        job_name=f"hvi-{name}", cores=cores, hours=hours, gpus=gpus,
        mem_gb=mem_gb, log_dir=os.path.join(run_dir, "logs"))
    job = submit_slurm(script, dry_run=args.dry_run)
    mani.set(name, job_id=job, script=script, submitted=time.time())
    return StageResult(name, "submitted", f"{hours} h, {cores} cores",
                       job_id=job)


def stage_hydro(run_dir, sc, args, mani) -> StageResult:
    from hvi_emp.solvers.m2c_stage1 import (M2CConfig, find_m2c,
                                            write_problem_directory)

    def deck(out):
        cfg = M2CConfig(projectile=sc.impact.projectile.material,
                        target=sc.impact.target,
                        diameter=2 * sc.impact.projectile.radius,
                        velocity=sc.impact.projectile.velocity,
                        angle_deg=args.angle)
        write_problem_directory(cfg, out, cores=args.cores)
        return os.path.join(out, "input.st")

    return _solver_stage(
        "hydro", run_dir, sc, args, mani,
        deck_fn=deck, found_fn=find_m2c,
        run_body=lambda out, d: (f"cd {out}\n"
                                 f"mpirun -np {args.cores} "
                                 f"${M2C_HOME:?set it to the directory holding the m2c binary}/m2c input.st"),
        cores=args.cores, hours=48, mem_gb=128,
        # Files inside results/, not the directory itself. The deck writer
        # now creates results/ up front (M2C does not mkdir it and dies at
        # the first output step without it), so testing for the directory
        # would report a stage as "done" the moment its deck was written.
        output_glob=lambda out: sorted(
            glob.glob(os.path.join(out, "results", "*"))))


def stage_mhd(run_dir, sc, args, mani) -> StageResult:
    from hvi_emp.solvers.openmhd_stage2 import write_fletcher_mhd_cases

    def deck(out):
        write_fletcher_mhd_cases(out, exp=sc.expansion)
        return out

    return _solver_stage(
        "mhd", run_dir, sc, args, mani,
        deck_fn=deck,
        found_fn=lambda: os.environ.get("OPENMHD"),
        run_body=lambda out, d: (f"cd {out}/fig7_perpendicular\n"
                                 f"make && mpirun -np {min(args.cores, 32)} "
                                 f"./openmhd"),
        cores=min(args.cores, 32), hours=12,
        output_glob=lambda out: [os.path.join(out, "fig7_perpendicular",
                                              "data")])


def stage_pic(run_dir, sc, args, mani) -> StageResult:
    from hvi_emp.solvers.picongpu_stage3 import (PIConGPUConfig,
                                                 write_input_set)

    def deck(out):
        cfg = PIConGPUConfig.from_expansion(sc.expansion)
        write_input_set(cfg, out)
        return out

    return _solver_stage(
        "pic", run_dir, sc, args, mani,
        deck_fn=deck,
        found_fn=lambda: os.environ.get("PICSRC"),
        run_body=lambda out, d: (f"cd {out}\n"
                                 f"pic-build\n"
                                 f"tbg -s bash -c etc/picongpu/N.cfg "
                                 f"-t etc/picongpu/bash/mpiexec.tpl run"),
        cores=min(args.cores, 16), hours=24, gpus=1,
        output_glob=lambda out: [os.path.join(out, "simOutput")])


STAGES = {"reduced": stage_reduced, "md": stage_md, "hydro": stage_hydro,
          "mhd": stage_mhd, "pic": stage_pic}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def scheduler_warning(args) -> str | None:
    """The `--submit` with no scheduler warning, or None if not applicable.

    Separated from ``main`` so it can be tested without running a scenario.

    Why it exists: on a standalone workstation ``--submit`` was accepted
    silently, every heavy stage reported "blocked", and the real cause
    appeared only as one clause inside a per-stage line near the bottom.
    It read as "the solvers are missing" when the actual problem was
    "there is nowhere to submit to". Those need different fixes, so the
    distinction has to be made loudly and early.
    """
    if not getattr(args, "submit", False) or slurm_available():
        return None
    hours = args.md_local_minutes / 60.0
    bar = "!" * 70
    return (
        f"\n{bar}\n"
        "--submit was given, but there is no `sbatch` on PATH.\n"
        "This machine is not a Slurm submit host, so NOTHING can be\n"
        "queued and every heavy stage below will report as blocked.\n"
        "This is NOT the same as the solvers being missing.\n"
        "\n"
        "  To run the MD stage here instead, raise the local budget:\n"
        f"      --md-local-minutes 600      (currently "
        f"{args.md_local_minutes:.0f} min = {hours:.1f} h)\n"
        "  The solver stages write their decks either way; run them by\n"
        "  hand from the run directory.\n"
        "  If this machine has a scheduler under another name (qsub,\n"
        "  bsub), the decks are portable -- wrap them yourself.\n"
        f"{bar}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="full_chain")
    ap.add_argument("--projectile", default="Fe")
    ap.add_argument("--target", default="Al")
    ap.add_argument("--mass", type=float, default=1e-12)
    ap.add_argument("--velocity", type=float, default=50e3)
    ap.add_argument("--angle", type=float, default=0.0)
    ap.add_argument("--run", action="store_true",
                    help="run the cheap stages (reduced, md)")
    ap.add_argument("--submit", action="store_true",
                    help="also queue the expensive stages on Slurm")
    ap.add_argument("--render", action="store_true",
                    help="build the composite from whatever has data")
    ap.add_argument("--force", action="store_true",
                    help="redo stages even if their output exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="write sbatch scripts but do not submit")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of " + ",".join(STAGE_ORDER))
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--quality", default="talk")
    ap.add_argument("--md-scale", default="talk",
                    help="smoke|talk|detailed|paper|fraile; --list-scales")
    ap.add_argument("--list-scales", action="store_true",
                    help="MD sizes with measured-rate cost estimates")
    ap.add_argument("--md-material", default="W")
    ap.add_argument("--md-velocity", type=float, default=9e3)
    ap.add_argument("--md-radius", type=float, default=None,
                    help="override the scale's projectile radius [m]")
    ap.add_argument("--md-cells", type=int, default=None,
                    help="override the scale's target block edge")
    ap.add_argument("--md-ps", type=float, default=None,
                    help="override the scale's production time [ps]")
    ap.add_argument("--md-dump-every", type=int, default=2000)
    ap.add_argument("--md-local-minutes", type=float, default=20.0,
                    help="MD longer than this is submitted, not run here")
    ap.add_argument("--movie", default=None,
                    help="frame pattern for the composite, "
                         "e.g. frames/composite.%%04d.png")
    args = ap.parse_args(argv)

    if args.list_scales:
        from hvi_emp.solvers.lammps_stage1 import (ATOM_STEPS_PER_SEC_CORE,
                                                   scale_table)
        print(f"{'scale':<10}{'cells':>7}{'atoms':>10}{'steps':>9}"
              f"{'RAM GB':>8}{'hours':>8}   what it is for")
        for r in scale_table(cores=args.cores, material=args.md_material):
            print(f"{r['scale']:<10}{r['cells']:>7}{r['atoms']/1e6:>9.2f}M"
                  f"{r['steps']:>9.3g}{r['memory_GB']:>8.1f}"
                  f"{r['wall_hours']:>8.2f}   {r['for']}")
        print(f"\nhours are for {args.cores} cores at "
              f"{ATOM_STEPS_PER_SEC_CORE:.2g} atom-steps/s/core -- MEASURED "
              f"on one core,")
        print("times an ASSUMED 0.85 MPI efficiency. The rate was measured "
              "on aarch64,")
        print("so it is a floor for a modern x86 core, not a target.")
        return 0

    # --submit implies --run: someone queueing a multi-day campaign wants the
    # seconds-long stages done too, and being told "pass --run" after asking
    # for everything is a foot-gun, not a safeguard.
    if args.submit:
        args.run = True

    run_dir = os.path.abspath(args.out)
    os.makedirs(run_dir, exist_ok=True)
    mani = Manifest(os.path.join(run_dir, "manifest.json"))

    print("=" * 70)
    print(f"{args.projectile} {args.mass:.2e} kg at {args.velocity/1e3:.0f} "
          f"km/s -> {args.target}   ({args.angle:.0f} deg)")
    print(f"run directory: {run_dir}")
    if not (args.run or args.submit or args.render):
        print("\nDRY: nothing will run. Add --run, --submit or --render.")

    warning = scheduler_warning(args)
    if warning:
        print(warning)
    print("=" * 70)

    sc = run_scenario(args.projectile, args.target, mass=args.mass,
                      velocity=args.velocity, angle_deg=args.angle,
                      t_end=1e-5)

    wanted = ([s.strip() for s in args.only.split(",")] if args.only
              else list(STAGE_ORDER))
    results = []
    for name in STAGE_ORDER:
        if name not in wanted:
            continue
        try:
            r = STAGES[name](run_dir, sc, args, mani)
        except Exception as exc:
            # One solver's bad day must not lose the other four stages.
            r = StageResult(name, "blocked",
                            f"{type(exc).__name__}: {exc}"[:90])
        results.append(r)
        mani.set(name, status=r.status, reason=r.reason,
                 outputs=r.outputs, seconds=r.seconds)

    print(report(results))

    # ------------------------------------------------------------ composite
    pvds = {}
    for name in STAGE_ORDER:
        for cand in (os.path.join(run_dir, name, f"{name}.pvd"),
                     os.path.join(run_dir, name, "md_atoms.pvd"),
                     os.path.join(run_dir, name, "impact", "impact.pvd")):
            if os.path.isfile(cand):
                pvds[name] = cand
                break
    if "reduced" not in pvds:
        cand = os.path.join(run_dir, "reduced", "plume", "plume.pvd")
        if os.path.isfile(cand):
            pvds["reduced"] = cand

    chapters = build_chapters(pvds, frames=args.frames)
    have = [c for c in chapters if c.available]
    print(f"\n  composite: {len(have)}/{len(chapters)} chapters have data")
    for c in chapters:
        mark = "  +" if c.available else "  -"
        print(f"{mark} {c.stage:<9}{c.code:<11}{c.banner() if c.available else 'no output yet'}")

    if args.render:
        if not have:
            print("\n  nothing to render yet -- run the stages first")
            return 1
        script = os.path.join(run_dir, "composite.py")
        write_composite_script(chapters, script, movie_path=args.movie,
                               font=24)
        print(f"\n  wrote {script}")
        print(f"  pvbatch {script}")
        print("\n  THE COMPOSITE IS NOT ONE SIMULATION. It joins independent")
        print("  runs at different scales with different projectile sizes;")
        print("  the handovers change model, not just magnification. That")
        print("  statement is burnt into every frame of the output.")
    else:
        print("\n  add --render to write the composite scene")

    print(f"\n  state: {os.path.join(run_dir, 'manifest.json')}")
    print("  re-run this command as jobs finish; finished stages are skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
