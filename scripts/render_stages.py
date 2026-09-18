#!/usr/bin/env python3
"""Render each stage as its own animation, rather than one composite.

    python scripts/render_stages.py --run-dir full_chain
    python scripts/render_stages.py --run-dir full_chain --emp

The composite exists to show the whole chain in one shot, and it pays for
that with a caveat burnt into every frame: the chapters are different codes
at different scales with different projectile sizes, so the handovers change
model rather than magnification. For anything you intend to *show* -- a
talk, a figure, a referee -- separate animations are the better artefact.
Nothing has to be disclaimed, because nothing is being joined.

This writes one ParaView script per stage:

    impact   crater and shock, ~20 um box, sub-nanosecond
    plume    expansion into vacuum, ~400 mm box, microseconds
    md       LAMMPS atoms, ejecta and fragments, ~25 nm box, picoseconds
    emp      the radiated far field, metre scale (with --emp)

The first three read data the chain has already written. The fourth is
reconstructed, and is the one to read the caveats on -- see
`hvi_emp/viz/emp_field.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

GRN, RED, YLW, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def ok(m):
    print(f"  {GRN}[ ok ]{RST} {m}")


def warn(m):
    print(f"  {YLW}[warn]{RST} {m}")


def bad(m):
    print(f"  {RED}[FAIL]{RST} {m}")


def head(m):
    print(f"\n{m}\n" + "=" * 68)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(_REPO / "full_chain"))
    ap.add_argument("--emp", action="store_true",
                    help="also reconstruct and write the EM far field")
    ap.add_argument("--emp-frames", type=int, default=48)
    ap.add_argument("--emp-shape", type=int, default=96)
    ap.add_argument("--projectile", default="Fe")
    ap.add_argument("--target", default="Al")
    ap.add_argument("--mass", type=float, default=1e-12)
    ap.add_argument("--velocity", type=float, default=50e3)
    ap.add_argument("--movie", action="store_true",
                    help="write a frame pattern into each stage directory")
    args = ap.parse_args(argv)

    root = Path(args.run_dir)
    if not root.is_dir():
        bad(f"no such run directory: {root}")
        print("  run the chain first:")
        print("    python scripts/run_full_chain.py --md-scale talk --run")
        return 2

    from hvi_emp.viz.paraview_scene import (write_particle_scene_script,
                                            write_scene_script)

    head(f"Stages in {root}")
    made = []

    # --- the .vti scenes the reduced chain writes -------------------------
    for scene in ("impact", "plume"):
        d = root / "reduced" / scene
        pvd = d / f"{scene}.pvd"
        script = d / f"scene_{scene}.py"
        if not pvd.is_file():
            warn(f"{scene}: no {pvd.name} -- stage has not run")
            continue
        if script.is_file():
            ok(f"{scene}: {script.relative_to(root)} (already written)")
        else:
            write_scene_script(str(pvd), str(script), preset="talk",
                               movie_path=(str(d / f"{scene}.%04d.png")
                                           if args.movie else None))
            ok(f"{scene}: wrote {script.relative_to(root)}")
        made.append(script)

    # --- MD atoms --------------------------------------------------------
    md_pvd = root / "md" / "md_atoms.pvd"
    if md_pvd.is_file():
        script = root / "md" / "scene_md.py"
        try:
            grid = root / "md" / "md_grid.pvd"
            write_particle_scene_script(
                str(md_pvd), str(script),
                grid_pvd=str(grid) if grid.is_file() else None,
                movie_path=(str(root / "md" / "md.%04d.png") if args.movie
                            else None))
            ok(f"md: wrote {script.relative_to(root)}")
            made.append(script)
        except Exception as exc:                            # noqa: BLE001
            warn(f"md: {type(exc).__name__}: {exc}")
    else:
        warn("md: no md_atoms.pvd -- run with --md-scale talk --run")

    # --- the EM far field ------------------------------------------------
    if args.emp:
        head("EM radiation")
        print("  This is RECONSTRUCTED from the dipole model, not simulated")
        print("  on a grid. The EMP stage produces a far-field time series at")
        print("  one sensor; the field elsewhere follows exactly from it for")
        print("  a dipole, which is the geometry the model already assumes.")
        print()
        from hvi_emp import run_scenario
        from hvi_emp.viz.emp_field import pulse_figure, wavefront_series

        sc = run_scenario(args.projectile, args.target, mass=args.mass,
                          velocity=args.velocity, t_end=1e-5)
        d = root / "emp"
        fig = pulse_figure(sc, str(d / "emp_pulse.png"))
        if fig:
            ok(f"pulse + spectrum: {Path(fig).relative_to(root)}")
        else:
            warn("matplotlib absent -- skipping the pulse figure "
                 "(pip install matplotlib)")

        n = args.emp_shape
        res = wavefront_series(sc, str(d / "wave"), n_frames=args.emp_frames,
                               shape=(n, n, n), verbose=True)
        script = d / "wave" / "scene_emp.py"
        # show_target=False and clip=False: there is no target surface at
        # metre scale, and clipping a spherical shell hides the thing the
        # animation exists to show.
        write_scene_script(res["pvd"], str(script), preset="talk",
                           colour_by="E_normalised", show_target=False,
                           clip=False,
                           title="EM far field (reconstructed dipole)",
                           movie_path=(str(d / "wave" / "emp.%04d.png")
                                       if args.movie else None))
        ok(f"wavefront: {script.relative_to(root)}")
        made.append(script)

    # ---------------------------------------------------------------------
    head("Render")
    if not made:
        bad("nothing to render")
        return 1
    for s in made:
        print(f"  pvbatch --force-offscreen-rendering {s}")
    print()
    print("  Each of these is ONE code at ONE scale. Unlike the composite,")
    print("  nothing here needs a caveat about joined simulations -- but the")
    print("  EMP wavefront still carries the 184x amplitude bracket, and its")
    print("  box is metres where the plume is millimetres.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
