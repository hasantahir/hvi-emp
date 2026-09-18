#!/usr/bin/env python3
"""Rebuild a missing .pvd from frames that are already on disk.

    python scripts/repair_pvd.py full_chain --dry-run
    python scripts/repair_pvd.py full_chain

A `.pvd` is an index: a list of frame files and the physical time of each.
The frames are the expensive part; the index is a few hundred bytes of XML.
When a stage is interrupted between writing its frames and writing its
index -- Ctrl-C, a walltime kill, a full disk -- the result is a directory
of perfectly good `.vti`/`.vtp` files that ParaView will not load, because
nothing tells it they form a series.

Re-running the solver to recover a few hundred bytes is the wrong trade.
This reconstructs the index instead.

**Times are recomputed, not invented.** They come from the same call the
writer used, `ImpactVolume.times(n, mode=...)`, which is a pure function of
the scenario, the scene and the frame count. Pass the scenario this run
used; if it does not match, the frame count check fails loudly rather than
writing plausible-looking nonsense.

Where the times genuinely cannot be recovered -- an unrecognised directory,
or a scenario that does not reproduce the frame count -- the fallback is
uniform indices, and the .pvd says so in a comment. Geometry renders
correctly either way; only the time labels and any time-based colouring
would be wrong, so it is worth knowing which you have.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRN, RED, YLW, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

#: Directory name -> the scene key `write_scene` used. The frame files are
#: named after the scene, so the directory tells us how the times were built.
SCENE_BY_NAME = {"impact": "impact", "plume": "plume",
                 "plume_comoving": "plume_comoving"}


def frames_in(d: Path) -> tuple[str, list[Path]]:
    """(prefix, sorted frames) for a directory of numbered VTK files."""
    files = sorted([p for p in d.iterdir()
                    if p.suffix in (".vti", ".vtp") and re.search(r"_\d+$",
                                                                  p.stem)])
    if not files:
        return "", []
    prefix = re.sub(r"_\d+$", "", files[0].stem)
    files = [p for p in files if re.sub(r"_\d+$", "", p.stem) == prefix]
    return prefix, files


def recompute_times(scene: str, n: int, args) -> tuple[list, str]:
    """The exact times the writer used, or None with a reason."""
    try:
        from hvi_emp import run_scenario
        from hvi_emp.viz.vti import SCENES, ImpactVolume
    except Exception as exc:                                # noqa: BLE001
        return None, f"hvi_emp not importable ({type(exc).__name__})"

    if scene not in SCENES:
        return None, f"unknown scene {scene!r}"
    try:
        from hvi_emp.viz.vti import scene_kwargs

        sc = run_scenario(args.projectile, args.target, mass=args.mass,
                          velocity=args.velocity, t_end=1e-5)
        spec = SCENES[scene]
        vol = ImpactVolume(sc, **scene_kwargs(sc, scene))

        # The impact scene CLAMPS its end time, and skipping that clamp is
        # not a small error: the first version of this returned 1e-5 s where
        # the writer used 9.22e-9 s, a factor of 1000. The frames would have
        # rendered with a time axis that was silently wrong -- the exact
        # plausible-looking nonsense this tool is supposed to refuse to
        # produce. Mirrors write_scene().
        t_end = None
        if scene == "impact":
            t_shock = 3.0 * max(vol.shock_transit_time(),
                                vol.crater_growth_time())
            t_end = min(t_shock, vol.plume_departure_time())

        times = vol.times(n, mode=spec["time_mode"], t_end=t_end)
        return list(times), "recomputed from the scenario"
    except Exception as exc:                                # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default="full_chain")
    ap.add_argument("--dry-run", "-n", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rewrite a .pvd even if one exists")
    ap.add_argument("--projectile", default="Fe")
    ap.add_argument("--target", default="Al")
    ap.add_argument("--mass", type=float, default=1e-12)
    ap.add_argument("--velocity", type=float, default=50e3)
    args = ap.parse_args(argv)

    root = Path(args.run_dir)
    if not root.is_dir():
        print(f"{RED}no such directory: {root}{RST}")
        return 2

    from hvi_emp.viz.vti import write_pvd

    print(f"scanning {root}")
    print(f"scenario for time recovery: {args.projectile} -> {args.target}, "
          f"{args.mass:.3g} kg at {args.velocity/1e3:.0f} km/s")
    print("  (pass --projectile/--target/--mass/--velocity if this run "
          "used different values)\n")

    candidates, repaired, skipped = [], 0, 0
    for d in sorted({p.parent for p in root.rglob("*.vti")}
                    | {p.parent for p in root.rglob("*.vtp")}):
        prefix, files = frames_in(d)
        if not files:
            continue
        pvd = d / f"{prefix}.pvd"
        if pvd.is_file() and not args.force:
            print(f"  {GRN}[ ok ]{RST} {d.relative_to(root)}  "
                  f"{prefix}.pvd already present ({len(files)} frames)")
            skipped += 1
            continue
        candidates.append((d, prefix, files, pvd))

    if not candidates:
        print(f"\n{GRN}nothing to repair{RST} -- "
              f"{skipped} collection(s) already indexed")
        return 0

    for d, prefix, files, pvd in candidates:
        rel = d.relative_to(root)
        scene = SCENE_BY_NAME.get(d.name)
        n = len(files)
        times, how = (recompute_times(scene, n, args) if scene
                      else (None, f"directory {d.name!r} is not a known scene"))

        if times is None or len(times) != n:
            if times is not None and len(times) != n:
                how = (f"scenario gave {len(times)} times for {n} frames "
                       f"-- wrong scenario, or the run was truncated")
            print(f"  {YLW}[warn]{RST} {rel}  {how}")
            print(f"         falling back to uniform indices; geometry will "
                  f"render, time labels will not be physical")
            times = list(range(n))
            comment = (f"REBUILT by repair_pvd.py. Times are FRAME INDICES, "
                       f"not physical times: {how}. Geometry is correct; do "
                       f"not read the clock.")
        else:
            comment = (f"REBUILT by repair_pvd.py from {n} existing frames. "
                       f"Times {how} "
                       f"({args.projectile}->{args.target}, {args.mass:.3g} kg, "
                       f"{args.velocity/1e3:.0f} km/s, scene {scene!r}). "
                       f"The frames themselves were not rewritten.")

        if args.dry_run:
            print(f"  {YLW}[dry ]{RST} {rel}  would write {prefix}.pvd "
                  f"with {n} frames, t = {times[0]:.4g} .. {times[-1]:.4g}")
            continue

        write_pvd(str(pvd), [f.name for f in files], times, comment=comment)
        print(f"  {GRN}[fixt]{RST} {rel}  wrote {prefix}.pvd, {n} frames, "
              f"t = {times[0]:.4g} .. {times[-1]:.4g}")
        repaired += 1

    print()
    if args.dry_run:
        print(f"dry run: {len(candidates)} collection(s) would be rebuilt")
        return 0
    print(f"{GRN}rebuilt {repaired} collection(s){RST}; "
          f"{skipped} already had an index")
    print("  Verify, then render:")
    print(f"    python scripts/check_render_data.py {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
