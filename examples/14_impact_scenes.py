"""Example 14 -- the impact, the crater and the plume, as ParaView scenes.

Writes two `.vti` series and, for each, a ParaView script that builds the
scene correctly first time.

    python examples/14_impact_scenes.py
    pvpython scenes/impact/scene_impact.py       # then just open the .pvsm

Why two scenes and not one
--------------------------
One domain cannot show both. On a box sized for the plume, the crater is
**0.013 cells across** -- a seventy-eighth of a single voxel -- and the
projectile is 1/1800th. They are not small on screen, they are absent, which
is why a default render looks like a featureless ball drifting through empty
space. Four orders of magnitude separate the crater from the plume.

So: two films of two events. Cut between them in the talk.

    impact   0 to ~40 ns      mm-scale box, crater 10-20 cells across
    plume    1 to 100 us      metre-scale box, the crater is invisible

Provenance, because it matters if anyone asks
----------------------------------------------
The **plume is solved**. The **crater and shock are drawn** from the model's
own scaling laws -- the crater as sqrt(t) to a Cour-Palais radius, the shock
as a hemisphere at Us*t carrying the decay law. Both are labelled in the
generated scene and in the .pvd header. For a solved crater, run M2C and
point the same script at its output (`source="m2c"`).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp import run_scenario
from hvi_emp.viz.paraview_scene import write_scene_for
from hvi_emp.viz.vti import (MINIMAL_FIELDS, QUALITY, SCENES,
                             estimate_memory_GB, write_scene)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="scenes")
    ap.add_argument("--projectile", default="Fe")
    ap.add_argument("--target", default="Al")
    ap.add_argument("--mass", type=float, default=1e-12)
    ap.add_argument("--velocity", type=float, default=50e3)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--quality", default="talk", choices=sorted(QUALITY),
                    help="grid size preset; see --list-quality")
    ap.add_argument("--n", type=int, default=None,
                    help="grid points per axis; overrides --quality")
    ap.add_argument("--renderer", default="gpu",
                    choices=("gpu", "index", "smart"),
                    help="gpu pins ParaView's GPU volume mapper (default); "
                         "index also tries the NVIDIA IndeX plugin")
    ap.add_argument("--movie", action="store_true",
                    help="also render a PNG frame series (run with pvbatch)")
    ap.add_argument("--minimal-fields", action="store_true",
                    help="write only the two arrays the scene renders")
    ap.add_argument("--allow-oversize", action="store_true")
    ap.add_argument("--list-quality", action="store_true")
    ap.add_argument("--preset", default="talk",
                    choices=("talk", "paper", "diagnostic"))
    ap.add_argument("--scenes", default="impact,plume",
                    help="comma-separated: " + ",".join(SCENES))
    args = ap.parse_args(argv)

    if args.list_quality:
        print(f"{'preset':<14}{'grid':>8}{'Mcells':>9}{'peak RAM':>11}"
              f"{'VRAM/array':>12}   what it is for")
        for k, v in QUALITY.items():
            m = estimate_memory_GB((v["n"],) * 3)
            print(f"{k:<14}{v['n']:>6}^3{m['cells']/1e6:>9.1f}"
                  f"{m['peak_GB']:>9.1f} GB{m['vram_GB']*1e3:>9.0f} MB"
                  f"   {v['for']}")
        print("\nPeak RAM is ~175 B/cell, measured on this sampler -- the")
        print("limit is host memory during generation, not the GPU. A volume")
        print("renderer uploads one array, which is 44x smaller.")
        return 0

    print(f"{args.projectile} {args.mass:.3e} kg at "
          f"{args.velocity/1e3:.1f} km/s -> {args.target}\n")
    sc = run_scenario(args.projectile, args.target, mass=args.mass,
                      velocity=args.velocity, t_end=1e-5)

    shape = (args.n,) * 3 if args.n else None
    fields = MINIMAL_FIELDS if args.minimal_fields else None
    made = []
    for scene in [s.strip() for s in args.scenes.split(",") if s.strip()]:
        if scene not in SCENES:
            print(f"  skipping unknown scene {scene!r}")
            continue
        print("-" * 70)
        directory = os.path.join(args.out, scene)
        res = write_scene(sc, directory, scene=scene, n_frames=args.frames,
                          shape=shape, quality=args.quality, fields=fields,
                          allow_oversize=args.allow_oversize)
        script = os.path.join(directory, f"scene_{scene}.py")
        write_scene_for(res, script, preset=args.preset,
                        renderer=args.renderer,
                        movie_path=(os.path.join(directory, "frames",
                                                 f"{scene}.png")
                                    if args.movie else None),
                        title=(f"{args.projectile} {args.mass:.0e} kg, "
                               f"{args.velocity/1e3:.0f} km/s -> "
                               f"{args.target}   |   {SCENES[scene]['what']}"))
        print(f"  scene script: {script}")
        made.append((scene, res, script))
        print()

    print("=" * 70)
    print("Open in ParaView:\n")
    for scene, res, script in made:
        print(f"  pvpython {script}"
              + ("      # or pvbatch, for headless GPU rendering"
                 if args.movie else ""))
    print("\nEach script builds the scene, then writes a .pvsm beside itself.")
    print("After the first run, open the .pvsm instead -- it is faster and it")
    print("was written by your own ParaView version.\n")

    for scene, res, _ in made:
        cells = res["cells_across_crater"]
        note = ("crater resolved" if cells >= 1.0
                else "crater NOT resolved -- this scene is the plume only")
        print(f"  {scene:<16} crater {cells:>8.2f} cells   {note}")

    if made:
        m = made[0][1]["memory"]
        print(f"\n  peak RAM was ~{m['peak_GB']:.1f} GB per frame; the array "
              f"ParaView uploads is {m['vram_GB']*1e3:.0f} MB.")
        print("  The GPU is not the constraint here -- host RAM during")
        print("  generation is. `--list-quality` has the table.")

    print("\nThe plume is solved. The crater and shock are DRAWN from the")
    print("model's scaling laws -- see docs/VISUALISATION.md before using")
    print("any of this as evidence of a computed crater.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
