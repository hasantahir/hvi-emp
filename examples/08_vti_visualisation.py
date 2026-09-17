"""Example 8 -- write a `.vti` time series of the impact, plume and smoke.

Produces a ParaView/VisIt-ready animation of the whole sequence: projectile
approaching, shock into the target, crater opening, plasma plume expanding,
and the slower condensate ("smoke") trailing behind it.

    python examples/08_vti_visualisation.py
    paraview vti_out/impact.pvd

Read `docs/VISUALISATION.md` before presenting the images: the plume fields
are solved, the crater and shock front are drawn from scaling laws.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

from hvi_emp import run_scenario
from hvi_emp.viz import FIELD_PROVENANCE, write_impact_series
from hvi_emp.viz.vti import DEFAULT_FIELDS

ap = argparse.ArgumentParser()
ap.add_argument("--projectile", default="Fe")
ap.add_argument("--target", default="Al")
ap.add_argument("--mass", type=float, default=1e-9, help="kg")
ap.add_argument("--velocity", type=float, default=30e3, help="m/s")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--n", type=int, default=128, help="grid points per side")
ap.add_argument("--out", default="vti_out")
ap.add_argument("--domain-mode", default="window",
                choices=["adaptive", "fixed"])
ap.add_argument("--all-fields", action="store_true",
                help="write every field, not just the common ones")
ap.add_argument("--no-compress", action="store_true")
ap.add_argument("--frame", choices=("lab", "plume"), default="plume",
                help="'lab' keeps the impact site in view; 'plume' follows "
                     "the plume so the expansion fills the frame")
args = ap.parse_args()

sc = run_scenario(args.projectile, args.target, mass=args.mass,
                  velocity=args.velocity)
print(sc.impact.summary())

if sc.expansion is None:
    print("\nThis impact produced no plume (below the vaporisation "
          "threshold) -- there is nothing to visualise.\n"
          "Raise --velocity or --mass.")
    sys.exit(1)

print(f"\nwriting {args.frames} frames at {args.n}^3 ...")
t0 = time.time()
res = write_impact_series(
    sc, args.out, n_frames=args.frames, shape=(args.n,) * 3,
    fields="all" if args.all_fields else DEFAULT_FIELDS,
    compress=not args.no_compress, domain_mode=args.domain_mode,
    frame_of_reference=args.frame)
dt = time.time() - t0

vol = res["volume"]
print(f"\n{dt:.1f} s, {res['bytes']/1e6:.1f} MB total "
      f"({res['bytes']/1e6/len(res['files']):.2f} MB/frame)")
print(f"time span : {res['times'][0]:.2e} to {res['times'][-1]:.2e} s")
print(f"domain    : {args.domain_mode}"
      + (f", grows to +/-{vol.domain*1e3:.1f} mm"
         if args.domain_mode == "adaptive" else
         f", fixed at +/-{vol.domain*1e3:.1f} mm"))

print("\nfields written:")
names = FIELD_PROVENANCE if args.all_fields else {
    k: FIELD_PROVENANCE[k] for k in DEFAULT_FIELDS}
for k, v in names.items():
    print(f"  {k:<18} {v}")

# The range to type into ParaView. Colouring by the raw density and
# rescaling to its data range is what makes a plume render as a flat blob:
# the field spans ~36 decades in one frame, so everything but the core lands
# in the bottom colour.
import numpy as np
lo, hi = np.inf, -np.inf
for t in res["times"]:
    lg = vol.frame(float(t))["log10_plasma_density"]
    lo, hi = min(lo, float(lg.min())), max(hi, float(lg.max()))

fills = []
for t in res["times"]:
    d = vol.frame(float(t))["plasma_density"].ravel()
    m = d.max()
    fills.append(100 * int((d > 1e-3 * m).sum()) / d.size if m > 0 else 0.0)
# Frames before contact have no plume at all; including them makes the
# growth ratio a division by zero dressed up as a huge number.
live = [f for f in fills if f > 0.0]
growth = (max(live) / min(live)) if len(live) > 1 else 1.0
print(f"\nplume fills {min(live):.2f}% -> {max(live):.1f}% of the grid "
      f"({growth:.1f}x on screen, over {len(live)} frames with plume)")
if growth < 2.0:
    print("  ! The plume barely changes size on screen. That is a rendering\n"
          "    setting, not the physics: the plume really does grow by ~4x\n"
          "    over this window.")
    if args.domain_mode == "adaptive":
        print("    Cause: --domain-mode adaptive resizes the box WITH the\n"
              "    plume, holding its apparent size fixed. Use 'window'.")
    elif args.frame == "lab":
        print("    Cause: the lab-frame box must hold the plume's whole\n"
              "    trajectory, which dwarfs it. Use --frame plume.")
for note in vol.notes:
    print(f"  ! {note}")

print(f"""
open it:
    paraview {res['pvd']}

a quick recipe in ParaView:
  1. open the .pvd, hit Apply
  2. Volume representation, colour by 'log10_plasma_density'
     -- NOT 'plasma_density': it spans ~36 decades in a single frame, so a
        linear colour bar shows a uniform blob and nothing else
  3. Rescale to Custom Data Range: {lo:.1f} to {hi:.1f}
     (that covers every frame, so the colours stay comparable in time)
  4. Threshold on 'material' == 1 for the target solid, == 2 for the crater
     (lab frame only -- the plume frame does not draw the target)
  5. play -- the timeline is in real seconds""")
