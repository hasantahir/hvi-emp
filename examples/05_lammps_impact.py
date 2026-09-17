"""Example 5 -- Stage 1 on LAMMPS, in the configuration of Fraile et al.

Runs a W-on-W molecular-dynamics impact (Nucl. Fusion 62, 026034 (2022)
protocol), compares the MD ejecta against the reduced-order Stage 1, and
feeds the MD result into Stages 2-4 of the framework.

Scale is set by `--scale`:
    smoke   ~1.3e4 atoms,  ~1 ps   (about a minute on one core; default)
    small   ~5e4 atoms,    ~3 ps   (several minutes)
    paper   ~4e6 atoms,  100 ps    (deck generation only -- submit to a
                                    cluster; this is Fraile et al.'s scale)

Requires:  pip install lammps mpich
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

from hvi_emp.solvers import have_lammps, lammps_diagnostics
from hvi_emp.solvers.lammps_stage1 import (LammpsImpactConfig,
                                           compare_with_reduced_model,
                                           generate_input_deck, run_impact)


def write_deck(cfg, path="in.impact"):
    """Write the deck and say exactly where it went and how to run it."""
    out = Path(path).resolve()
    out.write_text(generate_input_deck(cfg))
    print(f"\nwrote {out}  ({out.stat().st_size:,} bytes)")
    print("run it with the standalone LAMMPS executable:")
    print(f"    lmp -in {out.name}                 # serial")
    print(f"    mpirun -np 8 lmp -in {out.name}    # parallel")
    return out

SCALES = {
    "smoke": dict(projectile_radius=0.5e-9, target_cells=(16, 16, 12),
                  t_equilibrate=1e-13, t_production=1.0e-12, timestep=2e-16),
    "small": dict(projectile_radius=0.8e-9, target_cells=(26, 26, 18),
                  t_equilibrate=5e-13, t_production=3.0e-12, timestep=2e-16),
    # the paper's parameters: r = 1.3 nm sphere, 0.1 fs, 20 ps + 100 ps
    "paper": dict(projectile_radius=1.3e-9, target_cells=None,
                  t_equilibrate=2e-11, t_production=1.0e-10, timestep=1e-16),
}

parser = argparse.ArgumentParser()
parser.add_argument("--scale", choices=SCALES, default="smoke")
parser.add_argument("--velocity", type=float, default=8e3,
                    help="impact speed [m/s] (paper: up to 9e3)")
parser.add_argument("--deck-only", action="store_true",
                    help="write in.impact and exit (no run)")
args = parser.parse_args()

cfg = LammpsImpactConfig(material="W", velocity=args.velocity,
                         **SCALES[args.scale])
print(f"config: {args.scale}, ~{cfg.estimated_atom_count():,} atoms, "
      f"KE = {cfg.ke_per_atom_eV:.2f} eV/atom")

if args.scale == "paper" or args.deck_only:
    write_deck(cfg)
    if args.scale == "paper":
        print("\n(paper scale -- 1e7 atoms, 100 ps -- is deliberately not run\n"
              " in-process; submit the deck above to a cluster)")
    sys.exit(0)

diag = lammps_diagnostics()
if not diag["ok"]:
    print("\n" + "=" * 74)
    print("Cannot run LAMMPS in-process, so writing the input deck instead.")
    print(f"  reason: {diag['reason']}")
    print("  to enable in-process runs:")
    print(diag["hint"].rstrip())
    print("=" * 74)
    write_deck(cfg)
    print("\nThe deck is complete and runnable -- nothing about the physics\n"
          "setup depends on the Python bindings.")
    sys.exit(0)

print(f"LAMMPS ready: {diag['reason']}")

t0 = time.time()
res = run_impact(cfg)
print(f"\nMD finished in {time.time()-t0:.0f} s")
print(res.summary())

print("\n--- MD vs reduced-order Stage 1 "
      "(expect factor-of-a-few at nm scale) ---")
cmp = compare_with_reduced_model(res)
for k, v in cmp.items():
    if isinstance(v, float):
        print(f"  {k:<34} {v:.3g}")

if res.n_ejecta > 10:
    print("\n--- MD ejecta -> Stages 2-4 ---")
    from hvi_emp import simulate_expansion
    exp = simulate_expansion(res.to_impact_handoff(), t_end=1e-7)
    print(exp.summary())
else:
    print("\n(too few ejecta at this scale/speed for a meaningful Stage-2 "
          "handoff; increase --velocity or --scale)")
