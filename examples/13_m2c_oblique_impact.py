"""M2C decks for the one case the reduced chain cannot do: oblique impact.

`hvi_emp`'s Stage 1 reduces an incidence angle to `v cos(theta)` and is
otherwise unchanged, so its plume is axisymmetric about the surface normal at
0 degrees and at 60 degrees alike. Real oblique impacts are not: the ejecta
and the vapour plume run downrange. That is a limitation of the reduced
model, not a physical result, and no amount of tuning inside `hvi_emp` fixes
it -- the geometry is simply not represented.

This script writes three M2C problem directories from one scenario: the
normal-incidence axisymmetric reference, the same impact at 45 degrees in
3-D, and a chamber-gas variant for comparison with laboratory data. It writes
input decks only; running them needs M2C (see the README each directory
carries).

    python examples/13_m2c_oblique_impact.py [outdir]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp.materials import get_material
from hvi_emp.solvers.m2c_stage1 import (M2CConfig, find_m2c,
                                        write_problem_directory)


def main(outdir: str = "m2c_runs") -> int:
    projectile = get_material("al")
    target = get_material("al")
    diameter, velocity = 1.0e-3, 1.0e4          # 1 mm, 10 km/s

    cases = {
        "normal": dict(
            angle_deg=0.0,
            why="axisymmetric reference; directly comparable with "
                "hvi_emp's own Stage 1"),
        "oblique_45": dict(
            angle_deg=45.0,
            why="the case hvi_emp cannot represent -- 3-D, downrange plume"),
        "chamber_argon": dict(
            angle_deg=0.0, ambient="ar", ambient_pressure=10.0,
            why="argon at 10 Pa; Islam et al. (2023) find the ambient gas "
                "ionises in its own right, which hvi_emp omits entirely"),
    }

    os.makedirs(outdir, exist_ok=True)
    print(f"M2C decks -> {os.path.abspath(outdir)}\n")

    for name, spec in cases.items():
        why = spec.pop("why")
        cfg = M2CConfig(projectile=projectile, target=target,
                        diameter=diameter, velocity=velocity, **spec)
        res = write_problem_directory(cfg, os.path.join(outdir, name))
        est = res["estimate"]
        print(f"{name}")
        print(f"  {why}")
        print(f"  {est['dimensionality']}, {est['cells']:.2e} cells, "
              f"{est['memory_GB']:.1f} GB")
        print(f"  ~{est['wall_hours_estimate']:.1f} h on 64 cores "
              f"(order-of-magnitude estimate only)")
        for note in res["notes"]:
            print(f"  ! {note}")
        print(f"  -> {res['paths']['input']}")
        print()

    found = find_m2c()
    print(f"M2C: {found}" if found else
          "M2C is not installed. It is GPLv3 with no application process:\n"
          "  git clone https://github.com/kevinwgy/m2c.git\n"
          "Then verify the generated grammar against your checkout:\n"
          "  python -m hvi_emp.solvers.m2c_stage1 --check <path>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
