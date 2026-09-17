#!/usr/bin/env python3
"""MD trajectory -> crater -> inferred ionisation -> plasma, for ParaView.

    python examples/10_md_crater_to_plasma.py --dump impact.dump
    python examples/10_md_crater_to_plasma.py --dump impact.dump --cell 5e-10

Produces per-atom `.vtp` (lattice and ejecta, coloured by temperature or
ionisation) and binned `.vti` (continuum fields for volume rendering).

Classical MD has no electrons and cannot ionise. The ionisation shown is
inferred by applying the framework's Saha solver to the MD density and
temperature under an LTE assumption -- see
hvi_emp/solvers/lammps_postprocess.py.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvi_emp.solvers.lammps_postprocess import convert_dump


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, help="LAMMPS dump file")
    ap.add_argument("--material", default="W")
    ap.add_argument("--cell", type=float, default=None,
                    help="bin size [m]; default 1 nm")
    ap.add_argument("--timestep-fs", type=float, default=0.1,
                    help="MD timestep in fs (the Fraile protocol uses 0.1)")
    ap.add_argument("--outdir", default="md_out")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-atoms", action="store_true")
    ap.add_argument("--no-grid", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.dump):
        print(f"no such dump: {args.dump}\n"
              "Generate one with:\n"
              "    python -m hvi_emp lammps --material W --velocity 9e3 "
              "--run\n"
              "or add dump_every=100 to LammpsImpactConfig.")
        return 1

    print(f"reading {args.dump}")
    out = convert_dump(args.dump, args.outdir, material=args.material,
                       cell=args.cell, timestep_fs=args.timestep_fs,
                       max_frames=args.max_frames,
                       atoms=not args.no_atoms, grid=not args.no_grid)

    print(f"""
In ParaView
-----------
  atoms:  {out.get('atoms_pvd', '(skipped)')}
      representation "Point Gaussian", colour by Zbar (or T_eV)
      Edit Color Map -> Enable Log Scale, Rescale Over All Timesteps
  grid:   {out.get('grid_pvd', '(skipped)')}
      Volume representation, colour by electron_density, log scale

Remember: ionisation is INFERRED from the MD density and temperature via
Saha under LTE, not simulated. Classical MD has no electrons.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
