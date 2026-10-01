#!/usr/bin/env python3
"""Write a WarpX run initialised from an M2C handoff: Fletcher's waterfall, step three.

    python scripts/warpx_from_handoff.py full_chain/hydro/handoff/plume --ion Al
    python scripts/warpx_from_handoff.py <stem> --ion W --advance none
    python scripts/warpx_from_handoff.py <stem> --ion Al --budget 5e12

Reads the handoff written by `scripts/m2c_handoff.py`, samples electron/ion
macroparticle pairs from the hydrocode's plume cells, optionally advances
them ballistically to the epoch where the plume resonates with the measured
band (as Fletcher & Close 2017 start their PIC), and writes:

    <out>/particles.npz            the macroparticles WarpX will load
    <out>/warpx_from_handoff.py    the PICMI script (RZ, needs pywarpx)
    <out>/plan.json                grid, scheme, costs, and every assumption

Run it with:   cd <out> && mpirun -np <N> python warpx_from_handoff.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp.handoff import load_handoff
from hvi_emp.solvers.warpx_handoff import (WarpXHandoffConfig, n_resonant,
                                           write_warpx_from_handoff)

GRN, RED, YLW, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def ok(m):
    print(f"  {GRN}[ ok ]{RST} {m}")


def warn(m):
    print(f"  {YLW}[warn]{RST} {m}")


def bad(m):
    print(f"  {RED}[FAIL]{RST} {m}")


def head(m):
    print(f"\n{m}\n" + "=" * 72)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("handoff", help="handoff file stem (from m2c_handoff.py)")
    ap.add_argument("--ion", help="plume material, for the ion mass "
                                  "(e.g. Al, W)")
    ap.add_argument("--ion-mass-amu", type=float,
                    help="ion mass directly, instead of --ion")
    ap.add_argument("--out", help="run directory (default: next to the "
                                  "handoff, warpx/)")
    ap.add_argument("--advance", choices=("band", "none"), default="band")
    ap.add_argument("--scheme", choices=("auto", "explicit", "implicit"),
                    default="auto")
    ap.add_argument("--band", type=float, nargs="+", default=[315e6, 916e6],
                    help="band frequencies [Hz]")
    ap.add_argument("--ppc", type=int, default=8)
    ap.add_argument("--budget", type=float, default=2.0e11,
                    help="cell-steps an explicit run may cost")
    args = ap.parse_args(argv)

    if args.ion_mass_amu is None and args.ion is None:
        bad("give --ion (a material name) or --ion-mass-amu")
        return 2
    if args.ion_mass_amu is not None:
        A = args.ion_mass_amu
    else:
        from hvi_emp import get_material
        A = get_material(args.ion).A

    h = load_handoff(args.handoff)
    stem = args.handoff[:-4] if args.handoff.endswith((".npz",
                                                       ".json")) else args.handoff
    out = os.path.normpath(args.out or os.path.join(
        os.path.dirname(os.path.abspath(stem)), "..", "warpx"))
    cfg = WarpXHandoffConfig(ion_mass_amu=A, advance=args.advance,
                             scheme=args.scheme, band_hz=tuple(args.band),
                             ppc=args.ppc, cell_step_budget=args.budget)

    s = h.summary
    head(f"Handoff: t = {s['time_s']:.4e} s, {s['total_charge_C']:.3e} C, "
         f"T_e {s['T_e_eV_mean']:.3g} eV")
    if h.provenance.get("synthetic"):
        warn("SYNTHETIC handoff: a test cloud, not an M2C result")

    res = write_warpx_from_handoff(h, out, cfg)
    pl, st = res["plan"], res["stats"]

    head("PIC start")
    print(f"  {res['advance_note']}")
    print(f"  peak n_e {st['n_peak_binned']:.3e} m^-3   edge n_e "
          f"{st['n_edge_binned']:.3e} m^-3   (916 MHz resonant: "
          f"{n_resonant(916e6):.3e})")
    print(f"  plume rms {st['rms_z']:.3e} m axial, {st['rms_r']:.3e} m "
          f"radial;  t = {st['t']:.4e} s")

    head(f"Grid: {pl['scheme']}")
    print(f"  RZ {pl['nr']} x {pl['nz']}, dx = {pl['dx']:.3e} m "
          f"({pl['dx'] / pl['lambda_D_peak']:.3g} lambda_D at the peak)")
    print(f"  dt = {pl['dt']:.3e} s, {pl['n_steps']} steps, t_max = "
          f"{pl['t_max']:.3e} s")
    print(f"  cost: {pl['cell_steps']:.2e} cell-steps "
          f"(explicit would be {pl['cell_steps_explicit']:.2e})")
    for n in pl["notes"]:
        warn(n)
    if res["charge_in_particles_C"] > 0:
        ok(f"charge in particles {res['charge_in_particles_C']:.6e} C "
           f"(handoff {s['total_charge_C']:.6e} C)")
    ok(f"wrote {res['particles']}")
    ok(f"wrote {res['script']}")
    ok(f"wrote {os.path.join(out, 'plan.json')}")
    print(f"\n  Run:  cd {out} && mpirun -np <N> python "
          f"warpx_from_handoff.py")
    print("  The script has not been executed here; WarpX's boundary names "
          "and\n  implicit-RZ support are the first things to check on a "
          "new build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
