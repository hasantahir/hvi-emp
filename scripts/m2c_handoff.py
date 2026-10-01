#!/usr/bin/env python3
"""Hand an M2C run over to PIC: Fletcher's waterfall, step two.

    python scripts/m2c_handoff.py full_chain/hydro/results
    python scripts/m2c_handoff.py full_chain/hydro/results --fraction 0.8
    python scripts/m2c_handoff.py full_chain/hydro/results --time 2.4e-6
    python scripts/m2c_handoff.py full_chain/hydro/results --scan-all

Reads M2C's `solution.pvd` and its `.vtr` frames, finds the first frame in
which the plume has gone collisionless (electron plasma frequency above the
electron-ion collision frequency for at least `--fraction` of its free
electrons -- Fletcher 2021, Fig. 2), and writes the handoff file the PIC
bridges read:

    <out>.npz   the plume on the M2C mesh: n_e, n_heavy, Z-bar, T, velocity
    <out>.json  total charge, temperature, expansion velocity, and the
                criteria and provenance needed to reproduce the decision

If no frame qualifies, it says why and stops. Forcing one with `--time` is
allowed, and the file then records how much of the plume was still
collisional when the transition was declared instantaneous.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp.handoff import HandoffCriteria, assess, extract, find_handoff
from hvi_emp.solvers.m2c_output import read_m2c_series, read_m2c_vtr

GRN, RED, YLW, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def ok(m):
    print(f"  {GRN}[ ok ]{RST} {m}")


def warn(m):
    print(f"  {YLW}[warn]{RST} {m}")


def bad(m):
    print(f"  {RED}[FAIL]{RST} {m}")


def head(m):
    print(f"\n{m}\n" + "=" * 72)


def row(a):
    r = (a["omega_pe_median"] / a["nu_ei_median"]
         if a.get("nu_ei_median") else float("nan"))
    flag = "  <- handoff" if a.get("ready") else ""
    return (f"  t = {a['time']:.4e} s   Q = {a['total_charge_C']:.3e} C   "
            f"collisionless {a.get('collisionless_fraction', 0):6.1%}   "
            f"w_pe/nu_ei {r:9.3g}{flag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="M2C results directory (has solution.pvd)")
    ap.add_argument("--out", help="handoff file stem "
                                  "(default: <results>/../handoff/plume)")
    ap.add_argument("--fraction", type=float, default=0.5,
                    help="share of free electrons that must be collisionless")
    ap.add_argument("--time", type=float,
                    help="force the frame nearest this time [s]")
    ap.add_argument("--scan-all", action="store_true",
                    help="assess every frame, not just up to the handoff")
    ap.add_argument("--surface-x", type=float, default=0.0,
                    help="target surface on M2C's x axis [m]")
    ap.add_argument("--outward", type=float, default=-1.0,
                    help="sign of the outward surface normal along x")
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args(argv)

    crit = HandoffCriteria(surface_x=args.surface_x, outward=args.outward,
                           collisionless_fraction=args.fraction)
    try:
        series = read_m2c_series(args.results)
    except FileNotFoundError as exc:
        bad(str(exc))
        return 2
    if not series:
        bad("solution.pvd lists no frames -- the run wrote no output yet")
        return 2

    head(f"M2C run: {args.results}  ({len(series)} frames, "
         f"t = {series[0][0]:.3e} .. {series[-1][0]:.3e} s)")
    out = args.out or os.path.join(os.path.dirname(
        os.path.abspath(args.results.rstrip("/"))), "handoff", "plume")

    if args.time is not None:
        i = min(range(len(series)), key=lambda k: abs(series[k][0] - args.time))
        t, path = series[i]
        snap = read_m2c_vtr(path, time=t)
        a = assess(snap, crit)
        print(row(a))
        warn(f"frame forced by --time: {a.get('collisionless_fraction', 0):.0%}"
             f" collisionless (criterion asks for {args.fraction:.0%})")
    else:
        if args.scan_all:
            rows = [assess(read_m2c_vtr(p, time=t), crit) for t, p in series]
            idx = next((k for k, r in enumerate(rows) if r.get("ready")), None)
        else:
            idx, rows = find_handoff(series, crit)
        for a in rows:
            print(row(a))
        if idx is None:
            head("No handoff")
            best = max(rows, key=lambda r: r.get("collisionless_fraction", 0))
            bad(f"the plume never reached {args.fraction:.0%} collisionless "
                f"(best: {best.get('collisionless_fraction', 0):.1%} at "
                f"t = {best['time']:.3e} s)")
            fr = [r.get("collisionless_fraction", 0) for r in rows]
            if len(fr) > 2 and fr[-1] < max(fr):
                print("  The fraction is FALLING with time. That is what an "
                      "adiabatically cooling\n  plume does: n ~ s^-3 and "
                      "T ~ s^-2 hold nu_ei ~ n T^-3/2 constant while omega_pe"
                      "\n  falls. Close et al.'s mechanism needs the electron "
                      "temperature to stay up.\n  Check the plume temperature"
                      " history before forcing a handoff.")
            else:
                print("  The fraction is still rising: the run may simply be "
                      "too short. Extend\n  MaxTime, or force a frame with "
                      "--time and accept the recorded collisional share.")
            return 1
        t, path = series[idx]
        snap = read_m2c_vtr(path, time=t)
        a = rows[idx]

    try:
        h = extract(snap, crit, a, allow_synthetic=args.allow_synthetic)
    except ValueError as exc:
        bad(str(exc))
        return 1
    npz, js = h.save(out)

    s = h.summary
    head(f"Handoff at t = {s['time_s']:.4e} s")
    print("  Fletcher (2021): 'total charge generated, temperature, and "
          "expansion velocity'")
    print(f"    total charge         {s['total_charge_C']:.4e} C "
          f"({s['N_free_electrons']:.3e} electrons)")
    print(f"    electron temperature {s['T_e_eV_mean']:.3g} eV  "
          f"(p10 {s['T_e_eV_p10']:.3g}, p90 {s['T_e_eV_p90']:.3g})")
    print(f"    expansion speed      {s['expansion_speed_mean_m_s'] / 1e3:.3g}"
          f" km/s mean, {s['expansion_speed_p90_m_s'] / 1e3:.3g} km/s p90")
    print(f"    axial bulk velocity  "
          f"{s['bulk_velocity_axial_m_s'] / 1e3:+.3g} km/s")
    print(f"    mean charge state    {s['Zbar_mean']:.3g}")
    print(f"    plume rms size       {s['rms_axial_m'] * 1e3:.3g} mm axial, "
          f"{s['rms_radial_m'] * 1e3:.3g} mm radial")
    for note in h.notes:
        warn(note)
    ok(f"wrote {npz}")
    ok(f"wrote {js}")
    print("\n  Next: python scripts/warpx_from_handoff.py " + out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
