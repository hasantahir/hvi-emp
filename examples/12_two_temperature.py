#!/usr/bin/env python3
"""Why the plume needs two temperatures.

Runs the same impact through both Stage-2 closures and prints the three
things that matter:

1.  **Energy conservation.**  With radiation off, the 2T total energy is a
    conserved quantity, so its drift is a test of the model rather than of
    the physics.  This is the check that caught the ion-temperature floor --
    a clamp that held P_i finite after the ions had cooled, doing work on the
    expansion forever and inventing 1.31% of the plume's energy.  The tell
    was that the drift was *identical* at every integration tolerance.

2.  **Where T_e and T_i separate.**  They must, or the model is pointless.

3.  **The charge-yield exponent.**  This was the framework's largest
    disagreement with experiment: 6.70 predicted against 3.48 measured.  The
    2T model gives 3.65.

Run:  python examples/12_two_temperature.py [--sweep]

``--sweep`` adds the velocity series that measures beta (slower: ~2 min).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp import get_material                                  # noqa: E402
from hvi_emp.expansion import simulate_expansion                  # noqa: E402
from hvi_emp.expansion_2t import (energy_budget,                  # noqa: E402
                                  simulate_expansion_2t)
from hvi_emp.impact import Projectile, simulate_impact            # noqa: E402

warnings.simplefilter("ignore")
EV = 1.602176634e-19


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="also fit the charge-yield exponent (~2 min)")
    args = ap.parse_args()

    Fe, Al = get_material("Fe"), get_material("Al")
    imp = simulate_impact(Projectile(Fe, 1e-12, 50e3), Al, warn=False)
    print(f"Fe -> Al, 1 pg at 50 km/s\n"
          f"  plume: {imp.m_vapour:.3e} kg of vapour at {imp.plasma.T*8.617e-5:.3f} eV, "
          f"Zbar = {imp.plasma.Zbar:.3f}\n")

    # --- 1. energy conservation -----------------------------------------
    print("1. Energy conservation (radiation off, so the total is conserved)")
    print(f"   {'rtol':>8} {'drift':>12}")
    for rtol, atol in ((1e-6, 1e-10), (1e-8, 1e-12), (1e-10, 1e-14)):
        eb = energy_budget(simulate_expansion_2t(
            imp, t_end=1e-5, radiative_cooling=False, rtol=rtol, atol=atol))
        print(f"   {rtol:8.0e} {eb['drift']:+12.2e}")
    print("   A drift that does NOT shrink with rtol is a term in the model,\n"
          "   not integration error. That is how the T_i floor was found.\n")

    # --- 2. where the temperatures separate ------------------------------
    res2 = simulate_expansion_2t(imp, t_end=1e-5)
    Te, Ti = res2.diagnostics["T_e_eV"], res2.diagnostics["T_i_eV"]
    ratio = Te / np.maximum(Ti, 1e-30)
    i_dec = int(np.argmax(ratio > 2.0))
    print("2. Electron-ion decoupling")
    print(f"   {'t [s]':>10} {'T_e [eV]':>10} {'T_i [eV]':>10} {'T_e/T_i':>9}")
    for i in (0, len(res2.t) // 4, len(res2.t) // 2, -1):
        print(f"   {res2.t[i]:10.2e} {Te[i]:10.4f} {Ti[i]:10.3e} {ratio[i]:9.2f}")
    print(f"   T_e/T_i first exceeds 2 at t = {res2.t[i_dec]:.2e} s, "
          f"peaks at {ratio.max():.1f}\n")

    # --- 3. what it changes ----------------------------------------------
    res1 = simulate_expansion(imp, t_end=1e-5)
    q1, q2 = float(res1.Q_free[-1]), float(res2.Q_free[-1])
    print("3. Frozen charge at this one velocity")
    print(f"   1-temperature   Q = {q1:.3e} C")
    print(f"   2-temperature   Q = {q2:.3e} C   ({q2/q1:.1f}x)")
    print("   Recombination returns its binding energy to the ELECTRONS. In a\n"
          "   one-temperature plume that energy is shared with ions carrying\n"
          "   ~2.7x the heat capacity per particle, the electrons stay cold,\n"
          "   and the plasma over-recombines.\n")

    if not args.sweep:
        print("Re-run with --sweep to fit the charge-yield exponent.")
        return 0

    vs = np.array([40e3, 48e3, 56e3, 66e3])
    ims = [simulate_impact(Projectile(Fe, 1e-12, v), Al, warn=False)
           for v in vs]
    last = lambda x: float(np.atleast_1d(x)[-1])          # noqa: E731
    beta = lambda q: np.polyfit(np.log(vs), np.log(np.array(q)), 1)[0]  # noqa: E731
    q_form = [last(im.Q_free) for im in ims]
    q_1t = [last(simulate_expansion(im).Q_free) for im in ims]
    q_2t = [last(simulate_expansion_2t(im).Q_free) for im in ims]

    print("4. Charge-yield exponent, Q ~ v^beta over 40-66 km/s")
    print(f"   at formation (Stage 1)   {beta(q_form):6.3f}")
    print(f"   frozen, 1-temperature    {beta(q_1t):6.3f}   <- default")
    print(f"   frozen, 2-temperature    {beta(q_2t):6.3f}")
    print(f"   measured (Close 2013)     3.48")
    print("\n   Note the sign flip: 1T freeze-out ADDS ~2.0 to beta, 2T\n"
          "   freeze-out SUBTRACTS ~1.0. The residual 4% gap is inside the\n"
          "   framework's other uncertainties, and the comparison is still\n"
          "   not like for like -- Close fits across the vaporisation\n"
          "   threshold over 3-66 km/s, this is an above-40 km/s fit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
