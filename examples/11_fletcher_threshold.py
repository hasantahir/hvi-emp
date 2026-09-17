#!/usr/bin/env python3
"""Why this framework and Fletcher (2021) disagree about the vaporisation threshold.

    python examples/11_fletcher_threshold.py

Fletcher, NRL/MR/6757--20-10,138 (AD1123389), p.5:

    "We found a threshold velocity that lies between 10 km/s and 20 km/s at
     which the projectile is entirely vaporized (consistent with Zel'dovich
     and Raizer) and the plasma transitions from partially ionized to fully
     ionized."

for a **tungsten projectile on an aluminium target**.  This framework puts
complete vaporisation of tungsten at ~36 km/s -- roughly twice as fast.

Both numbers are defensible; they answer different questions.  This script
computes the two criteria side by side so the disagreement is quantified
rather than argued about, and so you can decide which one your replication
should target.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hvi_emp import get_material, run_scenario
from hvi_emp.eos import impedance_match, phase_thresholds

FLETCHER_VELOCITIES = (12e3, 22e3, 32e3, 42e3, 52e3, 62e3)


def main():
    W, Al = get_material("W"), get_material("Al")
    E_vap = phase_thresholds(W)["E_vap"]

    print(__doc__)
    print("=" * 78)
    print(f"Tungsten: complete-vaporisation energy {E_vap / 1e6:.2f} MJ/kg "
          f"({W.E_cohesive / 1.602176634e-19:.2f} eV/atom cohesive, "
          f"rho0 = {W.rho0:.0f} kg/m^3)")
    print("=" * 78)

    print(f"\n{'v':>7} {'P_shock':>9} | {'E_shock':>9} {'/E_vap':>7} | "
          f"{'E_waste':>9} {'/E_vap':>7} {'kept':>7} | {'f_vap':>6}")
    print(f"{'[km/s]':>7} {'[GPa]':>9} | {'[MJ/kg]':>9} {'':>7} | "
          f"{'[MJ/kg]':>9} {'':>7} {'':>7} | {'model':>6}")
    print("-" * 78)

    scan = (10e3, 12e3, 13e3, 16e3, 20e3, 26e3, 32e3, 36e3, 42e3, 52e3, 62e3)
    for v in scan:
        st = impedance_match(W, Al, v, warn=False)
        E_shock = 0.5 * st.up_projectile ** 2       # Rankine-Hugoniot, up^2/2
        sc = run_scenario("W", "Al", mass=1e-9, velocity=v)
        E_waste = sc.impact.E_res_projectile
        mark = "  <-- Fletcher" if 10e3 <= v <= 20e3 else ""
        print(f"{v / 1e3:7.0f} {st.P / 1e9:9.0f} | {E_shock / 1e6:9.2f} "
              f"{E_shock / E_vap:7.2f} | {E_waste / 1e6:9.2f} "
              f"{E_waste / E_vap:7.2f} {100 * E_waste / E_shock:6.1f}% | "
              f"{sc.impact.f_vap_projectile:6.3f}{mark}")

    # where each criterion crosses unity
    def crossing(fn):
        vs = np.arange(6e3, 74e3, 250.0)
        for v in vs:
            if fn(v) >= 1.0:
                return v
        return None

    def shock_ratio(v):
        st = impedance_match(W, Al, v, warn=False)
        return 0.5 * st.up_projectile ** 2 / E_vap

    def waste_ratio(v):
        return run_scenario("W", "Al", mass=1e-9,
                            velocity=v).impact.f_vap_projectile

    v_shock = crossing(shock_ratio)
    v_waste = crossing(waste_ratio)

    print(f"""
{'=' * 78}
THE TWO CRITERIA
{'=' * 78}

  Shock-energy (Zel'dovich & Raizer, cited by Fletcher)
      "the shock deposited enough energy to vaporise the material"
      E_shock = up^2/2  >=  E_vap
      crosses at   {v_shock / 1e3:.0f} km/s     <-- inside Fletcher's 10-20 km/s

  Release waste-heat (this framework, and Ahrens & O'Keefe)
      "enough energy SURVIVES decompression to leave it as vapour"
      E_waste(release to the spinodal)  >=  E_vap
      crosses at   {v_waste / 1e3:.0f} km/s

  Only ~12% of the shock internal energy is retained
  after isentropic release; the rest is returned as expansion work (P dV).
  That single factor is the whole disagreement.

WHICH IS RIGHT?
{'-' * 78}
The waste-heat criterion is the standard one in shock physics, and this
framework reproduces Ahrens & O'Keefe's aluminium thresholds with it to
within 2% and no free parameters (374 vs 380 GPa incipient, 1504 vs 1500 GPa
complete). By the classical criterion, ~36 km/s for tungsten is right.

But Fletcher is not using the classical criterion alone. ALEGRA runs SESAME
tables with the **LMD conductivity model, which includes pressure
ionisation**, and the report states that "pressure ionization significantly
increases the ionization state in early phases of impact". Pressure
ionisation converts energy into ionisation *while the material is still
compressed*. That energy is not recoverable as expansion work, so it changes
the release path in a way Saha-on-release cannot capture. This framework
lists exactly that as a known gap (THEORY.md section 7).

So the honest position is: the two numbers bracket the answer, the mechanism
that would close the gap is named and physical, and it is testable.

A CONCRETE PREDICTION FOR YOUR REPLICATION
{'-' * 78}
iSALE with an ANEOS tungsten table has **no pressure-ionisation model**
either. If pressure ionisation is what lowers Fletcher's threshold, then
iSALE should land near {v_waste / 1e3:.0f} km/s, not 10-20 km/s -- i.e. iSALE should
agree with this framework and *disagree* with ALEGRA.

If instead iSALE reproduces 10-20 km/s, the difference is not pressure
ionisation but the definition of "entirely vaporized": Fletcher judges it
from density snapshots 12 microseconds after impact, where hot dense fluid
above the critical point can look like vapour without having crossed the
complete-vaporisation energy.

Either outcome is a publishable result about the threshold, which is why it
is worth running the 12 and 22 km/s cases first.
""")

    print(f"{'=' * 78}\nFletcher's six speeds, this framework, W -> Al\n"
          f"{'=' * 78}")
    print(f"{'v [km/s]':>9} {'f_vap':>7} {'m_plasma/m_p':>13} {'T_e [eV]':>9} "
          f"{'Zbar':>6} {'v_exp [km/s]':>13}")
    for v in FLETCHER_VELOCITIES:
        sc = run_scenario("W", "Al", mass=1e-9, velocity=v)
        i = sc.impact
        print(f"{v / 1e3:9.0f} {i.f_vap_projectile:7.3f} "
              f"{i.m_plasma / i.projectile.mass:13.3f} {i.plasma.T_eV:9.3f} "
              f"{i.plasma.Zbar:6.3f} {i.v_expansion / 1e3:13.1f}")
    print("""
Compare against the report: "For 22 km/s and above, there is a significant
amount of gas and plasma generated that expands outwards at speeds of
several km/s." Our plasma appears at 32 km/s, one step later, consistent
with the threshold offset above.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
