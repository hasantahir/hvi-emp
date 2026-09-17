"""Example 4 -- the Olympus-1 end-of-life anomaly as a worked case.

On 11-12 August 1993, during the predicted peak of the Perseid shower, ESA's
Olympus-1 lost Earth pointing and entered an uncontrolled spin, ultimately
ending the mission.  Caswell, McBride & Taylor (Int. J. Impact Eng. 17, 139,
1995) concluded that an impact could not be proven but was a credible
scenario, and proposed that "the impact by a small meteoroid may have
generated a plasma triggering a discharge of charged surfaces entering the
grounded spacecraft via the umbilical and an external sensor."

This script runs that scenario through the framework for a range of Perseid
masses.  Perseids are cometary, low density (~1000 kg/m^3 or less), and strike
at 59-72 km/s -- the fastest annual shower, which is why the Perseids dominate
the meteoroid risk budget for GEO spacecraft in August.

Produces `figures/04_olympus.png`.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.simplefilter("ignore")

import numpy as np

from _plotting import figures_dir, get_pyplot

from hvi_emp import KAPTON, OLIVINE, run_scenario
from hvi_emp.coupling import UPSET_THRESHOLDS


V_PERSEID = 59e3        # km/s, the canonical Perseid geocentric speed
masses = np.geomspace(1e-9, 1e-4, 12)      # 1 ug to 100 mg

rows = []
for m in masses:
    try:
        sc = run_scenario(OLIVINE, "Al", mass=m, velocity=V_PERSEID,
                          t_end=2e-4, standoff=0.30,
                          loop_area=5e-2,        # a metre-scale harness loop
                          wire_length=0.5,
                          shielding_dB=40.0,
                          r_sensor=1.0)
    except Exception as exc:
        print(f"  m={m:.1e} kg: {exc}")
        continue
    if sc.expansion is None:
        continue
    rows.append({
        "m": m,
        "d_mm": 2e3 * sc.impact.projectile.radius,
        "Q": sc.expansion.Q_final,
        "V_diff": sc.charging.peak_differential_V,
        "V_ind": sc.coupling.shielded_V,
        "E": sc.emp.peak_field,
        "paths": sc.anomaly["pathways"],
    })

print(f"Perseid impact on an Al spacecraft panel at {V_PERSEID/1e3:.0f} km/s\n")
print(f"{'mass [kg]':>11}{'diam [mm]':>11}{'Q [C]':>12}"
      f"{'V_diff [V]':>12}{'V_induced [V]':>15}  pathways")
print("-" * 90)
for r in rows:
    print(f"{r['m']:>11.2e}{r['d_mm']:>11.3f}{r['Q']:>12.2e}"
          f"{r['V_diff']:>12.1f}{r['V_ind']:>15.3e}  "
          f"{'; '.join(r['paths']) if r['paths'] else '-'}")

m = np.array([r["m"] for r in rows])
Vd = np.array([r["V_diff"] for r in rows])
Vi = np.array([r["V_ind"] for r in rows])
Q = np.array([r["Q"] for r in rows])

plt = get_pyplot("the Olympus figure")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].loglog(m, Q, "o-")
ax[0].set(xlabel="meteoroid mass [kg]", ylabel="free charge Q [C]",
          title="Charge generated")
ax[0].grid(alpha=.3, which="both")

ax[1].loglog(m, np.maximum(Vd, 1e-6), "o-", label="differential potential")
ax[1].axhline(UPSET_THRESHOLDS["esd_dielectric_V"][0], color="r", ls="--",
              label="ESD threshold (500 V)")
ax[1].set(xlabel="meteoroid mass [kg]", ylabel="V", title="Surface charging")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")

ax[2].loglog(m, np.maximum(Vi, 1e-12), "o-", label="induced, after 40 dB shield")
ax[2].axhline(UPSET_THRESHOLDS["cmos_upset_V"][0], color="r", ls="--",
              label="CMOS upset (0.5 V)")
ax[2].set(xlabel="meteoroid mass [kg]", ylabel="V",
          title="EMP coupling into harness")
ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, which="both")

fig.suptitle("Olympus-1 scenario: Perseid impact at 59 km/s")
fig.tight_layout()
fig.savefig(figures_dir() / "04_olympus.png", dpi=140)
if plt:
    print(f"\nwrote {figures_dir() / '04_olympus.png'}")

crit_esd = [r["m"] for r in rows
            if any("ESD" in p for p in r["paths"])]
crit_emp = [r["m"] for r in rows
            if any("EMP" in p for p in r["paths"])]
print("\nSmallest meteoroid crossing each threshold:")
print(f"  surface ESD : {min(crit_esd):.2e} kg" if crit_esd
      else "  surface ESD : not reached in this mass range")
print(f"  EMP coupling: {min(crit_emp):.2e} kg" if crit_emp
      else "  EMP coupling: not reached in this mass range")
print("\nCaveat: thresholds are generic (NASA-HDBK-4002A class), the standoff "
      "and\nharness geometry are assumed, and the framework's absolute EMP "
      "amplitude is\nuncertain by ~2 orders of magnitude (see docs/THEORY.md "
      "Sec. 7). Treat this as\na structured way to bound the question, not as "
      "a root-cause determination.")
