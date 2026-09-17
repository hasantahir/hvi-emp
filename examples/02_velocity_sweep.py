"""Example 2 -- charge yield and EMP amplitude versus impact speed.

Reproduces the classic log-log Q(v) plot and compares with the empirical
scaling laws. Produces `figures/02_velocity_sweep.png`.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.simplefilter("ignore")

import numpy as np

from _plotting import figures_dir, get_pyplot

from hvi_emp import empirical_charge_yield, velocity_sweep


v = np.linspace(10e3, 72e3, 25)
res = velocity_sweep("Fe", "Al", mass=1e-12, velocities=v, t_end=1e-5)

plt = get_pyplot("the sweep figure")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

good = res["Q_frozen"] > 0
ax[0].loglog(v / 1e3, np.maximum(res["Q_stage1"], 1e-20), "o-",
             label="Q at formation")
ax[0].loglog(v / 1e3, np.maximum(res["Q_frozen"], 1e-20), "s-",
             label="Q after freeze-out")
ax[0].loglog(v / 1e3, [empirical_charge_yield(1e-12, x, "fe_on_al") for x in v],
             "k--", label=r"empirical $0.1\,m\,v^{3.4}$")
if good.sum() >= 3:
    hi = good & (v >= 40e3)
    b = np.polyfit(np.log(v[hi]), np.log(res["Q_frozen"][hi]), 1)[0]
    ax[0].set_title(f"Charge yield (asymptotic $\\beta$ = {b:.2f}, "
                    "measured 3.4-3.5)")
ax[0].set(xlabel="impact speed [km/s]", ylabel="Q [C]",
          ylim=(1e-16, 1e-3))
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")

ax[1].plot(v / 1e3, res["T_eV"], "o-", label=r"$T_e$ [eV]")
ax[1].plot(v / 1e3, res["Zbar"], "s-", label=r"$\bar{Z}$")
ax[1].plot(v / 1e3, res["vapour_efficiency"], "^-",
           label=r"$m_{vap}/m_{proj}$")
ax[1].axhspan(2.0, 2.5, color="g", alpha=.15,
              label="Fletcher & Close: 2-2.5 eV")
ax[1].set(xlabel="impact speed [km/s]", title="Plume state")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

ax[2].semilogy(v / 1e3, np.maximum(res["E_916MHz"], 1e-12), "o-")
ax[2].axhline(1.9e-3, color="k", ls="--",
              label="Close et al. (2013): 1.9 mV/m")
ax[2].set(xlabel="impact speed [km/s]", ylabel="E [V/m]",
          title="Field at 916 MHz, 0.30 m")
ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, which="both")

fig.tight_layout()
fig.savefig(figures_dir() / "02_velocity_sweep.png", dpi=140)
if plt:
    print(f"wrote {figures_dir() / '02_velocity_sweep.png'}")

print("\n  v[km/s]   m_vap/m_p    T_e[eV]    Zbar      Q_frozen[C]   E916[V/m]")
for i in range(len(v)):
    print(f"  {v[i]/1e3:7.1f} {res['vapour_efficiency'][i]:11.3f} "
          f"{res['T_eV'][i]:10.3f} {res['Zbar'][i]:9.4f} "
          f"{res['Q_frozen'][i]:14.3e} {res['E_916MHz'][i]:11.3e}")
