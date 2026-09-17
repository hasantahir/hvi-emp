"""Example 1 -- one impact, end to end, with diagnostic plots.

Fe micrometeoroid on an aluminium spacecraft panel at 50 km/s.
Produces `figures/01_single_impact.png`.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.simplefilter("ignore")

import numpy as np

from _plotting import figures_dir, get_pyplot

from hvi_emp import run_scenario


sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3, t_end=2e-5,
                  r_sensor=0.30, bands=(315e6, 916e6))
print(sc.summary())

plt = get_pyplot("the 6-panel diagnostic figure")

e, em = sc.expansion, sc.emp
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))

ax[0, 0].loglog(e.t, e.n_e, label=r"$n_e$ (peak)")
ax[0, 0].loglog(e.t, e.n_h, "--", label=r"$n_h$")
ax[0, 0].set(xlabel="t [s]", ylabel=r"density [m$^{-3}$]", title="Plume density")
ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

ax[0, 1].loglog(e.t, e.T_eV, label=r"$T_e$")
ax[0, 1].loglog(e.t, np.maximum(e.Zbar, 1e-6), label=r"$\bar{Z}$ (frozen)")
ax[0, 1].loglog(e.t, np.maximum(e.Zbar_eq, 1e-6), ":", label=r"$\bar{Z}$ (Saha eq.)")
ax[0, 1].set(xlabel="t [s]", ylabel="eV  /  charge state",
             title="Temperature and ionisation")
ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

ax[0, 2].loglog(e.t, e.nu_ei, label=r"$\nu_{ei}$")
ax[0, 2].loglog(e.t, e.omega_pe, label=r"$\omega_{pe}$")
if e.transition:
    ax[0, 2].axvline(e.transition["t"], color="k", ls="--", lw=1,
                     label="collisionless transition")
ax[0, 2].set(xlabel="t [s]", ylabel=r"rad s$^{-1}$",
             title="Collisional $\\to$ collisionless")
ax[0, 2].legend(); ax[0, 2].grid(alpha=.3)

ax[1, 0].loglog(em.f[1:], np.abs(em.E_spec[1:]))
for f, d in sc.bands.items():
    if d.get("reached"):
        ax[1, 0].axvline(f, color="r", ls=":", lw=1)
        ax[1, 0].text(f, ax[1, 0].get_ylim()[1], f" {f/1e6:.0f} MHz",
                      rotation=90, va="top", fontsize=8, color="r")
ax[1, 0].set(xlabel="f [Hz]", ylabel=r"$|E(f)|$ [V m$^{-1}$ Hz$^{-1}$]",
             title=f"EMP spectrum at {em.r_sensor:.2f} m")
ax[1, 0].grid(alpha=.3)

n = min(len(em.t), 1200)
ax[1, 1].plot(em.t[:n] * 1e9, em.E_t[:n] * 1e3, lw=.8)
ax[1, 1].set(xlabel="t [ns] (retarded)", ylabel="E [mV/m]",
             title="EMP waveform")
ax[1, 1].grid(alpha=.3)

c = sc.charging
ax[1, 2].semilogx(c.t, c.V_dielectric, label="dielectric potential")
ax[1, 2].semilogx(c.t, c.phi_float, label="floating potential")
ax[1, 2].set(xlabel="t [s]", ylabel="V", title="Surface charging")
ax[1, 2].legend(); ax[1, 2].grid(alpha=.3)

fig.suptitle("Fe (1 pg) -> Al at 50 km/s", fontsize=13)
fig.tight_layout()
fig.savefig(figures_dir() / "01_single_impact.png", dpi=140)
if plt:
    print(f"\nwrote {figures_dir() / '01_single_impact.png'}")
