"""Example 3 -- sensitivity to the model's free and semi-free parameters.

The framework has four parameters that are not fixed by first principles:

    n_decay                 shock-decay exponent, P ~ r^-n     (1.5 - 3)
    core_scale              isobaric-core radius / projectile radius (0.5 - 1.5)
    plume_expansion_factor  Stage-1 -> Stage-2 handoff density  (2 - 5)
    separation_factor       charge-separation displacement / Debye length (~1)

This script varies each over its physically defensible range and reports the
resulting spread in the observables, so that a quoted prediction can be
accompanied by an honest uncertainty rather than a single number.

Produces `figures/03_sensitivity.png`.
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


BASE = dict(mass=1e-12, velocity=50e3, t_end=1e-5)

SWEEPS = {
    "n_decay": np.linspace(1.5, 3.0, 7),
    "core_scale": np.linspace(0.5, 1.5, 7),
    "plume_expansion_factor": np.linspace(2.0, 5.0, 7),
    "separation_factor": np.linspace(0.5, 2.0, 7),
}

OBS = {
    "m_vapour": lambda s: s.impact.m_vapour,
    "T_e [eV]": lambda s: s.impact.plasma.T_eV,
    "Q_frozen [C]": lambda s: (s.expansion.Q_final if s.expansion else 0.0),
    "E@916MHz [V/m]": lambda s: (s.bands[916e6]["E_peak"]
                                 if s.bands.get(916e6, {}).get("reached")
                                 else 0.0),
}

results = {}
for pname, values in SWEEPS.items():
    rows = {k: [] for k in OBS}
    for val in values:
        kw = dict(BASE)
        kw[pname] = val
        try:
            sc = run_scenario("Fe", "Al", **kw)
        except Exception:
            for k in OBS:
                rows[k].append(np.nan)
            continue
        for k, fn in OBS.items():
            try:
                rows[k].append(fn(sc))
            except Exception:
                rows[k].append(np.nan)
    results[pname] = (values, {k: np.array(v) for k, v in rows.items()})

plt = get_pyplot("the sensitivity figure")

fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
for ax, (pname, (values, rows)) in zip(axes, results.items()):
    for k, arr in rows.items():
        ref = arr[len(arr) // 2]
        if ref and np.isfinite(ref):
            ax.plot(values, arr / ref, "o-", label=k)
    ax.set(xlabel=pname, ylabel="normalised to mid-range value",
           yscale="log", title=pname)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7)
fig.suptitle("Sensitivity of the observables to the free parameters "
             "(Fe 1 pg -> Al at 50 km/s)")
fig.tight_layout()
fig.savefig(figures_dir() / "03_sensitivity.png", dpi=140)
if plt:
    print(f"wrote {figures_dir() / '03_sensitivity.png'}\n")

print(f"{'parameter':<26}{'observable':<18}{'min':>12}{'max':>12}"
      f"{'spread':>10}")
print("-" * 78)
for pname, (values, rows) in results.items():
    for k, arr in rows.items():
        a = arr[np.isfinite(arr) & (arr > 0)]
        if a.size < 2:
            continue
        print(f"{pname:<26}{k:<18}{a.min():>12.3e}{a.max():>12.3e}"
              f"{a.max()/a.min():>10.2f}x")
