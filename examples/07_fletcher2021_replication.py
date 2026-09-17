"""Example 7 -- replicate Fletcher (2021), NRL/MR/6757-20-10,138.

That report used ALEGRA, a Sandia code that cannot be obtained. This script
sets up the closest reproducible equivalent and, where the framework can
answer directly, checks its quantitative claims now.

What it does
------------
1.  Runs Fletcher's six impact speeds (12-62 km/s, W -> Al) through the
    reduced chain and prints the vapour/plasma inventory, T_e, Zbar and the
    charge yield -- a direct test of his stated **10-20 km/s complete-
    vaporisation threshold**.
2.  Writes an M2C run directory per speed (his Figs. 4-5: mass density at
    12 us, pressure at 2.5 us), auto-sized so the mesh contains the crater,
    with a cost estimate for your core count.
3.  Writes both OpenMHD problems (his Figs. 6-7: field parallel and
    perpendicular to the surface -- the diamagnetic cavity).
4.  Writes a WarpX deck for the EMP stage that ALEGRA hands to a PIC.

Nothing here needs M2C, OpenMHD or WarpX installed: it generates the inputs
and tells you what to run. See docs/REPLICATION_FLETCHER2021.md.

    python examples/07_fletcher2021_replication.py --cores 128
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.simplefilter("ignore")

import numpy as np

from _plotting import figures_dir, get_pyplot
from hvi_emp import Projectile, get_material, run_scenario, simulate_impact
from hvi_emp.solvers.m2c_stage1 import M2CConfig, write_problem_directory
from hvi_emp.solvers.isale_stage1 import (FLETCHER_T_DENSITY,
                                          FLETCHER_VELOCITIES,
                                          FLETCHER_T_PRESSURE)
from hvi_emp.solvers.openmhd_stage2 import write_fletcher_mhd_cases
from hvi_emp.solvers.warpx_stage3 import (WarpXEMPConfig, write_picmi_script,
                                          write_warpx_inputs)

ap = argparse.ArgumentParser()
ap.add_argument("--cores", type=int, default=128)
ap.add_argument("--diameter", type=float, default=1.0e-3,
                help="projectile diameter [m]; the report does not state it")
ap.add_argument("--out", default="fletcher2021",
                help="directory for the generated solver inputs")
args = ap.parse_args()

OUT = Path(args.out).resolve()
OUT.mkdir(exist_ok=True)

print("=" * 78)
print("Replication of Fletcher (2021), NRL/MR/6757-20-10,138")
print("  original code: ALEGRA (Sandia, unobtainable)")
print("  substitutes:   M2C (hydro + ionisation) + OpenMHD (MHD) "
      "+ WarpX (PIC)")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1. The velocity series through the reduced chain -- testable right now
# ---------------------------------------------------------------------------
print("\n[1] Velocity series, W -> Al (his Figs. 4-5 conditions)\n")
print(f"{'v [km/s]':>9}{'P [GPa]':>10}{'m_vap/m_p':>11}{'m_pl/m_p':>10}"
      f"{'T_e [eV]':>10}{'Zbar':>8}{'Q [C]':>12}")

rows = []
W, AL = get_material("W"), get_material("Al")
m_proj = W.rho0 * np.pi * args.diameter ** 3 / 6.0
for v in FLETCHER_VELOCITIES:
    imp = simulate_impact(Projectile(W, m_proj, v), AL, warn=False)
    rows.append({
        "v": v, "P": imp.P_ic, "f_vap": imp.vaporisation_efficiency,
        "f_pl": imp.plasma_efficiency, "T_eV": imp.plasma.T_eV,
        "Zbar": imp.plasma.Zbar, "Q": imp.Q_free})
    print(f"{v/1e3:>9.0f}{imp.P_ic/1e9:>10.0f}"
          f"{imp.vaporisation_efficiency:>11.2f}"
          f"{imp.plasma_efficiency:>10.2f}{imp.plasma.T_eV:>10.2f}"
          f"{imp.plasma.Zbar:>8.3f}{imp.Q_free:>12.3e}")

# --- his threshold claim, tested like-for-like -----------------------------
# He writes: "a threshold velocity that lies between 10 km/s and 20 km/s at
# which the projectile is entirely vaporized". The matching quantity here is
# f_vap_projectile reaching 1.0 -- NOT the appearance of a superheated
# plasma-forming component, which is a stricter and later condition. Both are
# reported so the comparison cannot be quietly rigged by choosing the
# flattering definition.
print("\n[1a] Thresholds (he reports 10-20 km/s for complete projectile")
print("     vaporisation, W -> Al)")
v_scan = np.arange(4e3, 45e3, 0.25e3)
v_proj_vap = v_superheat = np.nan
for v in v_scan:
    r = simulate_impact(Projectile(W, m_proj, v), AL, warn=False)
    if np.isnan(v_proj_vap) and r.f_vap_projectile >= 1.0:
        v_proj_vap = v
    if np.isnan(v_superheat) and r.plasma_efficiency > 0:
        v_superheat = v
    if not (np.isnan(v_proj_vap) or np.isnan(v_superheat)):
        break

ok = 10e3 <= v_proj_vap <= 20e3
print(f"     projectile fully vaporised   : {v_proj_vap/1e3:5.1f} km/s"
      f"   <- compare with his 10-20 km/s   "
      f"[{'CONSISTENT' if ok else 'OUTSIDE'}]")
print(f"     superheated plasma appears   : {v_superheat/1e3:5.1f} km/s"
      "   (stricter condition, no counterpart in the report)")
v_thresh = v_proj_vap

# ---------------------------------------------------------------------------
# 2. M2C decks for Figs. 4-5 (and 6)
# ---------------------------------------------------------------------------
# This used to generate iSALE decks. It no longer does: access requests went
# unanswered, and M2C covers the same stage with no gatekeeper, does the
# oblique Fig. 6 case that iSALE2D structurally cannot, and solves ionisation
# inside the hydro step -- which is what makes the Section 5 threshold
# argument testable rather than merely arguable.
print("\n[2] M2C decks (Figs. 4-5: density at 12 us, pressure at 2.5 us)\n")
m2c_dir = OUT / "m2c"
m2c_cases = []
for v in FLETCHER_VELOCITIES:
    cfg = M2CConfig(projectile=get_material("w"), target=get_material("al"),
                    diameter=args.diameter, velocity=v,
                    t_end=FLETCHER_T_DENSITY, n_outputs=60,
                    ionisation="non-ideal", depression_model="Ebeling")
    res = write_problem_directory(cfg, str(m2c_dir / f"v{v/1e3:.0f}kms"),
                                  cores=args.cores)
    m2c_cases.append((v, cfg, res))

print(f"{'v [km/s]':>9}{'mesh':>16}{'Mcells':>9}{'dt [s]':>11}{'est h':>8}")
total = 0.0
for v, cfg, res in m2c_cases:
    e = res["estimate"]
    total += e["wall_hours_estimate"]
    print(f"{v/1e3:>9.0f}{e['dimensionality']:>16}{e['cells']/1e6:>9.2f}"
          f"{e['dt_s']:>11.2e}{e['wall_hours_estimate']:>8.2f}")
print(f"\n     {len(m2c_cases)} cases on {args.cores} cores, run "
      f"sequentially -> ~{total:.1f} h total")
print("     ORDER OF MAGNITUDE ONLY -- cell counts and a CFL step, "
      "not measured")
print(f"     cd {m2c_dir}/v12kms && mpirun -np {args.cores} "
      "${M2C_HOME:?set it to the directory holding the m2c binary}/m2c input.st")

# Fig. 6 is a 30-degree oblique impact. Neither the reduced chain nor an
# axisymmetric hydrocode can represent it at all; M2C switches to 3-D.
print("\n[2b] M2C oblique case (Fig. 6: 30 deg Al->Al at 30 km/s)\n")
obl = M2CConfig(projectile=get_material("al"), target=get_material("al"),
                diameter=args.diameter, velocity=30e3, angle_deg=30.0,
                t_end=FLETCHER_T_DENSITY)
res_obl = write_problem_directory(obl, str(m2c_dir / "fig6_oblique30"),
                                  cores=args.cores)
e = res_obl["estimate"]
print(f"     {e['dimensionality']}, {e['cells']/1e6:.0f}M cells, "
      f"{e['memory_GB']:.1f} GB, ~{e['wall_hours_estimate']:.1f} h")
for n in res_obl["notes"]:
    print(f"     ! {n}")

# ---------------------------------------------------------------------------
# 3. OpenMHD decks for Figs. 6-7
# ---------------------------------------------------------------------------
print("\n[3] OpenMHD decks (Figs. 6-7: the two field orientations)\n")
sc50 = run_scenario("W", "Al", mass=m_proj, velocity=30e3, t_end=1e-4)
mhd = write_fletcher_mhd_cases(str(OUT / "openmhd"), exp=sc50.expansion)
for name, d in mhd.items():
    cav = d["cavity"]
    print(f"     {name:<20} R_cavity ~ {cav['R_cavity']:.2f} m, "
          f"f ~ {cav['f_characteristic']:.0f} Hz")
print(f"     written to {OUT / 'openmhd'}")

# ---------------------------------------------------------------------------
# 4. WarpX deck for the PIC stage
# ---------------------------------------------------------------------------
print("\n[4] WarpX deck (the PIC half of his hydrocode -> PIC waterfall)\n")
warpx_dir = OUT / "warpx"
warpx_dir.mkdir(exist_ok=True)
wcfg = WarpXEMPConfig.from_expansion(sc50.expansion, at_frequency=916e6)
g = wcfg.grid()
write_picmi_script(wcfg, str(warpx_dir / "warpx_emp.py"))
write_warpx_inputs(wcfg, str(warpx_dir / "inputs"))
print(f"     n_e0 = {wcfg.n_e0:.2e} m^-3, f_pe = {wcfg.omega_pe/2/np.pi:.2e} Hz")
print(f"     grid {g['n_cells']}^2, {g['n_steps']:,} steps, "
      f"Debye resolved: {g['debye_resolved']}")
print(f"     written to {warpx_dir}")

# ---------------------------------------------------------------------------
# 5. Figure
# ---------------------------------------------------------------------------
plt = get_pyplot("the replication summary figure")
if plt:
    v = np.array([r["v"] for r in rows]) / 1e3
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(v, [r["P"] / 1e9 for r in rows], "o-")
    ax[0].set(xlabel="impact speed [km/s]", ylabel="peak shock pressure [GPa]",
              title="Shock pressure (cf. his Fig. 5)")
    ax[0].grid(alpha=.3)

    ax[1].plot(v, [r["f_vap"] for r in rows], "o-", label=r"$m_{vap}/m_p$")
    ax[1].plot(v, [r["f_pl"] for r in rows], "s-", label=r"$m_{plasma}/m_p$")
    if np.isfinite(v_thresh):
        ax[1].axvline(v_thresh / 1e3, color="k", ls="--", lw=1,
                      label=f"proj. fully vaporised {v_thresh/1e3:.0f} km/s")
    if np.isfinite(v_superheat):
        ax[1].axvline(v_superheat / 1e3, color="k", ls=":", lw=1,
                      label=f"superheat onset {v_superheat/1e3:.0f} km/s")
    ax[1].axvspan(10, 20, color="g", alpha=.15,
                  label="Fletcher: 10-20 km/s")
    ax[1].set(xlabel="impact speed [km/s]", ylabel="mass / projectile mass",
              title="Vaporisation (cf. his Fig. 4)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    ax[2].plot(v, [r["T_eV"] for r in rows], "o-", label=r"$T_e$ [eV]")
    ax[2].plot(v, [r["Zbar"] for r in rows], "s-", label=r"$\bar{Z}$")
    ax[2].axhspan(2.0, 2.5, color="g", alpha=.15,
                  label="Fletcher & Close: 2-2.5 eV")
    ax[2].set(xlabel="impact speed [km/s]", title="Plume state")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    fig.suptitle("Fletcher (2021) replication: reduced chain, "
                 f"W {args.diameter*1e3:.1f} mm -> Al")
    fig.tight_layout()
    out = figures_dir() / "07_fletcher2021.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")

print("\n" + "=" * 78)
print("Next steps are in docs/REPLICATION_FLETCHER2021.md:")
print("  - git clone https://github.com/kevinwgy/m2c (GPLv3, no "
      "application needed)")
print("  - M2C ships Tillotson and ANEOS, so tungsten is no longer a blocker")
print("  - run the 12 and 22 km/s cases first: they settle the threshold "
      "dispute")
print("  - what cannot be replicated at all is listed in section 6")
print("=" * 78)
