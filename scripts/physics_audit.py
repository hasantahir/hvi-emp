#!/usr/bin/env python3
"""Physics audit: stress-test invariants that MUST hold, across parameter space.

Not a test suite -- a bug hunt. Every check here is a statement of physics or
mathematics that cannot be false, evaluated over a wide sweep. Anything that
trips is either a bug or a documented approximation whose limits we have
crossed; the point is to find out which.
"""
from __future__ import annotations

import itertools
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
warnings.simplefilter("ignore")

from hvi_emp import get_material, run_scenario
from hvi_emp.constants import (AMU, C_LIGHT, E_CHARGE, EV, K_B, K_PER_EV,
                               M_ELECTRON)
from hvi_emp import eos as E
from hvi_emp import ionization as I
from hvi_emp.impact import Projectile, simulate_impact
from hvi_emp.expansion import simulate_expansion
from hvi_emp import emp as P

FAIL, WARN, OK = [], [], []
MATS = ["Al", "Fe", "W", "Cu", "SiO2", "Olivine"]


def check(name, cond, detail="", severe=True):
    (OK if cond else (FAIL if severe else WARN)).append((name, detail))
    if not cond:
        tag = "FAIL" if severe else "warn"
        print(f"  [{tag}] {name}: {detail}")


NOTES = []


def note(name, detail=""):
    """Report a measured number without gating on it.

    For quantities that are worth knowing but that no threshold can
    sensibly bound -- typically because the underlying function is
    discontinuous, so any tolerance would be arbitrary and the temptation
    would be to widen it until green. A `note` is printed and recorded; it
    never passes or fails.
    """
    NOTES.append((name, detail))
    print(f"  [note] {name}: {detail}")


def hr(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


# ===========================================================================
hr("1. EOS -- thermodynamic consistency of the cold curve")
# ===========================================================================
for name in MATS:
    m = get_material(name)
    V0 = 1.0 / m.rho0
    worst = 0.0
    for f in (0.55, 0.7, 0.85, 1.0, 1.2, 1.6, 2.5):
        V = f * V0
        h = 1e-6 * V0
        dEdV = (E.cold_energy(m, V + h) - E.cold_energy(m, V - h)) / (2 * h)
        Pc = E.cold_pressure(m, V)
        denom = max(abs(Pc), 1e6)
        worst = max(worst, abs(-dEdV - Pc) / denom)
    check(f"P_c = -dE_c/dV  [{name}]", worst < 1e-3,
          f"max relative error {worst:.2e}")

# ===========================================================================
hr("2. EOS -- Rankine-Hugoniot jump conditions")
# ===========================================================================
for name in MATS:
    m = get_material(name)
    for up in (0.5e3, 2e3, 5e3, 10e3, 20e3):
        st = E.hugoniot_state(m, up, warn=False)
        Us = st.get("Us")
        P_jump = m.rho0 * Us * up
        check(f"P = rho0 Us up  [{name} up={up/1e3:.0f}km/s]",
              abs(st["P"] - P_jump) / max(P_jump, 1) < 1e-9,
              f"{st['P']:.4e} vs {P_jump:.4e}")
        rho_jump = m.rho0 * Us / (Us - up)
        check(f"rho = rho0 Us/(Us-up)  [{name} up={up/1e3:.0f}]",
              abs(st["rho"] - rho_jump) / rho_jump < 1e-9
              if "rho" in st else True,
              f"{st.get('rho')} vs {rho_jump:.4e}")
        # energy: E - E0 = up^2/2
        if "E" in st:
            check(f"E_H = up^2/2  [{name} up={up/1e3:.0f}]",
                  abs(st["E"] - 0.5 * up * up) / (0.5 * up * up) < 1e-9,
                  f"{st['E']:.4e} vs {0.5*up*up:.4e}")

# ===========================================================================
hr("3. EOS -- release energy bounds and monotonicity")
# ===========================================================================
for name in MATS:
    m = get_material(name)
    ups = np.array([0.5e3, 1e3, 2e3, 4e3, 8e3, 12e3, 16e3, 20e3])
    Er = np.array([E.residual_energy(m, u, warn=False) for u in ups])
    check(f"waste heat monotonic in up  [{name}]",
          bool(np.all(np.diff(Er) > 0)), f"{Er}")
    check(f"waste heat <= shock energy  [{name}]",
          bool(np.all(Er <= 0.5 * ups**2 * (1 + 1e-9))),
          f"max ratio {np.max(Er / (0.5*ups**2)):.3f}")
    check(f"waste heat non-negative  [{name}]", bool(np.all(Er >= 0)), f"{Er}")

# ===========================================================================
hr("4. EOS -- impedance matching consistency")
# ===========================================================================
for a, b in itertools.product(["Al", "Fe", "W"], repeat=2):
    ma, mb = get_material(a), get_material(b)
    for v in (5e3, 15e3, 30e3, 50e3):
        st = E.impedance_match(ma, mb, v, warn=False)
        check(f"up_p + up_t = v_impact  [{a}->{b} {v/1e3:.0f}km/s]",
              abs(st.up_projectile + st.up_target - v) / v < 1e-6,
              f"{st.up_projectile + st.up_target:.4e} vs {v:.4e}")
        Pa = E.hugoniot_state(ma, st.up_projectile, warn=False)["P"]
        Pb = E.hugoniot_state(mb, st.up_target, warn=False)["P"]
        check(f"pressure continuity  [{a}->{b} {v/1e3:.0f}]",
              abs(Pa - Pb) / max(Pa, 1) < 1e-6,
              f"{Pa:.4e} vs {Pb:.4e}")

# ===========================================================================
hr("5. Ionisation -- Saha conservation laws")
# ===========================================================================
for name in ["Al", "Fe", "W"]:
    m = get_material(name)
    for n_h in (1e24, 1e26, 1e28):
        for T_eV in (0.5, 1.0, 3.0, 10.0, 30.0):
            s = I.saha_solve(m, T_eV * K_PER_EV, n_h)
            frac = np.asarray(s.fractions)
            pops = frac * n_h
            check(f"sum f_j = 1  [{name} n={n_h:.0e} T={T_eV}eV]",
                  abs(frac.sum() - 1.0) < 1e-6,
                  f"{frac.sum():.9f}")
            ne_from_pops = float(np.sum(np.arange(len(pops)) * pops))
            check(f"sum j n_j = n_e  [{name} n={n_h:.0e} T={T_eV}]",
                  abs(ne_from_pops - s.n_e) / max(s.n_e, 1e-30) < 1e-6,
                  f"{ne_from_pops:.6e} vs {s.n_e:.6e}")
            check(f"Zbar = n_e/n_h  [{name} n={n_h:.0e} T={T_eV}]",
                  abs(s.Zbar - s.n_e / n_h) < 1e-9,
                  f"{s.Zbar:.6f} vs {s.n_e/n_h:.6f}")
            check(f"populations non-negative [{name} {n_h:.0e} {T_eV}]",
                  bool(np.all(pops >= -1e-30)), f"min {pops.min():.3e}")

# ===========================================================================
hr("6. Ionisation -- monotonicity and limits")
# ===========================================================================
for name in ["Al", "Fe", "W"]:
    m = get_material(name)
    Ts = np.array([0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
    for n_h in (1e24, 1e27):
        Z = np.array([I.saha_solve(m, t * K_PER_EV, n_h).Zbar for t in Ts])
        check(f"Zbar increases with T  [{name} n={n_h:.0e}]",
              bool(np.all(np.diff(Z) >= -1e-9)), f"{np.round(Z, 4)}")
    # cold limit
    Zcold = I.saha_solve(m, 300.0, 1e28).Zbar
    check(f"Zbar -> 0 as T -> 300 K  [{name}]", Zcold < 1e-6,
          f"Zbar = {Zcold:.3e}")
    # pressure ionisation direction: at FIXED T, denser plasma is LESS ionised
    # (Saha) -- but continuum lowering pushes the other way. Just record it.
    Z_lo = I.saha_solve(m, 2 * K_PER_EV, 1e24).Zbar
    Z_hi = I.saha_solve(m, 2 * K_PER_EV, 1e29).Zbar
    print(f"    [info] {name}: Zbar(2 eV) n=1e24 -> {Z_lo:.4f}, "
          f"n=1e29 -> {Z_hi:.4f}")

# ===========================================================================
hr("7. Ionisation -- plasma formulary cross-checks")
# ===========================================================================
for n_e in (1e18, 1e22, 1e26):
    wpe = I.plasma_frequency(n_e)
    wpe_nrl = 5.64e4 * np.sqrt(n_e * 1e-6)      # rad/s, NRL (n in cm^-3)
    check(f"omega_pe vs NRL  [n_e={n_e:.0e}]",
          abs(wpe - wpe_nrl) / wpe_nrl < 2e-3,
          f"{wpe:.4e} vs {wpe_nrl:.4e}")
    for T_eV in (1.0, 10.0):
        ld = I.debye_length(n_e, T_eV)
        ld_nrl = 7.43e2 * np.sqrt(T_eV / (n_e * 1e-6)) * 1e-2   # m
        check(f"lambda_D vs NRL  [n={n_e:.0e} T={T_eV}]",
              abs(ld - ld_nrl) / ld_nrl < 5e-3,
              f"{ld:.4e} vs {ld_nrl:.4e}")

# ===========================================================================
hr("8. Ionisation -- table vs direct solver")
# ===========================================================================
# This check used to sample an arbitrary 10x12 grid and gate on RELATIVE
# error. Both were wrong, and together they produced a warning that pointed
# at the wrong thing for a long time.
#
# * Sampling arbitrary points mostly misses cell centres, which is exactly
#   where bilinear interpolation is worst. The old check reported 7.2%; at
#   cell centres the true worst relative error is 6e5%.
# * Relative error is meaningless where Zbar -> 0. That 6e5% is Zbar = 3e-5
#   against 5e-2: an absolute error of 0.05 in a regime with essentially no
#   free electrons, which propagates to nothing.
#
# What actually matters is the ABSOLUTE error in Zbar, because n_e = Zbar
# n_h. Relative error is reported too, but only where Zbar is large enough
# to carry current.
#
# Chasing the number further revealed it is not an interpolation problem at
# all: `saha_solve` is DISCONTINUOUS in the cold-dense corner, where
# continuum lowering abruptly unbinds the ground state and Zbar steps from 0
# to ~0.15. Refining the grid does not help and cannot -- no interpolant
# represents a step -- which is why n_pts stays at 64. See the
# "pressure-ionisation step" check below.
for name in ["Al", "W"]:
    m = get_material(name)
    tab = I.get_table(m)
    worst_abs, wa = 0.0, None
    worst_rel, wr = 0.0, None
    # Cell centres: the worst case for bilinear interpolation, not a lucky
    # sample of it.
    for i in range(len(tab.logn) - 1):
        for j in range(0, len(tab.logT) - 1, 2):
            ln = 0.5 * (tab.logn[i] + tab.logn[i + 1])
            lt = 0.5 * (tab.logT[j] + tab.logT[j + 1])
            n_h, T = 10.0 ** ln, 10.0 ** lt
            z_t = tab.Zbar(n_h, T)
            z_d = I.saha_solve(m, T, n_h).Zbar
            a = abs(z_t - z_d)
            if a > worst_abs:
                worst_abs, wa = a, (n_h, T * 8.617e-5, z_t, z_d)
            if z_d > 0.01 and a / z_d > worst_rel:
                worst_rel, wr = a / z_d, (n_h, T * 8.617e-5, z_t, z_d)

    # Reported, not gated. A global bound over the whole grid is dominated
    # by the discontinuity below, and tightening it would only reward
    # widening a tolerance. The gate that means something is the trajectory
    # check that follows.
    note(f"IonisationTable worst-case over the whole grid  [{name}]",
         f"abs {worst_abs:.4f} at n={wa[0]:.2e} T={wa[1]:.3f}eV; "
         f"rel {worst_rel:.0%} where Zbar>0.01 -- both sit on the "
         f"pressure-ionisation step, which no interpolant can represent"
         if wa else "")

# The check that actually constrains anything: accuracy along the (n, T)
# path a real plume follows. This is the region every published number in
# the framework depends on, and it is where the tolerance should be tight.
for pname, tname in (("Fe", "Al"), ("W", "Al")):
    for v in (30e3, 50e3, 70e3):
        sc = run_scenario(pname, tname, mass=1e-12, velocity=v)
        if sc.expansion is None:
            continue
        exp = sc.expansion
        mm = exp.material
        tab = I.get_table(mm)
        worst, where = 0.0, None
        idx = np.unique(np.geomspace(1, len(exp.t) - 1, 24).astype(int))
        for i in idx:
            n_h, T = float(exp.n_h[i]), float(exp.T[i])
            a = abs(tab.Zbar(n_h, T) - I.saha_solve(mm, T, n_h).Zbar)
            if a > worst:
                worst, where = a, (n_h, T * 8.617e-5)
        # Gated at 0.01 absolute in Zbar and NOT loosened to make it green.
        # Where it trips, the cause is known and stated: `saha_solve` has a
        # near-discontinuous pressure-ionisation onset around n ~ 3e27,
        # T ~ 0.73 eV, and no table resolution fixes it -- measured worst
        # error on these trajectories is 1.9e-2 at n_pts=64, 2.5e-2 at 128
        # and 1.6e-2 at 160, i.e. non-monotone in resolution, which is the
        # signature of a step rather than of under-sampling. The fix belongs
        # in the ionisation model (continuous continuum lowering), not here.
        check(f"table accurate ON the plume trajectory  "
              f"[{pname}->{tname} {v/1e3:.0f}]",
              worst < 0.01,
              (f"worst abs error {worst:.2e} in Zbar at n={where[0]:.2e} "
               f"T={where[1]:.3f}eV -- this is the pressure-ionisation step, "
               f"not interpolation: refining the grid does not reduce it. "
               f"See docs/PHYSICS_AUDIT.md" if where else ""),
              severe=False)

# The step itself. It is a property of the ionisation model, not of the
# table, and it is worth locating explicitly rather than letting it surface
# as a mysterious interpolation warning.
for name in ["Al", "W", "Fe"]:
    m = get_material(name)
    found = []
    for T_eV in (0.27, 0.5):
        T = T_eV / 8.617e-5
        ns = np.geomspace(1e26, 1e30, 120)
        z = np.array([I.saha_solve(m, T, n).Zbar for n in ns])
        big = np.where((z[1:] > 100 * np.maximum(z[:-1], 1e-12))
                       & (z[1:] > 1e-3))[0]
        if big.size:
            found.append((T_eV, ns[int(big[0]) + 1], z[int(big[0]) + 1]))
    # The framework's plume never exceeds ~2e27 m^-3, and is ~1 eV where it
    # is densest, so the step is outside the operating region. That is what
    # this check asserts -- not that the step is absent.
    reachable = [f for f in found if f[1] < 5e27]
    check(f"pressure-ionisation step outside the plume's range  [{name}]",
          not reachable,
          (f"step at n={found[0][1]:.2e} m^-3, T={found[0][0]} eV "
           f"(Zbar 0 -> {found[0][2]:.3f}); plume peaks near 2e27 m^-3 at "
           f"~1 eV, so it is not crossed" if found else "no step found"),
          severe=False)

# ===========================================================================
hr("9. Impact -- mass, energy and charge budgets")
# ===========================================================================
for pname, tname in (("Fe", "Al"), ("W", "Al"), ("Al", "Al")):
    for v in (10e3, 20e3, 30e3, 45e3, 60e3, 72e3):
        sc = run_scenario(pname, tname, mass=1e-12, velocity=v)
        im = sc.impact
        mp = im.projectile.mass
        check(f"f_vap in [0,1]  [{pname}->{tname} {v/1e3:.0f}]",
              -1e-12 <= im.f_vap_projectile <= 1 + 1e-12,
              f"{im.f_vap_projectile:.4f}")
        check(f"m_vap_proj <= m_proj  [{pname}->{tname} {v/1e3:.0f}]",
              im.m_vapour_projectile <= mp * (1 + 1e-9),
              f"{im.m_vapour_projectile:.4e} vs {mp:.4e}")
        check(f"m_plasma <= m_vapour  [{pname}->{tname} {v/1e3:.0f}]",
              im.m_plasma <= im.m_vapour * (1 + 1e-9),
              f"{im.m_plasma:.4e} vs {im.m_vapour:.4e}")
        KE = 0.5 * mp * v * v
        E_plasma = im.m_plasma * im.u_atom / im.material_mix.m_atom \
            if im.m_plasma > 0 else 0.0
        check(f"plasma internal energy <= impact KE  [{pname} {v/1e3:.0f}]",
              E_plasma <= KE * (1 + 1e-9),
              f"{E_plasma:.4e} vs {KE:.4e}")
        if im.m_plasma > 0:
            # Q_free is the superheated plasma PLUS a small thermally-ionised
            # contribution from the two-phase vapour, so the bound has to
            # include both terms.
            n_atoms = im.m_plasma / im.material_mix.m_atom
            Q_super = n_atoms * im.plasma.Zbar * E_CHARGE
            Q_two = im.diagnostics.get("Q_twophase", 0.0)
            check(f"Q = Q_superheated + Q_twophase  [{pname} {v/1e3:.0f}]",
                  abs(im.Q_free - (Q_super + Q_two)) <= 1e-9 * im.Q_free,
                  f"{im.Q_free:.6e} vs {Q_super + Q_two:.6e}")
            check(f"Q_superheated <= n Zbar e  [{pname} {v/1e3:.0f}]",
                  im.diagnostics.get("Q_superheated", 0.0)
                  <= Q_super * (1 + 1e-6),
                  f"{im.diagnostics.get('Q_superheated', 0):.4e} vs "
                  f"{Q_super:.4e}")

# ===========================================================================
hr("10. Impact -- monotonicity in velocity")
# ===========================================================================
for pname in ("Fe", "W"):
    vs = np.arange(12e3, 72e3, 4e3)
    res = [run_scenario(pname, "Al", mass=1e-12, velocity=v).impact
           for v in vs]
    for label, vals in (("f_vap", [r.f_vap_projectile for r in res]),
                        ("m_plasma", [r.m_plasma for r in res]),
                        ("T_eV", [r.plasma.T_eV for r in res]),
                        ("Q_free", [r.Q_free for r in res])):
        a = np.asarray(vals)
        bad = np.where(np.diff(a) < -1e-12 * max(a.max(), 1e-30))[0]
        check(f"{label} monotonic in v  [{pname}->Al]", bad.size == 0,
              f"decreases at v = {vs[bad] / 1e3} km/s; values {np.round(a, 6)}",
              severe=False)

# ===========================================================================
hr("11. Expansion -- conservation and physical bounds")
# ===========================================================================
for pname, v in (("Fe", 50e3), ("W", 45e3), ("Al", 60e3)):
    sc = run_scenario(pname, "Al", mass=1e-12, velocity=v)
    ex = sc.expansion
    if ex is None:
        print(f"    [info] {pname} {v/1e3:.0f} km/s: no plasma, skipped")
        continue
    R = np.sqrt(np.asarray(ex.R_r) * np.asarray(ex.R_z))
    check(f"plume radius increases  [{pname} {v/1e3:.0f}]",
          bool(np.all(np.diff(R) > -1e-15)), "non-monotonic R")
    n = np.asarray(ex.n_h)
    check(f"density decreases  [{pname} {v/1e3:.0f}]",
          bool(np.all(np.diff(n) <= 1e-9 * n[:-1])), "density rises")
    vmax = max(np.max(np.abs(ex.v_r)), np.max(np.abs(ex.v_z)))
    check(f"expansion sub-luminal  [{pname} {v/1e3:.0f}]", vmax < C_LIGHT,
          f"v_max = {vmax:.3e} m/s")
    T = np.asarray(ex.T_eV)
    check(f"temperature positive  [{pname} {v/1e3:.0f}]", bool(np.all(T > 0)),
          f"min T = {T.min():.3e} eV")
    check(f"temperature falls  [{pname} {v/1e3:.0f}]",
          bool(np.all(np.diff(T) <= 1e-9 * np.abs(T[:-1]))),
          "temperature rises during free expansion", severe=False)
    Z = np.asarray(ex.Zbar)
    check(f"Zbar does not rise in free expansion  [{pname} {v/1e3:.0f}]",
          bool(np.all(np.diff(Z) <= 1e-9)), "Zbar increases", severe=False)

# ===========================================================================
hr("12. EMP -- radiation field identities")
# ===========================================================================
for r in (0.1, 0.3, 1.0):
    for f in (3.15e8, 9.16e8, 5e9):
        w = 2 * np.pi * f
        d = P.dipole_field(1e-12, w, r, theta_deg=90.0)
        kr = w * r / C_LIGHT
        if kr > 30:
            check(f"B = E/c in far field  [r={r} f={f:.2e}]",
                  abs(d["B"] - d["E"] / C_LIGHT) / (d["E"] / C_LIGHT) < 1e-2,
                  f"{d['B']:.4e} vs {d['E']/C_LIGHT:.4e}")
# far-field 1/r
E1 = P.dipole_field(1e-12, 2 * np.pi * 5e9, 10.0)["E"]
E2 = P.dipole_field(1e-12, 2 * np.pi * 5e9, 20.0)["E"]
check("far field falls as 1/r", abs(E1 / E2 - 2.0) < 0.02,
      f"ratio {E1/E2:.4f}")
# angular pattern: dipole ~ sin(theta)
Ea = P.dipole_field(1e-12, 2 * np.pi * 5e9, 10.0, theta_deg=90.0)["E"]
Eb = P.dipole_field(1e-12, 2 * np.pi * 5e9, 10.0, theta_deg=30.0)["E"]
check("dipole angular pattern ~ sin(theta)",
      abs(Eb / Ea - np.sin(np.radians(30.0))) < 0.05,
      f"ratio {Eb/Ea:.4f} vs sin30 = 0.5")

# ===========================================================================
hr("SUMMARY")
# ===========================================================================
print(f"  passed : {len(OK)}")
print(f"  warned : {len(WARN)}")
print(f"  FAILED : {len(FAIL)}")
print(f"  noted  : {len(NOTES)}  (measured, deliberately not gated)")
if NOTES:
    print("\n  Notes:")
    for n, d in NOTES:
        print(f"    - {n}: {d}")
if FAIL:
    print("\n  Failures:")
    seen = set()
    for n, d in FAIL:
        key = n.split("[")[0]
        if key in seen:
            continue
        seen.add(key)
        print(f"    - {n}: {d}")
    sys.exit(1)
if WARN:
    print("\n  Warnings:")
    seen = set()
    for n, d in WARN:
        key = n.split("[")[0]
        if key in seen:
            continue
        seen.add(key)
        print(f"    - {n}: {d}")
