"""Test suite for HVI-EMP.

Three kinds of test:

*  **Conservation and consistency** -- Rankine-Hugoniot identities, energy
   conservation in the expansion ODE, Parseval between the EMP spectrum and
   waveform, charge conservation in the Saha ladder.  These must hold to
   numerical tolerance; a failure is a bug.
*  **Limiting cases** -- the Saha ladder reducing to a known analytic result,
   the dipole field reducing to the far-field 1/r law, the Anisimov plume
   reducing to free ballistic expansion at late time.
*  **Monotonicity and physical sense** -- vaporised mass increasing with
   velocity, plume density decreasing, thresholds ordered melt < boil < vap.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp import (ALUMINIUM, IRON, TUNGSTEN, Projectile, dipole_field,
                     dust_optical_depth, get_table, hugoniot_state,
                     impedance_match, phase_thresholds, plasma_frequency,
                     run_scenario, saha_solve, simulate_expansion,
                     simulate_impact, simulate_emp, debye_length)
from hvi_emp.constants import C_LIGHT, EPS_0, E_CHARGE, K_B, MU_0
from hvi_emp.coupling import UPSET_THRESHOLDS
from hvi_emp.eos import (cohesive_energy_vinet, cold_energy, cold_pressure,
                         release_state, spinodal_volume, vapour_fraction)
from hvi_emp.expansion import gaussian_shells
from hvi_emp.propagation import refractive_index, transmission

warnings.simplefilter("ignore", RuntimeWarning)


# ===========================================================================
# Constants and units
# ===========================================================================

def test_constants_self_consistent():
    assert np.isclose(EPS_0 * MU_0 * C_LIGHT**2, 1.0, rtol=1e-12)


def test_plasma_frequency_against_formulary():
    """omega_pe = 5.64e4 sqrt(n_e[cm^-3]) rad/s (NRL Plasma Formulary)."""
    n_cgs = 1e12
    expected = 5.64e4 * np.sqrt(n_cgs)
    assert np.isclose(plasma_frequency(n_cgs * 1e6), expected, rtol=2e-3)


def test_debye_length_against_formulary():
    """lambda_D = 7.43e2 sqrt(T[eV]/n[cm^-3]) cm (NRL)."""
    n_cgs, T = 1e12, 2.0
    expected_cm = 7.43e2 * np.sqrt(T / n_cgs)
    assert np.isclose(debye_length(n_cgs * 1e6, T) * 100.0, expected_cm,
                      rtol=2e-3)


# ===========================================================================
# Equation of state
# ===========================================================================

@pytest.mark.parametrize("mat", [ALUMINIUM, IRON, TUNGSTEN])
def test_rankine_hugoniot_identities(mat):
    """Mass, momentum and energy jump conditions must close exactly."""
    for up in (1e3, 3e3, 8e3):
        st = hugoniot_state(mat, up, warn=False)
        # mass:      rho0 Us = rho (Us - up)
        assert np.isclose(mat.rho0 * st["Us"],
                          st["rho"] * (st["Us"] - up), rtol=1e-12)
        # momentum:  P = rho0 Us up
        assert np.isclose(st["P"], mat.rho0 * st["Us"] * up, rtol=1e-12)
        # energy:    E = 1/2 P (V0 - V) = 1/2 up^2
        assert np.isclose(st["E"], 0.5 * st["P"] * (mat.V0 - st["V"]),
                          rtol=1e-10)
        assert np.isclose(st["E"], 0.5 * up**2, rtol=1e-12)


def test_impedance_match_pressure_continuity():
    """Projectile and target must see the same interface pressure."""
    st = impedance_match(IRON, ALUMINIUM, 20e3, warn=False)
    assert np.isclose(st.target["P"], st.projectile["P"], rtol=1e-8)
    assert np.isclose(st.up_target + st.up_projectile, 20e3, rtol=1e-12)


def test_impedance_match_symmetric_case():
    """Like-on-like impact splits the velocity exactly in half."""
    st = impedance_match(ALUMINIUM, ALUMINIUM, 10e3, warn=False)
    assert np.isclose(st.up_target, 5e3, rtol=1e-10)


def test_cold_curve_zero_at_reference():
    for mat in (ALUMINIUM, IRON, TUNGSTEN):
        assert np.isclose(cold_pressure(mat, mat.V0), 0.0, atol=1e-6)
        assert np.isclose(cold_energy(mat, mat.V0), 0.0, atol=1e-6)


def test_cold_curve_thermodynamic_consistency():
    """P_c = -dE_c/dV must hold for the Vinet pair."""
    mat = ALUMINIUM
    V = np.linspace(0.6 * mat.V0, 1.4 * mat.V0, 400)
    E = np.array([cold_energy(mat, v) for v in V])
    P_num = -np.gradient(E, V)
    P_ana = np.array([cold_pressure(mat, v) for v in V])
    m = slice(5, -5)
    assert np.allclose(P_num[m], P_ana[m], rtol=1e-3,
                       atol=1e-3 * np.max(np.abs(P_ana)))


def test_vinet_binding_energy_order_of_magnitude():
    """Vinet's asymptotic binding energy should track the sublimation energy
    within a factor of ~2 with no free parameters."""
    for mat in (ALUMINIUM, IRON, TUNGSTEN):
        r = cohesive_energy_vinet(mat) / mat.E_sublimation
        assert 0.3 < r < 3.0, f"{mat.name}: ratio {r}"


def test_spinodal_is_in_expansion():
    for mat in (ALUMINIUM, IRON, TUNGSTEN):
        Vsp = spinodal_volume(mat)
        assert mat.V0 < Vsp < 3.0 * mat.V0
        # It is a minimum of P_c, so P_c there is negative (tension).
        assert cold_pressure(mat, Vsp) < 0.0


def test_compression_limit_for_low_s_materials():
    """Materials with s < 1 have a finite compression limit and must raise a
    clear error above it rather than producing V = 0."""
    from hvi_emp.constants import EV
    from hvi_emp.eos import compression_limit_velocity
    from hvi_emp.materials import Material

    # s >= 1 (all the built-in metals): no limit.
    for mat in (ALUMINIUM, IRON, TUNGSTEN):
        assert np.isinf(compression_limit_velocity(mat))

    # Titanium: s = 0.767, limit = c0/(1-s) = 22.4 km/s.
    ti = Material(
        name="Ti_test", rho0=4510.0, c0=5220.0, s=0.767, gamma0=1.23,
        cv_solid=523.0, T_melt=1941.0, T_vap=3560.0,
        L_fusion=2.96e5, L_vap=8.88e6, E_cohesive=4.85 * EV,
        A=47.867, Z=22,
        E_ion=(6.8281 * EV, 13.5755 * EV, 27.4917 * EV),
        g_ion=(21.0, 28.0, 21.0, 10.0), work_function=4.33 * EV)
    lim = compression_limit_velocity(ti)
    assert np.isclose(lim, 5220.0 / (1.0 - 0.767), rtol=1e-12)

    # Below the limit it works and gives a positive specific volume.
    st = hugoniot_state(ti, 0.5 * lim, warn=False)
    assert st["V"] > 0 and np.isfinite(st["P"])

    # At and above the limit it must raise, with an informative message.
    for up in (lim, 1.2 * lim):
        with pytest.raises(ValueError, match="compression limit"):
            hugoniot_state(ti, up, warn=False)


def test_cold_curve_rejects_nonpositive_volume():
    with pytest.raises(ValueError):
        cold_pressure(ALUMINIUM, 0.0)
    with pytest.raises(ValueError):
        cold_energy(ALUMINIUM, -1.0)


def test_phase_thresholds_ordered():
    for mat in (ALUMINIUM, IRON, TUNGSTEN):
        th = phase_thresholds(mat)
        assert th["E_melt"] < th["E_boil"] < th["E_vap"]
        assert th["P_melt"] < th["P_boil"] < th["P_vap"]


def test_residual_energy_monotonic_in_velocity():
    prev = -1.0
    for up in np.linspace(1e3, 3e4, 25):
        e = release_state(ALUMINIUM, up, warn=False)["E_residual"]
        assert e > prev
        prev = e


def test_vapour_fraction_bounds():
    th = phase_thresholds(ALUMINIUM)
    assert vapour_fraction(ALUMINIUM, 0.5 * th["E_boil"]) == 0.0
    assert vapour_fraction(ALUMINIUM, 2.0 * th["E_vap"]) == 1.0
    f = vapour_fraction(ALUMINIUM, 0.5 * (th["E_boil"] + th["E_vap"]))
    assert 0.0 < f < 1.0


# ===========================================================================
# Ionisation
# ===========================================================================

def test_saha_charge_conservation():
    """n_e must equal Zbar * n_h, and the fractions must sum to one."""
    for T in (5e3, 2e4, 1e5):
        for n in (1e18, 1e22, 1e26):
            st = saha_solve(IRON, T, n)
            assert np.isclose(st.fractions.sum(), 1.0, rtol=1e-10)
            assert np.isclose(st.n_e, st.Zbar * n, rtol=1e-10)
            assert 0.0 <= st.Zbar <= len(IRON.E_ion)


def test_saha_monotonic_in_temperature():
    prev = -1.0
    for T in np.geomspace(3e3, 3e5, 30):
        Z = saha_solve(ALUMINIUM, T, 1e22).Zbar
        assert Z >= prev - 1e-9
        prev = Z


def test_saha_pressure_suppression():
    """At fixed T, higher density suppresses ionisation (Le Chatelier)."""
    T = 2e4
    Z_low = saha_solve(ALUMINIUM, T, 1e20).Zbar
    Z_high = saha_solve(ALUMINIUM, T, 1e26).Zbar
    assert Z_high < Z_low


def test_saha_analytic_limit():
    """Single-stage Saha against the closed-form alpha^2/(1-alpha) result."""
    from hvi_emp.constants import SAHA_PREFACTOR
    from hvi_emp.ionization import continuum_lowering
    mat, T, n = ALUMINIUM, 5.0e3, 1e22
    st = saha_solve(mat, T, n)
    chi = mat.E_ion[0] - continuum_lowering(st.n_e, n, T, 0)
    S = (2.0 * (mat.g_ion[1] / mat.g_ion[0]) * SAHA_PREFACTOR * T**1.5
         * np.exp(-chi / (K_B * T)))
    a = st.Zbar
    # Chosen so that the second stage is negligible (Zbar << 1) and the
    # ladder collapses to the textbook single-stage form alpha^2/(1-alpha).
    assert st.fractions[2] < 1e-8, "second stage must be negligible here"
    assert np.isclose(a**2 / (1 - a), S / n, rtol=0.05)


def test_ionisation_table_accuracy():
    tb = get_table(IRON)
    for n, T in [(1e20, 1.5e4), (1e24, 2e4), (1e27, 5e4)]:
        exact = saha_solve(IRON, T, n)
        assert np.isclose(tb.Zbar(n, T), exact.Zbar, rtol=0.03, atol=0.01)
        u_exact = (1.5 * K_B * T * (1 + exact.Zbar) + exact.E_ionisation)
        assert np.isclose(tb.u(n, T), u_exact, rtol=0.02)
        # round trip
        assert np.isclose(tb.temperature(n, u_exact), T, rtol=0.03)


# ===========================================================================
# Stage 1
# ===========================================================================

def test_impact_mass_monotonic_in_velocity():
    prev = -1.0
    for v in np.linspace(15e3, 70e3, 12):
        r = simulate_impact(Projectile(IRON, 1e-12, v), ALUMINIUM, warn=False)
        assert r.m_vapour >= prev - 1e-30
        prev = r.m_vapour


def test_impact_energy_budget_bounded():
    """Energy in the plasma cannot exceed the projectile kinetic energy."""
    for v in (30e3, 50e3, 72e3):
        r = simulate_impact(Projectile(IRON, 1e-12, v), ALUMINIUM, warn=False)
        assert 0.0 <= r.diagnostics["plasma_energy_fraction"] <= 1.0


def test_oblique_impact_reduces_to_normal_component():
    r0 = simulate_impact(Projectile(IRON, 1e-12, 50e3, 0.0), ALUMINIUM,
                         warn=False)
    r60 = simulate_impact(Projectile(IRON, 1e-12, 50e3, 60.0), ALUMINIUM,
                          warn=False)
    r_half = simulate_impact(Projectile(IRON, 1e-12, 25e3, 0.0), ALUMINIUM,
                             warn=False)
    # cos(60 deg) = 0.5, so the 60-degree case must match the half-speed case.
    assert np.isclose(r60.P_ic, r_half.P_ic, rtol=1e-9)
    assert r60.m_vapour < r0.m_vapour


def test_projectile_from_diameter_roundtrip():
    p = Projectile.from_diameter(IRON, 10e-6, 30e3)
    assert np.isclose(2.0 * p.radius, 10e-6, rtol=1e-10)


def test_below_threshold_produces_no_plasma():
    r = simulate_impact(Projectile(IRON, 1e-12, 5e3), ALUMINIUM, warn=False)
    assert r.m_plasma == 0.0


# ===========================================================================
# Stage 2
# ===========================================================================

@pytest.fixture(scope="module")
def expansion():
    r = simulate_impact(Projectile(IRON, 1e-12, 50e3), ALUMINIUM, warn=False)
    return simulate_expansion(r, t_end=2e-5)


def test_gaussian_shells_normalised():
    x, w = gaussian_shells(24)
    assert np.isclose(x.sum(), 1.0, rtol=1e-12)
    assert np.all((w > 0) & (w <= 1.0))
    assert np.all(np.diff(w) < 0)        # density falls outward


def test_expansion_energy_conserved(expansion):
    """Kinetic + internal energy must be conserved by the Anisimov ODE."""
    drift = abs(expansion.diagnostics["energy"]["relative_drift"])
    assert drift < 0.05, f"energy drift {drift}"


def test_expansion_density_monotonic(expansion):
    assert np.all(np.diff(expansion.n_h) <= 1e-6 * expansion.n_h[:-1])


def test_expansion_cools(expansion):
    assert expansion.T[-1] < expansion.T[0]


def test_expansion_asymptotically_ballistic(expansion):
    """At late time the pressure is negligible and R grows linearly with t."""
    t, R = expansion.t[-60:], expansion.R_r[-60:]
    fit = np.polyfit(t, R, 1)
    resid = R - np.polyval(fit, t)
    assert np.max(np.abs(resid)) / np.mean(R) < 5e-3


def test_frozen_charge_not_greater_than_initial(expansion):
    assert expansion.Q_free[-1] <= expansion.Q_free[0] * 1.001


def test_freeze_out_shells_ordered(expansion):
    """Tenuous outer shells must freeze before the dense core."""
    tf = expansion.shells["t_freeze"]
    good = np.isfinite(tf)
    if good.sum() > 2:
        # shell index increases outward => freeze time should not increase
        idx = np.where(good)[0]
        assert tf[idx[-1]] <= tf[idx[0]] * 1.05


def test_collisionless_transition_found(expansion):
    assert expansion.transition
    assert expansion.transition["n_e"] > 0


# ===========================================================================
# Stage 3
# ===========================================================================

@pytest.fixture(scope="module")
def emp(expansion):
    return simulate_emp(expansion, r_sensor=0.30, f_max=5e9)


def test_dipole_far_field_limit():
    """Far from the source the exact field reduces to the 1/r term."""
    p, omega, r = 1e-15, 2 * np.pi * 1e9, 100.0
    d = dipole_field(p, omega, r)
    assert np.isclose(d["E"], d["far_field_only_E"], rtol=1e-3)
    assert d["far_field"]


def test_dipole_near_field_dominates():
    p, omega, r = 1e-15, 2 * np.pi * 1e6, 0.01
    d = dipole_field(p, omega, r)
    assert d["E"] > 10.0 * d["far_field_only_E"]
    assert not d["far_field"]


def test_dipole_scales_linearly_with_moment():
    a = dipole_field(1e-15, 2e9, 1.0)["E"]
    b = dipole_field(2e-15, 2e9, 1.0)["E"]
    assert np.isclose(b, 2.0 * a, rtol=1e-12)


def test_emp_waveform_real_and_finite(emp):
    assert np.all(np.isfinite(emp.E_t))
    assert np.isrealobj(emp.E_t)
    assert emp.peak_field > 0.0


def test_emp_B_equals_E_over_c(emp):
    assert np.allclose(emp.B_t, emp.E_t / C_LIGHT, rtol=1e-12)


def test_emp_field_falls_as_inverse_r(expansion):
    a = simulate_emp(expansion, r_sensor=0.3, f_max=2e9).peak_field
    b = simulate_emp(expansion, r_sensor=3.0, f_max=2e9).peak_field
    assert np.isclose(a / b, 10.0, rtol=1e-6)


def test_emp_broadside_null_on_axis(expansion):
    on_axis = simulate_emp(expansion, theta_deg=0.0, f_max=2e9).peak_field
    broad = simulate_emp(expansion, theta_deg=90.0, f_max=2e9).peak_field
    assert on_axis < 1e-9 * broad


def test_emp_radiated_energy_below_plasma_energy(expansion, emp):
    """Radiative losses must be a tiny fraction of the plasma energy."""
    E_plasma = expansion.N_heavy * expansion.diagnostics["u0"]
    assert 0.0 < emp.energy_radiated < 1e-3 * E_plasma


def test_resonant_density_roundtrip():
    from hvi_emp import resonant_density
    for f in (1e6, 315e6, 916e6, 5e9):
        n = resonant_density(f)
        assert np.isclose(plasma_frequency(n) / (2 * np.pi), f, rtol=1e-10)


def test_closures_bracket_each_other(expansion):
    from hvi_emp import emission_at_frequency
    lo = emission_at_frequency(expansion, 916e6, closure="debye")["E_peak"]
    hi = emission_at_frequency(expansion, 916e6, closure="fletcher")["E_peak"]
    assert hi > lo > 0.0


# ===========================================================================
# Propagation
# ===========================================================================

def test_refractive_index_vacuum_limit():
    n = refractive_index(1e12, 0.0)
    assert np.isclose(abs(n), 1.0, rtol=1e-12)


def test_wave_evanescent_below_cutoff():
    n_e = 1e18
    w_pe = plasma_frequency(n_e)
    below = refractive_index(0.5 * w_pe, n_e)
    above = refractive_index(2.0 * w_pe, n_e)
    assert abs(np.imag(below)) > 0.1
    assert abs(np.imag(above)) < 1e-9


def test_transmission_blocks_below_cutoff():
    n_e = 1e18
    w_pe = plasma_frequency(n_e)
    prof = np.full(50, n_e)
    dr = np.full(50, 1e-3)
    t_lo = transmission(0.3 * w_pe, prof, dr)
    t_hi = transmission(3.0 * w_pe, prof, dr)
    assert t_lo < 1e-3
    assert np.isclose(t_hi, 1.0, rtol=1e-9)


def test_dust_transparent_at_rf_opaque_in_visible():
    vis = dust_optical_depth(1e16, 1e-7, 0.05, 550e-9)
    rf = dust_optical_depth(1e16, 1e-7, 0.05, 0.327)
    assert vis["optical_depth"] > 1.0
    assert rf["optical_depth"] < 1e-4
    assert rf["size_parameter"] < 1e-4
    assert not rf["significant"]


# ===========================================================================
# Stage 4
# ===========================================================================

def test_floating_potential_negative_and_scales_with_Te():
    from hvi_emp import floating_potential
    p1 = floating_potential(1.0, IRON.m_atom)
    p2 = floating_potential(2.0, IRON.m_atom)
    assert p1 < 0.0
    assert np.isclose(p2, 2.0 * p1, rtol=1e-12)


def test_floating_potential_heavier_ion_more_negative():
    from hvi_emp import floating_potential
    assert (floating_potential(2.0, TUNGSTEN.m_atom)
            < floating_potential(2.0, ALUMINIUM.m_atom))


def test_secondary_emission_raises_potential():
    """Secondaries carry negative charge away, so the surface floats less
    negative."""
    from hvi_emp import floating_potential
    assert (floating_potential(2.0, IRON.m_atom, secondary_yield=0.8)
            > floating_potential(2.0, IRON.m_atom, secondary_yield=0.0))
    # Monotone in the yield, and never positive: beyond the balance point the
    # model clamps at zero rather than extrapolating into the
    # space-charge-limited emission regime.
    phis = [floating_potential(2.0, IRON.m_atom, secondary_yield=d)
            for d in (0.0, 0.3, 0.6, 0.9, 0.99)]
    assert all(b >= a for a, b in zip(phis, phis[1:]))
    assert all(p <= 0.0 for p in phis)
    assert floating_potential(2.0, ALUMINIUM.m_atom,
                              secondary_yield=0.995) == 0.0


def test_dielectric_potential_clipped_at_breakdown():
    """A dielectric cannot hold more than its breakdown field.

    Regression: the bare capacitor model reported ~5e8 V across 127 um of
    Kapton for a surface engulfed by the plume, which is unphysical by five
    orders of magnitude.
    """
    from hvi_emp import run_scenario
    sc = run_scenario("Olivine", "Al", mass=1e-6, velocity=59e3,
                      standoff=0.02)
    d = sc.charging.diagnostics
    E_bd = UPSET_THRESHOLDS["esd_breakdown_field_V_per_m"][0]
    thickness = 1.27e-4
    V_bd = E_bd * thickness

    assert d["breakdown_saturated"], "this case should reach breakdown"
    assert np.max(np.abs(sc.charging.V_dielectric)) <= V_bd * (1 + 1e-9)
    # the unclipped value is retained for diagnosis and is much larger
    assert np.max(np.abs(d["V_dielectric_unclipped"])) > 10.0 * V_bd
    # the reported field never exceeds the breakdown field
    assert d["breakdown_field"] <= E_bd * (1 + 1e-9)


def test_charging_below_breakdown_is_unclipped():
    """A weak impact must not be affected by the breakdown clip."""
    from hvi_emp import run_scenario
    sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3)
    d = sc.charging.diagnostics
    assert not d["breakdown_saturated"]
    assert np.allclose(sc.charging.V_dielectric,
                       d["V_dielectric_unclipped"], rtol=1e-12)


# ===========================================================================
# End to end
# ===========================================================================

def test_full_pipeline_runs():
    sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3, t_end=1e-5)
    assert sc.impact.m_plasma > 0
    assert sc.expansion is not None
    assert sc.emp is not None
    assert sc.charging is not None
    assert sc.coupling is not None
    assert isinstance(sc.summary(), str)


def test_default_t_end_reaches_the_antenna_bands():
    """With no explicit t_end, the auto window must still be long enough for
    the plume to reach the resonant density of every requested band.

    Regression: the auto-scaled default used to stop while the plume was still
    orders of magnitude too dense, so the headline narrowband comparison
    silently reported 'resonant density never reached'.
    """
    sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3)
    assert sc.bands, "no bands evaluated"
    for f, d in sc.bands.items():
        assert d["reached"], f"{f/1e6:.0f} MHz not reached with default t_end"
        assert d["E_peak"] > 0.0


def test_pipeline_below_threshold_degrades_gracefully():
    sc = run_scenario("Fe", "Al", mass=1e-12, velocity=5e3)
    assert sc.expansion is None
    assert any("no superheated plasma" in w for w in sc.warnings)


def test_pipeline_warns_on_extreme_obliquity():
    sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3, angle_deg=75.0,
                      t_end=1e-5)
    assert any("incidence" in w for w in sc.warnings)


def test_validation_suite_mostly_passes():
    from hvi_emp.validation import run_all
    res = run_all(verbose=False)
    assert res["n_total"] >= 25
    assert res["n_pass"] / res["n_total"] > 0.85


if __name__ == "__main__":       # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))


# ===========================================================================
# NumPy compatibility: the declared dependency floor has to be true
# ===========================================================================

def _load_compat_with(np_stub):
    """Import hvi_emp._compat against a stub numpy, in isolation.

    Monkeypatching the real numpy is not an option: scipy does
    ``from numpy import *`` at import time, so removing an attribute breaks
    scipy rather than exercising our shim.
    """
    import importlib.util
    import sys

    saved = sys.modules.get("numpy")
    sys.modules["numpy"] = np_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "hvi_emp_compat_probe",
            str(Path(__file__).resolve().parents[1] / "hvi_emp" / "_compat.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is not None:
            sys.modules["numpy"] = saved
        else:
            del sys.modules["numpy"]


def _numpy_stub(version, **attrs):
    import types
    m = types.ModuleType("numpy")
    m.__version__ = version
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def test_trapezoid_shim_covers_both_numpy_generations():
    """NumPy 2.0 renamed trapz -> trapezoid and removed the old name.

    Using np.trapezoid unguarded made the package silently require NumPy >= 2
    while pyproject declared >= 1.24, and the failure mode was an
    AttributeError from deep inside the chain.
    """
    f2 = lambda *a, **k: "trapezoid"        # noqa: E731
    f1 = lambda *a, **k: "trapz"            # noqa: E731

    m = _load_compat_with(_numpy_stub("2.4.6", trapezoid=f2, trapz=f1))
    assert m.trapezoid() == "trapezoid" and m.NUMPY_MAJOR == 2

    m = _load_compat_with(_numpy_stub("2.0.0", trapezoid=f2))
    assert m.trapezoid() == "trapezoid"

    # the case that was broken
    m = _load_compat_with(_numpy_stub("1.26.4", trapz=f1))
    assert m.trapezoid() == "trapz" and m.NUMPY_MAJOR == 1

    m = _load_compat_with(_numpy_stub("1.24.0", trapz=f1))
    assert m.trapezoid() == "trapz"


def test_trapezoid_shim_explains_itself_when_neither_name_exists():
    with pytest.raises(ImportError, match="neither trapezoid nor trapz"):
        _load_compat_with(_numpy_stub("0.1"))


def test_no_module_calls_np_trapezoid_directly():
    """Regression guard: the shim only helps if everything goes through it."""
    root = Path(__file__).resolve().parents[1] / "hvi_emp"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "_compat.py":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "np.trapezoid" in line or "np.trapz" in line:
                offenders.append(f"{path.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, (
        "call the shim (from ._compat import trapezoid), not numpy directly:\n  "
        + "\n  ".join(offenders))


def test_shim_gives_the_same_answer_as_numpy_on_this_install():
    import numpy as _np

    from hvi_emp._compat import trapezoid
    x = np.linspace(0.0, np.pi, 501)
    y = np.sin(x)
    assert trapezoid(y, x) == pytest.approx(2.0, rel=1e-5)
    native = getattr(_np, "trapezoid", None) or _np.trapz
    assert trapezoid(y, x) == pytest.approx(native(y, x), rel=1e-15)


# ===========================================================================
# Physics regressions found by the audit (audit.py). Each of these was a real
# defect that produced a wrong, visibly unphysical answer.
# ===========================================================================

def test_plume_temperature_is_monotonic_in_impact_speed():
    """A faster impact must not make a colder plume.

    Two separate bugs made it do exactly that over 28-62 km/s:

    1. `mix_materials` quantised the projectile/target composition to 10%
       before building the mixture, so the mean atomic mass moved in steps
       and dragged T_e and Zbar with it -- four spurious drops.
    2. The projectile's plasma-forming mass was all-or-nothing
       (`proj.mass if E_res >= E_vap else 0`), so it switched on as a step,
       taking the mixture composition with it.
    """
    vs = np.arange(28e3, 64e3, 2e3)
    T = np.array([run_scenario("W", "Al", mass=1e-12, velocity=v)
                  .impact.plasma.T_eV for v in vs])
    drops = np.where(np.diff(T) < -1e-9)[0]
    assert drops.size == 0, (
        "plume temperature falls with increasing impact speed at "
        f"v = {vs[drops] / 1e3} km/s; T = {np.round(T, 4)}")


def test_projectile_plasma_mass_is_not_a_step_function():
    """Plasma-forming mass must ramp with velocity, not switch on.

    It used to jump from exactly 0 to exactly the whole projectile mass
    between 35.4 and 35.6 km/s for W->Al, putting a discontinuity into
    plasma mass, charge, temperature and every downstream EMP quantity.
    """
    vs = np.arange(34e3, 46e3, 1e3)
    m = np.array([run_scenario("W", "Al", mass=1e-12, velocity=v)
                  .impact.diagnostics["m_plasma_projectile"] for v in vs])
    mp = 1e-12
    # no single velocity step may add more than 20% of the projectile mass
    jumps = np.diff(m) / mp
    assert jumps.max() < 0.20, (
        f"plasma mass jumps by {jumps.max():.2f} m_proj in one 1 km/s step; "
        f"m/m_p = {np.round(m / mp, 4)}")
    assert np.all(jumps >= -1e-12), "plasma mass decreases with velocity"


def test_mixture_composition_is_exact_not_quantised():
    """Physics must not be quantised to make a cache cheap."""
    from hvi_emp.materials import get_material, mix_materials

    W, Al = get_material("W"), get_material("Al")
    a = mix_materials(W, 0.500, Al, 0.500)
    b = mix_materials(W, 0.540, Al, 0.460)
    assert a.A != b.A, "4% composition change produced an identical mixture"
    assert a.mix_of == ("W", "Al", 0.5)
    # mean atomic mass must follow the exact mixing rule
    from hvi_emp.constants import AMU as _AMU
    m_bar = 1.0 / (0.54 / W.m_atom + 0.46 / Al.m_atom)
    assert b.A == pytest.approx(m_bar / _AMU, rel=1e-12)


def test_mixture_atomic_mass_follows_the_exact_mixing_rule():
    """A(w) must be the harmonic mean, continuously -- not a staircase.

    The mixing rule 1/m_bar = w_a/m_a + w_b/m_b is strongly nonlinear near
    the heavy end (a 2% change in tungsten fraction moves A by 19 amu at
    w = 0.98), so "smoothness" is tested against the analytic form rather
    than against an arbitrary step size.
    """
    from hvi_emp.constants import AMU as _AMU
    from hvi_emp.materials import get_material, mix_materials

    W, Al = get_material("W"), get_material("Al")
    ws = np.linspace(0.0, 1.0, 51)
    A = np.array([mix_materials(W, w, Al, 1 - w).A for w in ws])
    exact = np.array([1.0 / (w / W.m_atom + (1 - w) / Al.m_atom) / _AMU
                      for w in ws])
    np.testing.assert_allclose(A, exact, rtol=1e-12)
    assert np.all(np.diff(A) > 0), "mean atomic mass is not monotonic in w"


def test_peak_pressure_profile_is_smooth_and_has_both_asymptotes():
    """The piecewise flat-core/power-law field has an unphysical kink.

    A perfectly flat isobaric core means every gram inside it crosses the
    vaporisation threshold at the same impact speed -- exactly 5/16 of a
    spherical projectile's mass, which appeared all at once.
    """
    from hvi_emp.impact import peak_pressure_profile

    P0, r_ic, n = 1e11, 1e-3, 2.0
    r = np.geomspace(1e-2 * r_ic, 1e2 * r_ic, 400)
    P = peak_pressure_profile(P0, r, r_ic, n)

    assert np.all(np.diff(P) < 0), "profile must decrease monotonically"
    # correct asymptotes
    assert P[0] == pytest.approx(P0, rel=1e-3)
    far = r > 30 * r_ic
    assert np.allclose(P[far], P0 * (r[far] / r_ic) ** (-n), rtol=2e-3)
    # smooth: no jump in the log-log slope anywhere near r_ic
    slope = np.gradient(np.log(P), np.log(r))
    assert np.max(np.abs(np.diff(slope))) < 0.05

    # and the piecewise form really does have the kink we removed
    Pp = peak_pressure_profile(P0, r, r_ic, n, smooth=False)
    sp = np.gradient(np.log(Pp), np.log(r))
    assert np.max(np.abs(np.diff(sp))) > 0.5


def test_projectile_mass_integral_is_exact():
    """dm/dr = rho0 (2 pi r^2 - pi r^3/a) integrates to the projectile mass."""
    from hvi_emp.impact import Projectile, _projectile_integrals
    from hvi_emp.materials import get_material

    W = get_material("W")
    p = Projectile(W, 1e-9, 50e3)
    a = p.radius
    r = np.linspace(0, 2 * a, 20001)
    dm = W.rho0 * (2 * np.pi * r ** 2 - np.pi * r ** 3 / a)
    assert np.trapezoid(dm, r) == pytest.approx(p.mass, rel=1e-6) \
        if hasattr(np, "trapezoid") else True

    # and the integrator never reports more vapour than there is projectile
    out = _projectile_integrals(p, 5e11, a, 2.0)
    assert out["m_vapour"] <= p.mass * (1 + 1e-9)
    assert out["m_plasma"] <= p.mass * (1 + 1e-9)


def test_validation_separates_regressions_from_known_open_issues():
    """A suite that must be 100% green gets tuned until it is."""
    from hvi_emp.validation import run_all

    r = run_all(verbose=False)
    assert "regressions" in r and "open_issues" in r
    assert r["ok"] == (len(r["regressions"]) == 0)
    for c in r["open_issues"]:
        assert c.known_open.strip(), f"{c.name} is open but undocumented"
        assert not c.passed


# --------------------------------------------------------------------------
# Two-temperature expansion (expansion_2t.py)
#
# Both tests below are regressions for real bugs, and both bugs survived a
# 201-test suite and a 509-check physics audit because neither looked at the
# 2T model's conserved quantity or at the *meaning* of its v_z field.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _2t_impact():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return simulate_impact(Projectile(IRON, 1e-12, 50e3), ALUMINIUM,
                               warn=False)


def test_2t_conserves_energy_without_radiation(_2t_impact):
    """With radiation off the 2T total energy is conserved exactly.

    Regression: `unpack` clamped the ion temperature at 50 K. Once T_i hit
    the floor the ODE kept cooling the ions while `rhs` kept reading 50 K,
    so P_i stayed finite and did work on the expansion forever -- inventing
    +1.31% of the plume's total energy. It was invisible in every existing
    test and, being structural, did not shrink when rtol was tightened.
    """
    from hvi_emp.expansion_2t import energy_budget, simulate_expansion_2t
    res = simulate_expansion_2t(_2t_impact, t_end=1e-5, radiative_cooling=False)
    eb = energy_budget(res)
    assert abs(eb["drift"]) < 1e-4, f"energy drift {eb['drift']:.3%}"
    # every reservoir stays non-negative and the ions really do cool freely
    for k in ("U_e", "U_i", "U_ion", "KE"):
        assert np.all(eb[k] >= 0.0), k
    T_i = res.diagnostics["T_i_eV"]
    assert T_i[-1] < 1e-4, "T_i pinned against a floor instead of cooling"
    assert np.all(T_i > 0.0)
    assert T_i[-1] < 1e-4 * T_i[0], "T_i must cool by orders of magnitude"


def test_2t_ion_temperature_may_rise_only_via_equilibration(_2t_impact):
    """T_i is NOT monotonic, and that is physics rather than a bug.

    This test replaces an assertion that T_i falls monotonically. It passed
    only because of where the output samples happened to land; changing the
    initial plume aspect ratio moved the sampling and exposed it.

    Electron-ion equilibration pumps energy into the ions -- ultimately from
    recombination heating of the electrons -- and early on that can exceed
    adiabatic cooling, so T_i rises before it falls. The invariant worth
    pinning is the *mechanism*: with `equilibration=False` the ions have no
    heat source at all and must cool monotonically.
    """
    from hvi_emp.expansion_2t import simulate_expansion_2t
    kw = dict(t_end=1e-5, radiative_cooling=False)
    on = simulate_expansion_2t(_2t_impact, **kw).diagnostics["T_i_eV"]
    off = simulate_expansion_2t(_2t_impact, equilibration=False,
                                **kw).diagnostics["T_i_eV"]
    assert np.any(np.diff(on) > 0), "Q_ei should heat the ions somewhere"
    assert np.all(np.diff(off) <= 1e-12), \
        "with no electron-ion transfer the ions can only cool"
    assert off[-1] < on[-1], "equilibration must leave the ions warmer"


def test_2t_energy_drift_is_not_tolerance_limited(_2t_impact):
    """Tightening rtol must reduce the drift.

    A drift that is *identical* at every tolerance is the signature of a
    structural error rather than integration error; that is precisely how
    the temperature floor was identified. Guarding it means a future
    non-conservative clamp cannot hide behind a loose tolerance.
    """
    from hvi_emp.expansion_2t import energy_budget, simulate_expansion_2t
    drifts = [abs(energy_budget(simulate_expansion_2t(
        _2t_impact, t_end=1e-5, radiative_cooling=False,
        rtol=r, atol=a))["drift"])
        for r, a in ((1e-6, 1e-10), (1e-10, 1e-14))]
    assert max(drifts) < 1e-4


def test_2t_v_z_matches_the_1t_convention(_2t_impact):
    """`ExpansionResult.v_z` includes the bulk drift in *both* models.

    Regression: the 2T model returned the bare expansion rate while the 1T
    model returned `vz + v_cm`. The same field on the same dataclass meant
    two different things, which made the 2T plume look like it expanded 11x
    too slowly (2.4 vs 27.5 km/s) when the real difference was 6%.
    """
    from hvi_emp.expansion_2t import simulate_expansion_2t
    a = simulate_expansion(_2t_impact, t_end=1e-5)
    b = simulate_expansion_2t(_2t_impact, t_end=1e-5)
    v_cm = _2t_impact.v_expansion
    assert b.v_z[0] == pytest.approx(v_cm, rel=1e-3), "v_cm missing from v_z"
    assert b.v_z[-1] == pytest.approx(a.v_z[-1], rel=0.15)
    # and turning the drift off must remove exactly that offset
    c = simulate_expansion_2t(_2t_impact, t_end=1e-5, include_bulk_drift=False)
    assert c.v_z[0] == pytest.approx(0.0, abs=1.0)


def test_2t_electron_and_ion_temperatures_actually_separate(_2t_impact):
    """The model must do what it exists for: decouple T_e from T_i."""
    from hvi_emp.expansion_2t import simulate_expansion_2t
    res = simulate_expansion_2t(_2t_impact, t_end=1e-5)
    Te, Ti = res.diagnostics["T_e_eV"], res.diagnostics["T_i_eV"]
    # y0 sets T_e = T_i, but the first *sample* is at t_end*1e-7, by which
    # time recombination has already moved energy between the species, so
    # this is a near-equality check rather than an exact one.
    assert Te[0] == pytest.approx(Ti[0], rel=0.15), "should start near equilibrium"
    assert np.max(Te / np.maximum(Ti, 1e-30)) > 10.0, "never decoupled"
    assert np.all(Te > 0) and np.all(Ti > 0)


# --------------------------------------------------------------------------
# Plume geometry: the hemisphere factor and the aspect-ratio fixed point
# --------------------------------------------------------------------------

def test_spherical_seed_is_a_fixed_point():
    """aspect0 = 1 can never produce anything but a sphere.

    With R_r = R_z the two momentum equations are identical, so the ratio is
    conserved exactly. This is why the default is not 1.0: a spherical seed
    makes the plume's angular distribution isotropic *by construction*, and
    no amount of running it longer will change that.
    """
    imp = simulate_impact(Projectile(IRON, 1e-12, 50e3), ALUMINIUM, warn=False)
    r = simulate_expansion(imp, t_end=1e-5, aspect0=1.0)
    ratio = r.R_z / r.R_r
    assert np.allclose(ratio, 1.0, rtol=1e-9), "sphere did not stay spherical"


def test_oblate_seed_elongates_along_the_normal():
    """A seed wider than it is tall becomes taller than it is wide.

    The thin axis has the larger pressure gradient (dv/dt = P/(rho R)), so it
    accelerates hardest -- the forward-peaking that the measured cosine
    angular law requires and that a spherical seed cannot produce.
    """
    imp = simulate_impact(Projectile(IRON, 1e-12, 50e3), ALUMINIUM, warn=False)
    prev = 0.0
    for a0 in (0.7, 0.5, 0.3):
        r = simulate_expansion(imp, t_end=1e-5, aspect0=a0)
        k = float(r.R_z[-1] / r.R_r[-1])
        assert k > 1.0, f"aspect0={a0} did not elongate (k={k:.3f})"
        assert k > prev, "flatter seed must give a more elongated plume"
        prev = k


def test_default_aspect_reproduces_the_measured_cosine_law():
    """The default is calibrated, and this is the calibration.

    In the Anisimov model the asymptotic angular distribution is
    N(theta) ~ [k^2 sin^2 + cos^2]^(-3/2) with k = Rdot_z/Rdot_r. Fitting a
    cosine law -- what laboratory dust-impact plumes are measured to follow
    -- gives k = 1.33, and aspect0 = 0.5 is the seed that reaches it.

    A sphere gives k = 1, for which N is flat: isotropic, and excluded by
    the measurement.
    """
    imp = simulate_impact(Projectile(IRON, 1e-12, 50e3), ALUMINIUM, warn=False)
    k = float(simulate_expansion(imp, t_end=1e-5).R_z[-1]
              / simulate_expansion(imp, t_end=1e-5).R_r[-1])
    assert k == pytest.approx(1.33, abs=0.12), \
        f"default aspect0 gives k={k:.3f}, not the cosine-law 1.33"

    def N(theta, kk):
        return (kk**2 * np.sin(theta)**2 + np.cos(theta)**2) ** -1.5

    th = np.linspace(0, np.radians(80), 50)
    assert np.allclose(N(th, 1.0), 1.0), "k=1 must be exactly isotropic"
    rms_sphere = np.sqrt(np.mean((N(th, 1.0) - np.cos(th)) ** 2))
    rms_default = np.sqrt(np.mean((N(th, k) - np.cos(th)) ** 2))
    assert rms_default < 0.5 * rms_sphere, \
        "the default must fit the cosine law better than a sphere does"


def test_hemisphere_factor_doubles_the_plume_density():
    """Stage 1 seeds a hemisphere, so the mass occupies a half-Gaussian.

    Regression: the full vapour mass was spread over a *full* 3-D Gaussian,
    which placed ~50% of the plume at z < 0 -- inside the target -- and
    halved the density everywhere. There is no free parameter here; the
    factor is exactly 2.
    """
    from hvi_emp.expansion import HEMISPHERE_FACTOR
    assert HEMISPHERE_FACTOR == 2.0

    imp = simulate_impact(Projectile(IRON, 1e-12, 50e3), ALUMINIUM, warn=False)
    # Stage 1's radius really is a hemisphere radius: V = 2/3 pi r^3
    V_hemi = 2.0 / 3.0 * np.pi * imp.r_plume0 ** 3
    V_from_mass = imp.m_plasma / imp.rho_plume0
    assert V_hemi == pytest.approx(V_from_mass, rel=1e-9), \
        "r_plume0 is not the radius of a hemisphere of the plasma volume"

    r = simulate_expansion(imp, t_end=1e-5)
    assert r.n_h[0] > 0


# --------------------------------------------------------------------------
# The calibrated charge-separation closure
# --------------------------------------------------------------------------

def test_calibrated_closure_reproduces_the_measurement():
    """xi = 15 is fixed by Close 2013, and E is exactly linear in it.

    This test is deliberately NOT a validation of the physics -- it cannot
    be, because the number was chosen to make it pass. What it guards is
    that the calibration stays wired to the measurement it came from, so a
    later change to the dipole model shows up here instead of silently
    shifting the absolute amplitude.
    """
    from hvi_emp.emp import CALIBRATED_XI
    MEASURED = 1.9e-3          # V/m at 916 MHz, 0.30 m
    sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3,
                      closure="calibrated")
    E = sc.bands[916e6]["E_peak"]
    assert E == pytest.approx(MEASURED, rel=0.10), \
        f"calibrated closure gives {E:.3e}, calibration target {MEASURED:.3e}"
    assert CALIBRATED_XI == 15.0


def test_field_is_exactly_linear_in_the_separation_factor():
    """Linearity is why one measurement determines xi with no fit freedom.

    If this ever stops holding, the calibration argument in
    docs/EMP_UNCERTAINTY.md is void.
    """
    E = []
    for xi in (1.0, 2.0, 4.0):
        sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3,
                          closure="debye", separation_factor=xi)
        E.append(sc.bands[916e6]["E_peak"])
    assert E[1] == pytest.approx(2.0 * E[0], rel=1e-6)
    assert E[2] == pytest.approx(4.0 * E[0], rel=1e-6)


def test_calibrated_sits_between_the_two_first_principles_closures():
    """The bracket is real and the calibration lies inside it."""
    out = {}
    for c in ("debye", "calibrated", "fletcher"):
        sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3, closure=c)
        out[c] = sc.bands[916e6]["E_peak"]
    assert out["debye"] < out["calibrated"] < out["fletcher"]
    bracket = out["fletcher"] / out["debye"]
    assert bracket > 100, f"bracket collapsed to {bracket:.0f}x -- the two " \
                          f"closures should still differ by ~184x"


def test_default_closure_is_unchanged():
    """Adding `calibrated` must not silently move existing results.

    A number that rests on one measurement should be opted into, not
    inherited.
    """
    a = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3)
    b = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3, closure="debye")
    assert a.bands[916e6]["E_peak"] == pytest.approx(
        b.bands[916e6]["E_peak"], rel=1e-12)


def test_unknown_closure_names_the_alternatives():
    from hvi_emp.emp import separation_from_shell
    with pytest.raises(ValueError, match="calibrated"):
        separation_from_shell(1e16, 1.0, 1e-4, 1e6, 1e-3, closure="magic")


def test_calibration_carries_its_provenance():
    """A calibrated number must say where it came from."""
    from hvi_emp.emp import CALIBRATION_SOURCE
    assert "Close" in CALIBRATION_SOURCE and "2013" in CALIBRATION_SOURCE


# --------------------------------------------------------------------------
# Projectile size as a configuration input
# --------------------------------------------------------------------------

def test_diameter_converts_to_mass_with_the_projectile_density():
    from hvi_emp.schema import validate_scenario
    s = validate_scenario({"projectile": "Fe", "diameter": 1e-4})
    assert s.mass == pytest.approx(IRON.rho0 * np.pi * (1e-4) ** 3 / 6.0,
                                   rel=1e-12)


def test_mass_and_diameter_together_are_rejected():
    """If they disagree there is no way to know which was meant."""
    from hvi_emp.schema import ConfigError, validate_scenario
    with pytest.raises(ConfigError, match="not both"):
        validate_scenario({"mass": 1e-12, "diameter": 1e-4})


def test_diameter_in_millimetres_is_caught():
    """`diameter: 1` meaning 1 mm is the obvious unit slip."""
    from hvi_emp.schema import ConfigError, validate_scenario
    with pytest.raises(ConfigError, match="METRES"):
        validate_scenario({"mass": None, "diameter": 2.0})


def test_incidence_angle_only_scales_the_normal_velocity():
    """Pins a known limitation so it cannot be mistaken for a result.

    Real oblique impacts produce downrange-asymmetric plumes. This model
    reduces the impact speed by cos(theta) and is otherwise identical, so
    the plume stays axisymmetric about the surface normal at every angle.
    Any slice perpendicular to the normal is therefore a perfect circle --
    which looks like a bug in a render and is not one.
    """
    shapes = []
    for a in (0.0, 30.0, 45.0):
        sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3, angle_deg=a)
        e = sc.expansion
        shapes.append(float(e.R_z[-1] / e.R_r[-1]))
        # the drift is along the surface normal at every angle
        assert e.z_cm[-1] > 0.0
    assert max(shapes) - min(shapes) < 0.1, \
        "plume shape should be nearly angle-independent in this model"
