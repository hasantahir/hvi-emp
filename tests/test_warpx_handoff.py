"""WarpX initialised from the hydrocode handoff.

The generated PICMI script needs pywarpx and is not executed here. What is
tested is everything that decides what it will simulate: which particles,
where, how fast, how heavy, and on what grid.
"""

import json
import py_compile

import numpy as np
import pytest

from hvi_emp.constants import AMU, E_CHARGE
from hvi_emp.handoff import HandoffCriteria, extract, find_handoff
from hvi_emp.solvers.m2c_output import read_m2c_series, read_m2c_vtr
from hvi_emp.solvers.m2c_synthetic import SyntheticPlume, synthetic_plume_run
from hvi_emp.solvers.warpx_handoff import (WarpXHandoffConfig,
                                           advance_ballistic, cloud_stats,
                                           make_particles, n_resonant, plan,
                                           sample_cloud, time_to_peak_density,
                                           write_warpx_from_handoff)

CFG = WarpXHandoffConfig(ion_mass_amu=26.98)


@pytest.fixture(scope="module")
def handoff(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("w") / "results")
    synthetic_plume_run(d, plume=SyntheticPlume(M=1e-6))
    series = read_m2c_series(d)
    crit = HandoffCriteria()
    idx, rows = find_handoff(series, crit)
    t, path = series[idx]
    return extract(read_m2c_vtr(path, time=t), crit, rows[idx],
                   allow_synthetic=True)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def test_charge_handed_to_pic_is_the_hydrocode_charge(handoff):
    cloud = sample_cloud(handoff, CFG)
    assert cloud["w_e"].sum() * E_CHARGE == \
        pytest.approx(handoff.summary["total_charge_C"], rel=1e-12)


def test_particles_stay_inside_their_source_cells(handoff):
    cloud = sample_cloud(handoff, CFG)
    f = handoff.fields
    mask = f["plume_mask"].astype(bool)
    r = np.hypot(cloud["x"], cloud["y"])
    assert r.min() >= 0.0
    assert cloud["z"].min() >= f["x_edges"][np.nonzero(mask)[0].min()] - 1e-15
    assert cloud["z"].max() <= f["x_edges"][np.nonzero(mask)[0].max() + 1]
    assert r.max() <= f["y_edges"][np.nonzero(mask)[1].max() + 1] + 1e-15


def test_azimuth_is_uniform_so_rz_is_honoured(handoff):
    cloud = sample_cloud(handoff, CFG)
    th = np.arctan2(cloud["y"], cloud["x"])
    hist, _ = np.histogram(th, bins=8)
    assert hist.min() > 0.85 * hist.mean()


def test_pairs_start_exactly_neutral(handoff):
    """Co-located electrons and ions: no sampling-noise field at t = 0."""
    cloud = sample_cloud(handoff, CFG)
    p = make_particles(cloud, CFG)
    e, i = p["electrons"], p["ions"]
    np.testing.assert_array_equal(e["x"], i["x"])
    np.testing.assert_array_equal(e["z"], i["z"])
    Zi = cloud["meta"]["ion_charge_state"]
    np.testing.assert_allclose(e["w"], Zi * i["w"], rtol=1e-12)


def test_bulk_velocity_is_the_hydrocode_velocity_not_an_imposed_drift(
        handoff):
    """The old bridge gave electrons sqrt(mi/me) x the ion drift by hand.

    Here both species carry the cell's single fluid velocity; any
    separation must come out of the PIC. Ions (small thermal spread) pin
    the bulk tightly; electrons agree within their sampling error.
    """
    cloud = sample_cloud(handoff, CFG)
    p = make_particles(cloud, CFG)
    w = cloud["w_e"]
    bulk = (w * cloud["vz"]).sum() / w.sum()
    # weights span orders of magnitude: the effective sample size is
    # (sum w)^2 / sum w^2, far below the particle count
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    kT = 1.380649e-23 * cloud["T"].mean()
    for name, m in (("ions", cloud["meta"]["ion_mass_kg"]),
                    ("electrons", 9.1093837e-31)):
        sp = p[name]
        mean = (sp["w"] * sp["uz"]).sum() / sp["w"].sum()
        se = np.sqrt(kT / m) / np.sqrt(n_eff)
        assert abs(mean - bulk) < 4 * se, (name, mean, bulk, se)


def test_particle_budget_caps_ppc(handoff):
    small = WarpXHandoffConfig(ion_mass_amu=26.98, ppc=8, max_particles=4000)
    cloud = sample_cloud(handoff, small)
    assert cloud["meta"]["ppc_used"] == 1
    assert cloud["w_e"].sum() * E_CHARGE == \
        pytest.approx(handoff.summary["total_charge_C"], rel=1e-12)


# --------------------------------------------------------------------------
# ballistic advance to the band
# --------------------------------------------------------------------------

def test_resonant_density_matches_fletcher_and_close():
    """F&C (2017): measured RF corresponds to a peak density of 1e16 m^-3."""
    assert n_resonant(916e6) == pytest.approx(1.04e16, rel=0.01)
    assert n_resonant(315e6) == pytest.approx(1.23e15, rel=0.01)


def test_advance_conserves_charge_and_moves_on_straight_lines(handoff):
    cloud = sample_cloud(handoff, CFG)
    later = advance_ballistic(cloud, 1e-6)
    np.testing.assert_allclose(later["z"] - cloud["z"], cloud["vz"] * 1e-6)
    assert later["w_e"].sum() == cloud["w_e"].sum()
    assert later["t"] == pytest.approx(cloud["t"] + 1e-6)


def test_time_to_band_hits_the_target_density(handoff):
    cloud = sample_cloud(handoff, CFG)
    n0 = cloud["meta"]["n_e_peak_handoff"]
    target = n_resonant(916e6)
    t = time_to_peak_density(cloud, n0, target)
    st = cloud_stats(advance_ballistic(cloud, t))
    # binned peak is smoothed by the bins, so allow a factor of two
    assert 0.5 * target < st["n_peak_binned"] < 2.0 * target


def test_no_advance_when_already_tenuous(handoff):
    cloud = sample_cloud(handoff, CFG)
    assert time_to_peak_density(cloud, 1e15, 1e16) == 0.0


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

def test_at_the_handoff_the_window_is_unaffordable_and_says_so(handoff):
    cloud = sample_cloud(handoff, CFG)
    pl = plan(cloud_stats(cloud), CFG)
    assert pl["scheme"] == "implicit"
    assert pl["cell_steps_explicit"] > 1e20
    assert any("too dense for an EM PIC window" in n for n in pl["notes"])


#: Fletcher & Close's band epoch: peak 1e16 m^-3 at a ~40 mm scale.
FC_EPOCH = dict(t=0.0, centroid_z=-0.04, rms_z=0.02, rms_r=0.03,
                z_min=-0.08, z_max=0.0, r_max=0.1, n_peak_binned=1.0e16,
                n_edge_binned=5.0e14, v_p90=1e4, T_mean_K=3.0e4, N_e=1.0)


def test_fletcher_close_epoch_is_an_overnight_explicit_run():
    """Resolving lambda_D at their band epoch costs ~2e12 cell-steps.

    That is above the default budget (about an hour or two on one GPU), so
    auto chooses implicit -- and the plan carries the explicit figure so
    the user can decide to spend the night instead.
    """
    pl = plan(FC_EPOCH, CFG)
    assert 1e12 < pl["cell_steps_explicit"] < 1e13
    assert pl["scheme"] == "implicit"


def test_auto_picks_explicit_when_debye_resolution_fits_the_budget():
    big = WarpXHandoffConfig(ion_mass_amu=26.98, cell_step_budget=1e13)
    pl = plan(FC_EPOCH, big)
    assert pl["scheme"] == "explicit"
    assert pl["dx"] <= pl["lambda_D_peak"] * (1 + 1e-12)
    assert pl["omega_pe_peak"] * pl["dt"] <= 0.2 + 1e-12


def test_run_covers_three_cycles_of_the_lowest_band(handoff):
    cloud = sample_cloud(handoff, CFG)
    pl = plan(cloud_stats(cloud), CFG)
    assert pl["t_max"] >= 3.0 / 315e6


def test_implicit_resolves_the_edge_plasma_frequency(handoff):
    cloud = sample_cloud(handoff, CFG)
    pl = plan(cloud_stats(cloud), WarpXHandoffConfig(ion_mass_amu=26.98,
                                                     scheme="implicit"))
    assert pl["omega_pe_edge"] * pl["dt"] <= 0.2 + 1e-12


# --------------------------------------------------------------------------
# the run directory
# --------------------------------------------------------------------------

def test_run_directory_is_complete_and_consistent(handoff, tmp_path):
    out = write_warpx_from_handoff(handoff, str(tmp_path / "pic"), CFG)
    py_compile.compile(out["script"], doraise=True)
    rec = json.load(open(tmp_path / "pic" / "plan.json"))
    assert rec["advance"]["mode"] == "band"
    assert rec["charge_in_particles_C"] == \
        pytest.approx(handoff.summary["total_charge_C"], rel=1e-12)
    assert "Fletcher & Close (2017)" in rec["advance"]["note"]
    assert rec["particles"]["max_velocity_underestimate"] >= 1.0
    p = np.load(tmp_path / "pic" / "particles.npz")
    for sp in ("electrons", "ions"):
        for k in ("x", "y", "z", "ux", "uy", "uz", "w"):
            assert p[f"{sp}_{k}"].size == p["electrons_w"].size
    text = open(out["script"]).read()
    assert "CylindricalGrid" in text and "add_particles" in text
    assert f"{26.98 * AMU:.9e}" in text


def test_non_axisymmetric_handoff_is_refused_not_mangled(handoff):
    bad = type(handoff)(summary=handoff.summary, fields=handoff.fields,
                        criteria=handoff.criteria,
                        provenance=dict(handoff.provenance,
                                        geometry="cartesian"))
    with pytest.raises(NotImplementedError, match="3-D"):
        sample_cloud(bad, CFG)
