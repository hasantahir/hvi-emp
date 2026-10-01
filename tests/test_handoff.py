"""The hydrocode -> PIC handoff (Fletcher 2021's waterfall).

The reader is tested against PETSc's exact VTR layout -- multi-rank pieces,
UInt64 byte prefixes, M2C's duplicated array -- and the extractor against
plumes whose integrals are known by construction.
"""

import json
import os

import numpy as np
import pytest

from hvi_emp.constants import E_CHARGE, EV_PER_K
from hvi_emp.handoff import (HandoffCriteria, assess, coulomb_log, extract,
                             find_handoff, load_handoff, nu_ei,
                             plasma_frequency, plume_mask)
from hvi_emp.solvers.m2c_output import (cell_edges, read_m2c_series,
                                        read_m2c_vtr)
from hvi_emp.solvers.m2c_synthetic import (SyntheticPlume,
                                           synthetic_plume_run,
                                           write_m2c_pvd, write_petsc_vtr)

CRIT = HandoffCriteria()


@pytest.fixture(scope="module")
def iso(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("iso") / "results")
    run = synthetic_plume_run(d, plume=SyntheticPlume(M=1e-6))
    return d, run


@pytest.fixture(scope="module")
def adiabatic(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("ad") / "results")
    synthetic_plume_run(d, plume=SyntheticPlume(T_law="adiabatic", M=1e-6))
    return d


# --------------------------------------------------------------------------
# reader: PETSc's byte layout
# --------------------------------------------------------------------------

def test_multi_rank_pieces_are_stitched_exactly(tmp_path):
    """A 64-rank M2C run writes 64 pieces; values must land in place."""
    x, y, z = np.arange(7.0), np.arange(5.0), np.arange(3.0)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    f = 100 * X + 10 * Y + Z                       # every cell distinct
    v = np.stack([X, -Y, 2 * Z], axis=-1)
    p = write_petsc_vtr(str(tmp_path / "a.vtr"), x, y, z,
                        {"density": f, "velocity": v}, ranks=(3, 2, 2))
    s = read_m2c_vtr(p, si=False, geometry="cartesian")
    assert s.n_pieces == 12
    np.testing.assert_array_equal(s["density"], f)
    np.testing.assert_array_equal(s["velocity"], v)
    np.testing.assert_array_equal(s.x, x)


def test_duplicate_electron_density_is_tolerated(tmp_path):
    x, y, z = np.arange(3.0), np.arange(2.0) + 0.5, np.zeros(1)
    a = np.ones((3, 2, 1))
    p = write_petsc_vtr(str(tmp_path / "d.vtr"), x, y, z,
                        {"electron_density": a})
    s = read_m2c_vtr(p, si=False)
    assert any("duplicate" in n for n in s.notes)
    np.testing.assert_array_equal(s["electron_density"], a)


def test_units_come_back_si(tmp_path):
    """g/mm^3 -> kg/m^3, 1/mm^3 -> 1/m^3, mm -> m: the deck's own table."""
    x, y, z = np.array([1.0, 2.0]), np.array([0.5]), np.zeros(1)
    p = write_petsc_vtr(str(tmp_path / "u.vtr"), x, y, z, {
        "density": np.full((2, 1, 1), 2.7e-3),
        "electron_density": np.full((2, 1, 1), 1.0e13),
        "velocity": np.full((2, 1, 1, 3), 1.0e6)})
    s = read_m2c_vtr(p)
    assert s["density"][0, 0, 0] == pytest.approx(2700.0)
    assert s["electron_density"][0, 0, 0] == pytest.approx(1.0e22)
    assert s["velocity"][0, 0, 0, 0] == pytest.approx(1.0e3)
    assert s.x[1] == pytest.approx(2.0e-3)


def test_truncated_file_fails_loudly(tmp_path):
    x, y, z = np.arange(4.0), np.arange(3.0) + 0.5, np.zeros(1)
    p = write_petsc_vtr(str(tmp_path / "t.vtr"), x, y, z,
                        {"density": np.ones((4, 3, 1))}, ranks=(2, 1, 1))
    raw = open(p, "rb").read()
    open(p, "wb").write(raw[: len(raw) - 60])
    with pytest.raises(Exception):
        read_m2c_vtr(p)


def test_unclosed_pvd_from_an_interrupted_run_is_recovered(tmp_path):
    write_m2c_pvd(str(tmp_path), [(1e-7, "a.vtr"), (2e-7, "b.vtr")],
                  closed=False)
    rows = read_m2c_series(str(tmp_path))
    assert [t for t, _ in rows] == [1e-7, 2e-7]


def test_cylindrical_volumes_integrate_the_full_disc():
    """sum of cell volumes = pi R^2 L, with the first edge on the axis."""
    from hvi_emp.solvers.m2c_output import M2CSnapshot
    n, R, L = 40, 3e-3, 5e-3
    r = (np.arange(n) + 0.5) * R / n
    x = (np.arange(30) + 0.5) * L / 30
    s = M2CSnapshot(path="v", time=0, x=x, y=r, z=np.zeros(1), fields={},
                    geometry="cylindrical")
    assert s.cell_volumes().sum() == pytest.approx(np.pi * R * R * L,
                                                   rel=1e-12)
    assert cell_edges(r, lower=0.0)[0] == 0.0


# --------------------------------------------------------------------------
# the synthetic run is exactly what it claims
# --------------------------------------------------------------------------

def test_integrated_charge_matches_construction(iso):
    d, run = iso
    for (t, path), ex in zip(read_m2c_series(d), run["exact"]):
        s = read_m2c_vtr(path, time=t)
        N = (s["electron_density"] * s.cell_volumes()).sum()
        assert N == pytest.approx(ex["N_e"], rel=1e-9)


# --------------------------------------------------------------------------
# plasma frequencies against hand values
# --------------------------------------------------------------------------

def test_plasma_frequency_hand_value():
    # omega_pe = 56.4 sqrt(n_e[m^-3]) rad/s
    assert plasma_frequency(1e18) == pytest.approx(5.64e10, rel=2e-3)


def test_collision_frequency_hand_value():
    """NRL: nu_e = 2.91e-6 n_e[cc] lnL T^-3/2."""
    n, T = 1e24, 10.0                               # m^-3, eV
    lnL = coulomb_log(n, T, 1.0)
    assert nu_ei(n, T, 1.0) == pytest.approx(2.91e-6 * 1e18 * lnL
                                             * T ** -1.5)


def test_partially_ionised_plasma_uses_singly_charged_ions():
    """Z-bar < 1 means mostly Z = 1 ions plus neutrals: factor 1, not Z."""
    assert nu_ei(1e24, 5.0, 0.1) == pytest.approx(nu_ei(1e24, 5.0, 1.0))


# --------------------------------------------------------------------------
# finding the handoff
# --------------------------------------------------------------------------

def test_isothermal_plume_crosses_over_between_frames(iso):
    """Fletcher's Fig. 2 behaviour: nu_ei falls faster than omega_pe."""
    d, _run = iso
    idx, rows = find_handoff(read_m2c_series(d), CRIT)
    assert idx == 1
    assert rows[0]["collisionless_fraction"] < 0.5 \
        <= rows[1]["collisionless_fraction"]


def test_adiabatic_plume_never_hands_over(adiabatic):
    """A real finding the tool must report, not paper over.

    Adiabatic monatomic expansion gives T ~ s^-2 and n ~ s^-3, so
    nu_ei ~ n T^-3/2 is CONSTANT while omega_pe ~ s^-3/2 falls: the plume
    becomes more collisional as it expands. If a hydrocode plume cools like
    this, Close et al.'s transition never happens.
    """
    idx, rows = find_handoff(read_m2c_series(adiabatic), CRIT)
    assert idx is None
    fr = [r["collisionless_fraction"] for r in rows]
    assert fr[-1] < fr[0]


def test_target_and_ambient_are_never_plume(iso):
    d, _run = iso
    t, path = read_m2c_series(d)[2]
    s = read_m2c_vtr(path, time=t)
    m = plume_mask(s, CRIT)
    X = np.broadcast_to(s.x[:, None, None], m.shape)
    assert not m[X >= 0].any()                      # target half
    assert not m[np.rint(s["materialid"]) == 0].any()


# --------------------------------------------------------------------------
# the handoff itself
# --------------------------------------------------------------------------

def test_synthetic_input_is_refused_by_default(iso):
    d, _run = iso
    t, path = read_m2c_series(d)[1]
    with pytest.raises(ValueError, match="synthetic"):
        extract(read_m2c_vtr(path, time=t), CRIT)


def test_reports_fletchers_three_quantities(iso):
    d, run = iso
    t, path = read_m2c_series(d)[1]
    h = extract(read_m2c_vtr(path, time=t), CRIT, allow_synthetic=True)
    ex = run["exact"][1]
    s = h.summary
    # all ionised material in front of the target, either direction, is the
    # exact constructed total; the handed-over charge is that minus backflow
    assert s["charge_in_front_any_direction_C"] == \
        pytest.approx(ex["N_e"] * E_CHARGE, rel=1e-6)
    assert s["total_charge_C"] + s["charge_excluded_backflow_C"] == \
        pytest.approx(ex["N_e"] * E_CHARGE, rel=1e-6)
    assert s["T_e_eV_mean"] == pytest.approx(ex["T_K"] * EV_PER_K, rel=1e-9)
    assert s["expansion_speed_mean_m_s"] > 0
    assert s["bulk_velocity_axial_m_s"] < 0         # leaving the target
    assert s["Zbar_mean"] == pytest.approx(ex["Zbar"], rel=1e-6)


def test_the_size_of_the_instantaneous_assumption_is_recorded(iso):
    d, _run = iso
    t, path = read_m2c_series(d)[1]
    h = extract(read_m2c_vtr(path, time=t), CRIT, allow_synthetic=True)
    still = 1 - h.summary["collisionless_fraction"]
    assert any("instantaneous" in n and f"{still:.0%}" in n for n in h.notes)


def test_handoff_round_trips_and_is_code_neutral(iso, tmp_path):
    d, _run = iso
    t, path = read_m2c_series(d)[1]
    h = extract(read_m2c_vtr(path, time=t), CRIT, allow_synthetic=True)
    npz, js = h.save(str(tmp_path / "plume"))
    meta = json.load(open(js))
    assert meta["format"] == "hvi_emp.handoff/1"
    assert meta["provenance"]["synthetic"] is True
    assert "Fletcher" in meta["provenance"]["strategy"]
    h2 = load_handoff(str(tmp_path / "plume"))
    for k in ("n_e", "T_K", "velocity", "cell_volume", "x", "y"):
        np.testing.assert_array_equal(h2.fields[k], h.fields[k])


def test_fields_outside_the_plume_are_zeroed(iso):
    d, _run = iso
    t, path = read_m2c_series(d)[1]
    h = extract(read_m2c_vtr(path, time=t), CRIT, allow_synthetic=True)
    off = h.fields["plume_mask"] == 0
    assert not h.fields["n_e"][off].any()
    assert not h.fields["velocity"][off].any()


def test_missing_output_fields_name_the_deck_fix(tmp_path):
    x, y, z = np.arange(3.0) - 3, np.arange(2.0) + 0.5, np.zeros(1)
    p = write_petsc_vtr(str(tmp_path / "m.vtr"), x, y, z,
                        {"density": np.ones((3, 2, 1))})
    with pytest.raises(KeyError, match="ElectronDensity"):
        assess(read_m2c_vtr(p), CRIT)


def test_backflow_is_excluded_and_reported_not_hidden(iso):
    """Electrons in material moving back toward the target.

    In the synthetic cloud the self-similar spread beats the drift on the
    target side, so part of it moves back. That charge is not handed over
    -- and the summary and notes must say how much.
    """
    d, run = iso
    t, path = read_m2c_series(d)[1]
    snap = read_m2c_vtr(path, time=t)
    h = extract(snap, CRIT, allow_synthetic=True)
    back = h.summary["charge_excluded_backflow_C"]
    assert back > 0
    # independent check from the constructed velocity field
    p = run["plume"]
    X, R = np.meshgrid(snap.x, snap.y, indexing="ij")
    _rho, vx, _vr, _T, _Z = p.state(X, R, t)
    n_e = snap["electron_density"][..., 0]
    vol = snap.cell_volumes()[..., 0]
    returning = (X < 0) & (vx >= 0) & (n_e > CRIT.min_n_e)
    assert back == pytest.approx((n_e * vol)[returning].sum() * E_CHARGE,
                                 rel=1e-6)
    assert any("moving back toward it" in n for n in h.notes)
    everything = HandoffCriteria(min_outward_speed=-1e30)
    h_all = extract(snap, everything, allow_synthetic=True)
    assert h_all.summary["charge_excluded_backflow_C"] == \
        pytest.approx(0.0, abs=1e-12)
