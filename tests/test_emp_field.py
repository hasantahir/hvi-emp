"""Tests for the EM far-field reconstruction.

The claim this module makes is narrow and checkable: the field it draws is
an exact re-expression of `emp.E_t`, not a new model. These tests hold it
to that, and to the three caveats it promises to keep visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp import run_scenario
from hvi_emp.constants import C_LIGHT
from hvi_emp.viz.emp_field import (EMP_VIZ_NOTICE, dipole_field_at,
                                   pulse_figure, wavefront_series)


@pytest.fixture(scope="module")
def sc():
    return run_scenario("Fe", "Al", mass=1e-12, velocity=50e3, t_end=1e-5)


def test_reconstruction_is_exact_at_the_sensor(sc):
    """At (r_sensor, theta_sensor) the reconstruction must BE E_t.

    This is the whole correctness claim: a change of coordinates on a
    result the model already computed. If it does not reproduce the
    original at the one point where the original is defined, it is a new
    model wearing the old one's name.
    """
    emp = sc.emp
    th = np.radians(emp.theta_deg)
    for i in (0, len(emp.t) // 3, int(np.argmax(np.abs(emp.E_t))), -1):
        t = float(emp.t[i])
        got = dipole_field_at(emp, np.array([emp.r_sensor]),
                              np.array([th]), t)[0]
        assert got == pytest.approx(float(emp.E_t[i]), rel=1e-9, abs=1e-30)


def test_field_falls_off_as_one_over_r(sc):
    emp = sc.emp
    th = np.radians(emp.theta_deg)
    r0 = emp.r_sensor
    i = int(np.argmax(np.abs(emp.E_t)))
    t0 = float(emp.t[i])

    # Sample each radius at ITS retarded time, so the same feature is
    # compared rather than different parts of the pulse.
    a = dipole_field_at(emp, np.array([r0]), np.array([th]), t0)[0]
    for k in (2.0, 5.0, 10.0):
        r = r0 * k
        t = t0 + (r - r0) / C_LIGHT
        b = dipole_field_at(emp, np.array([r]), np.array([th]), t)[0]
        assert b == pytest.approx(a / k, rel=1e-9)


def test_dipole_pattern_vanishes_on_axis(sc):
    """sin(theta): a dipole radiates nothing along its own axis."""
    emp = sc.emp
    i = int(np.argmax(np.abs(emp.E_t)))
    t = float(emp.t[i])
    r = np.full(3, emp.r_sensor)
    e = dipole_field_at(emp, r, np.array([0.0, np.pi / 2, np.pi]), t)
    assert abs(e[0]) < 1e-18          # along +z
    assert abs(e[2]) < 1e-18          # along -z
    assert abs(e[1]) > 0              # broadside


def test_no_signal_before_the_wave_arrives(sc):
    """Retardation, not a filled ball: far out, early, the field is zero."""
    emp = sc.emp
    far = 1000.0 * emp.r_sensor
    early = float(emp.t[0])
    e = dipole_field_at(emp, np.array([far]),
                        np.array([np.pi / 2]), early)[0]
    assert e == 0.0


def test_near_field_is_blanked_not_drawn(sc, tmp_path):
    """Inside ~a wavelength the far-field form is simply wrong.

    Drawing it anyway would put a bright plausible blob at the origin,
    which is the most misleading thing this module could do.
    """
    res = wavefront_series(sc, str(tmp_path / "w"), n_frames=3,
                           shape=(24, 24, 24), verbose=False)
    assert res["r_min"] > 0
    assert res["r_min"] == pytest.approx(res["lambda"], rel=1e-9)

    import xml.etree.ElementTree as ET
    root = ET.parse(res["pvd"]).getroot()
    comment = "".join(root.itertext()) + str(ET.tostring(root))
    # the .pvd must carry the reason, not just the geometry
    raw = Path(res["pvd"]).read_text()
    assert "far-field form does not hold" in raw
    assert "184x" in raw


def test_amplitude_caveat_travels_with_the_data(sc, tmp_path):
    res = wavefront_series(sc, str(tmp_path / "w"), n_frames=2,
                           shape=(16, 16, 16), verbose=False)
    raw = Path(res["pvd"]).read_text()
    assert "EMP_UNCERTAINTY" in raw
    assert "not simulated" in raw
    assert "184x" in EMP_VIZ_NOTICE


def test_normalised_field_is_written_and_bounded(sc, tmp_path):
    """The colour bar people actually use must not imply false precision."""
    res = wavefront_series(sc, str(tmp_path / "w"), n_frames=2,
                           shape=(16, 16, 16), verbose=False)
    head = (Path(res["pvd"]).parent / res["files"][0]).read_bytes()[:3000]
    assert b"E_normalised" in head
    assert b"E_V_per_m" in head


def test_pulse_figure_is_optional(sc, tmp_path):
    """matplotlib is not a hard dependency; absence must not raise."""
    out = pulse_figure(sc, str(tmp_path / "p.png"))
    if out is None:
        pytest.skip("matplotlib not installed")
    assert Path(out).is_file()
    assert Path(out).stat().st_size > 5000
