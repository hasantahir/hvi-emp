"""The Tillotson EOS in the M2C deck.

Each test pins a failure that is silent in M2C: it starts, runs, and gives
a wrong answer. The source facts are from IoData.cpp (TillotsonModelData)
and VarFcnTillot.h at github.com/kevinwgy/m2c.
"""

import re

import numpy as np
import pytest

from hvi_emp import get_material
from hvi_emp.solvers.m2c_stage1 import (TILLOTSON, VERIFIED_KEYWORDS,
                                        VERIFIED_VALUES, M2CConfig,
                                        TillotsonParams, tillotson_for,
                                        tillotson_pressure, write_input)

AL = get_material("Al")
W = get_material("W")
P = TILLOTSON["Al"]

#: Every key TillotsonModelData::setup registers. Omit one and M2C uses the
#: value for WATER (Brundage 2013, Table 1).
ALL_KEYS = {"ReferenceDensity", "ReferenceSpecificInternalEnergy", "a", "b",
            "A", "B", "Alpha", "Beta", "IncipientVaporizationDensity",
            "IncipientVaporizationSpecificInternalEnergy",
            "CompleteVaporizationSpecificInternalEnergy",
            "SpecificHeatAtConstantVolume", "ReferenceTemperature",
            "TemperatureDependsOnDensity"}


def _deck(tmp_path, proj=AL, **kw):
    cfg = M2CConfig(projectile=proj, target=AL, diameter=1e-3,
                    velocity=32e3, **kw)
    return cfg, write_input(cfg, str(tmp_path / "input.st"))


def _tillotson_blocks(text):
    out = []
    for m in re.finditer(r"under TillotsonModel \{(.*?)\}", text, re.S):
        out.append(dict(re.findall(r"^\s*(\w+)\s*=\s*([^;]+);", m.group(1),
                                   re.M)))
    return out


# --------------------------------------------------------------------------
# what the deck says
# --------------------------------------------------------------------------

def test_every_key_is_written_or_m2c_uses_water(tmp_path):
    _cfg, text = _deck(tmp_path)
    blocks = _tillotson_blocks(text)
    assert len(blocks) == 2                       # target and projectile
    for b in blocks:
        assert set(b) == ALL_KEYS, f"missing: {ALL_KEYS - set(b)}"


def test_temperature_law_is_the_cold_curve_one(tmp_path):
    """The default law puts material at rest near -5300 K.

    T = T0 + (e - E0)/cv with e = 0 at rest and E0 = 5 MJ/kg. That number
    goes straight into the Saha solver.
    """
    _cfg, text = _deck(tmp_path)
    for b in _tillotson_blocks(text):
        assert b["TemperatureDependsOnDensity"].strip() == "Yes"
        assert float(b["SpecificHeatAtConstantVolume"]) > 0.0
    t_default_law = 300.0 + (0.0 - P.E0) / AL.cv_solid
    assert t_default_law < -5000.0                # the trap, quantified


def test_units_are_m2c_mm_g_s(tmp_path):
    _cfg, text = _deck(tmp_path)
    b = _tillotson_blocks(text)[0]
    assert float(b["ReferenceDensity"]) == pytest.approx(2.7e-3)    # g/mm^3
    assert float(b["ReferenceSpecificInternalEnergy"]) == \
        pytest.approx(5.0e12)                                       # mm^2/s^2
    assert float(b["A"]) == pytest.approx(75.2e9)                   # Pa
    assert float(b["CompleteVaporizationSpecificInternalEnergy"]) == \
        pytest.approx(13.9e12)


def test_source_is_written_into_the_deck(tmp_path):
    _cfg, text = _deck(tmp_path)
    assert "Melosh (1989)" in text and "Tillotson (1962)" in text


def test_every_emitted_key_and_value_is_grammar_checked(tmp_path):
    _cfg, text = _deck(tmp_path)
    for b in _tillotson_blocks(text):
        for k in b:
            assert k in VERIFIED_KEYWORDS
    assert "Tillotson" in VERIFIED_VALUES and "Yes" in VERIFIED_VALUES


# --------------------------------------------------------------------------
# what M2C will compute from it
# --------------------------------------------------------------------------

def test_zero_pressure_at_rest():
    assert tillotson_pressure(P, P.rho0, 0.0) == pytest.approx(0.0, abs=1e-6)


def test_sound_speed_at_rest_matches_the_independent_library_value():
    """Two independent data sets for Al agree to 2%.

    sqrt(A/rho0) from the Tillotson constants against the Mie-Grueneisen
    bulk sound speed in the material library. Neither was fitted to the
    other, so agreement is evidence the constants were transcribed right.
    """
    d = 1e-4 * P.rho0
    c2 = (tillotson_pressure(P, P.rho0 + d, 0.0)
          - tillotson_pressure(P, P.rho0 - d, 0.0)) / (2 * d)
    assert np.sqrt(c2) == pytest.approx(AL.c0, rel=0.03)


def test_hot_vapour_has_positive_pressure():
    """Above complete vaporisation the plume must push, not pull."""
    assert tillotson_pressure(P, 0.05 * P.rho0, 20e6) > 0.0


def test_expanded_partial_vapour_keeps_its_pressure():
    """The rho_IV regression.

    At 0.6 rho0 and 12 MJ/kg (between E_iv and E_cv) the blended branch
    gives positive vapour pressure. With the first default of
    rho_IV = 0.8 rho0 this state fell to the cold-expanded branch, came out
    tens of GPa in tension, and PressureCutOff clamped it to 1 Pa -- the
    p = 1 state seen in the failed 50 km/s run.
    """
    assert P.rho_iv < 0.6 * P.rho0
    assert tillotson_pressure(P, 0.6 * P.rho0, 12e6) > 1e9

    high = TillotsonParams(**{**P.__dict__, "rho_iv_ratio": 0.8})
    assert tillotson_pressure(high, 0.6 * high.rho0, 12e6) < 0.0


def test_rho_iv_sits_just_above_the_cold_curve_turnover():
    assert P.rho_turnover < P.rho_iv < 1.1 * P.rho_turnover
    # below the turnover, the compressed cold curve rises on expansion
    rho = np.linspace(0.2, 0.42, 50) * P.rho0
    p_cold = P.A * (rho / P.rho0 - 1) + P.B * (rho / P.rho0 - 1) ** 2
    assert np.all(np.diff(p_cold) < 0)            # unphysical: falls as rho rises


# --------------------------------------------------------------------------
# validation mirrors M2C's own aborts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("change, match", [
    ({"E_cv": 2.0e6}, "E_cv"),
    ({"rho_iv_ratio": 1.2}, "rho_IV"),
    ({"E0": 0.0}, "E0"),
    ({"rho_iv_ratio": 0.3}, "turnover"),
    ({"source": "  "}, "source"),
])
def test_bad_constant_sets_fail_before_a_job_queues(change, match):
    with pytest.raises(ValueError, match=match):
        TillotsonParams(**{**P.__dict__, **change}).validate("Al")


# --------------------------------------------------------------------------
# materials without verified constants
# --------------------------------------------------------------------------

def test_tungsten_has_no_invented_constants():
    assert tillotson_for(W) is None


def test_auto_keeps_unsourced_material_on_mie_gruneisen_and_says_so(tmp_path):
    cfg, text = _deck(tmp_path, proj=W)
    assert re.findall(r"EquationOfState = (\w+);", text) == \
        ["StiffenedGas", "Tillotson", "ExtendedMieGruneisen"]
    assert any("W uses ExtendedMieGruneisen" in n for n in cfg.notes)


def test_strict_tillotson_refuses_rather_than_guessing():
    with pytest.raises(ValueError, match="no verified Tillotson set for W"):
        M2CConfig(projectile=W, target=AL, diameter=1e-3, velocity=32e3,
                  eos="tillotson")


def test_old_behaviour_is_still_available_for_comparison(tmp_path):
    _cfg, text = _deck(tmp_path, eos="mie-gruneisen")
    assert "Tillotson" not in re.findall(r"EquationOfState = (\w+);", text)
