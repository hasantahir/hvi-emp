"""scripts/patch_m2c_tillotson.py: the M2C Tillotson cold-curve patch.

The patch edits third-party C++ we do not ship, so these tests check the
text surgery on the exact blocks it targets. The physics check -- compile
M2C's own VarFcnTillot.h and drive it -- runs when M2C_SRC points at an M2C
checkout and g++ is present (it is how the abort was reproduced).
"""

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_m2c_tillotson.py"
_spec = importlib.util.spec_from_file_location("patch_m2c_tillotson", _SCRIPT)
pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pm)


def _fake_header() -> str:
    """The three target blocks, with M2C's trailing whitespace on one line."""
    parts = ["// VarFcnTillot.h excerpt\n", pm.OLD_1, "\n  }\n\n",
             "void\nVarFcnTillot::ExtendColdEnergyTrajectoryUpwards(double rhomax)\n{\n",
             "  if(mysize>2) {\n    drho0 = 1.0;\n", pm.OLD_2, "}\n\nvoid\n",
             pm.OLD_3, "\n"]
    text = "".join(parts)
    return text.replace("[](double*){return false;},",
                        "[](double*){return false;}, ")


def test_patch_applies_each_block_once():
    out = pm.patch_text(_fake_header())
    assert out.count(pm.MARK) >= 3
    # 1: elat is set before the cold curve is built
    assert out.index("elat = eCV-eIV;") < out.index("SetupColdEnergyTrajectory();")
    # 2: the upward first step is capped
    assert "std::min(drho0, rhomax - rho_plus[mysize-1])" in out
    # 3: the downward RK45 call and its assert are gone
    down = out[out.index("ExtendColdEnergyTrajectoryDownwards"):]
    assert "MathTools::runge_kutta_45" not in down
    assert "assert(!err)" not in down
    assert "ecold_minus.push_back(eCV)" in down


def test_patch_is_idempotent():
    once = pm.patch_text(_fake_header())
    assert pm.patch_text(once) == once


def test_patch_refuses_a_different_m2c():
    text = _fake_header().replace("drho0 = -1e-3*rho0;", "drho0 = -2e-3*rho0;")
    with pytest.raises(ValueError, match="downward cold curve"):
        pm.patch_text(text)


def test_cli_patches_from_the_build_dir_and_keeps_the_original(tmp_path):
    src = tmp_path / "m2c"
    (src / "build").mkdir(parents=True)
    hdr = src / "VarFcnTillot.h"
    hdr.write_text(_fake_header())
    assert pm.main([str(src / "build"), "--check"]) == 1
    assert pm.main([str(src / "build")]) == 0
    assert pm.is_patched(hdr.read_text())
    assert (src / "VarFcnTillot.h.orig").read_text() == _fake_header()
    assert pm.main([str(src), "--check"]) == 0


def test_deck_says_m2c_needs_the_patch():
    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import M2CConfig
    al = get_material("Al")
    cfg = M2CConfig(projectile=al, target=al, diameter=1e-3, velocity=32e3)
    assert any("patch_m2c_tillotson" in n for n in cfg.notes)


_M2C_SRC = os.environ.get("M2C_SRC")


@pytest.mark.skipif(not (_M2C_SRC and shutil.which("g++")),
                    reason="set M2C_SRC to an M2C checkout (and have g++)")
def test_selftest_against_real_m2c(tmp_path):
    """Unpatched M2C aborts; the patched copy passes."""
    orig = tmp_path / "orig"
    shutil.copytree(_M2C_SRC, orig, ignore=shutil.ignore_patterns("build", ".git"))
    hdr = orig / "VarFcnTillot.h"
    if pm.is_patched(hdr.read_text()):
        pytest.skip("M2C_SRC is already patched")
    assert pm.selftest(hdr) == 1
    assert pm.main([str(orig)]) == 0
    assert pm.selftest(hdr) == 0
