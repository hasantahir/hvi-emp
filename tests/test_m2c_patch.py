"""scripts/patch_m2c_tillotson.py: the M2C Tillotson robustness patch.

The patch edits third-party C++ we do not ship, so these tests check the
text surgery on the exact blocks it targets. The physics check -- compile
M2C's own VarFcnTillot.h and drive it -- runs when M2C_SRC points at an M2C
checkout and g++ is present (it is how the start-up abort was reproduced).
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
    """Every target block, with M2C's trailing whitespace on one line."""
    parts = ["// VarFcnTillot.h excerpt\n", "  int GetCaseWithRhoE(double rho, double e) {\n",
             pm.OLD_5, "\n  }\n", pm.OLD_1, "\n  }\n\n",
             "void\nVarFcnTillot::ExtendColdEnergyTrajectoryUpwards(double rhomax)\n{\n",
             "  if(mysize>2) {\n    drho0 = 1.0;\n", pm.OLD_2, "}\n\nvoid\n",
             pm.OLD_3, "\n\n", pm.OLD_8, "\n\n", pm.OLD_6, "\n  else if(rho<rhoIV) {\n",
             pm.OLD_4, "\n  }\n"]
    text = "".join(parts)
    return text.replace("[](double*){return false;},",
                        "[](double*){return false;}, ")


def _fake_riemann(n: int = pm.RIEMANN_COUNT) -> str:
    lines = []
    for k in range(n):
        sp = " " if k % 3 == 0 else ""          # M2C has both "if(" and "if ("
        lines.append(f"  if{sp}(rho_{k % 6}<=0 || c_{k % 6}_square<0) {{\n    return false;\n  }}\n")
    return "".join(lines)


def test_patch_applies_each_block_once():
    out = pm.patch_text(_fake_header())
    assert pm.is_patched(out)
    # 1: elat is set before the cold curve is built
    assert out.index("elat = eCV-eIV;") < out.index("SetupColdEnergyTrajectory();")
    # 2: the upward first step is capped
    assert "std::min(drho0, rhomax - rho_plus[mysize-1])" in out
    # 3: the downward RK45 call and its assert are gone
    down = out[out.index("ExtendColdEnergyTrajectoryDownwards"):]
    assert "MathTools::runge_kutta_45" not in down
    assert "assert(!err)" not in down
    assert "ecold_minus.push_back(eCV)" in down
    # 4-6, 6b: no exit(-1) left in any patched block
    for _, old, new in pm.EDITS:
        if "exit(-1)" in old:
            assert "exit(-1)" not in new
    assert "e = eCV;" in out
    assert "if(!(rho>0.0))" in out


def test_patch_is_idempotent_and_completes_an_earlier_patch():
    once = pm.patch_text(_fake_header())
    assert pm.patch_text(once) == once
    # a source patched by the first release (edits 1-3) gets only the rest
    first = _fake_header()
    for name, old, new in pm.EDITS[:3]:
        first = pm._block(old).sub(lambda _m, new=new: new, first, count=1)
    assert pm.missing_edits(first) == [n for n, _, _ in pm.EDITS[3:]]
    assert pm.is_patched(pm.patch_text(first))


def test_riemann_checks_become_nan_safe():
    out = pm.patch_riemann_text(_fake_riemann())
    assert pm.riemann_patched(out)
    assert "<=0 ||" not in out
    assert out.count("!(rho_") == pm.RIEMANN_COUNT
    assert pm.patch_riemann_text(out) == out


def test_patch_refuses_a_different_m2c():
    text = _fake_header().replace("drho0 = -1e-3*rho0;", "drho0 = -2e-3*rho0;")
    with pytest.raises(ValueError, match="downward cold curve"):
        pm.patch_text(text)
    with pytest.raises(ValueError, match="rarefaction-stage checks"):
        pm.patch_riemann_text(_fake_riemann(7))


def test_cli_patches_from_the_build_dir_and_keeps_the_originals(tmp_path):
    src = tmp_path / "m2c"
    (src / "build").mkdir(parents=True)
    hdr = src / "VarFcnTillot.h"
    rsv = src / pm.RIEMANN_FILE
    hdr.write_text(_fake_header())
    rsv.write_text(_fake_riemann())
    assert pm.main([str(src / "build"), "--check"]) == 1
    assert pm.main([str(src / "build")]) == 0
    assert pm.source_patched(src)
    assert (src / "VarFcnTillot.h.orig").read_text() == _fake_header()
    assert (src / (pm.RIEMANN_FILE + ".orig")).read_text() == _fake_riemann()
    assert pm.main([str(src), "--check"]) == 0


def test_refusal_changes_nothing(tmp_path):
    src = tmp_path / "m2c"
    src.mkdir()
    (src / "VarFcnTillot.h").write_text(_fake_header())
    (src / pm.RIEMANN_FILE).write_text(_fake_riemann(7))
    assert pm.main([str(src)]) == 1
    assert (src / "VarFcnTillot.h").read_text() == _fake_header()
    assert not (src / "VarFcnTillot.h.orig").exists()


def test_deck_says_m2c_needs_the_patch():
    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import M2CConfig
    al = get_material("Al")
    cfg = M2CConfig(projectile=al, target=al, diameter=1e-3, velocity=32e3)
    assert any("patch_m2c_tillotson" in n for n in cfg.notes)


def test_ambient_gas_pressure_is_capped():
    """A cell the metal leaves must not keep the metal's pressure."""
    from hvi_emp import get_material
    from hvi_emp.solvers import m2c_stage1 as m
    al = get_material("Al")
    cfg = m.M2CConfig(projectile=al, target=al, diameter=1e-3, velocity=32e3)
    deck = m.generate_input_file(cfg) if hasattr(m, "generate_input_file") else None
    gas = m._gas(cfg.ambient)
    p_cap = m.ambient_pressure_cap(gas, cfg.ambient_density(), cfg.velocity)
    c_cap = (gas["gamma"] * p_cap / cfg.ambient_density()) ** 0.5
    assert c_cap == pytest.approx(4 * 32e3)
    block = m._stiffened_gas_block(gas, cfg.ambient_density(), 0,
                                   velocity=cfg.velocity)
    assert "PressureUpperLimit" in block
    if deck is not None:
        assert "PressureUpperLimit" in deck


def test_near_vacuum_ambient_is_flagged_and_100_pa_is_light():
    from hvi_emp import get_material
    from hvi_emp.solvers import m2c_stage1 as m
    al = get_material("Al")
    vac = m.M2CConfig(projectile=al, target=al, diameter=1e-3, velocity=32e3)
    assert any("density floor" in n for n in vac.notes)
    gas = m.M2CConfig(projectile=al, target=al, diameter=1e-3, velocity=32e3,
                      ambient_pressure=m.ROBUST_AMBIENT_PA)
    assert not any("density floor" in n for n in gas.notes)
    assert gas.ambient_mass_fraction() < 0.01


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
    assert pm.source_patched(orig)
    assert pm.selftest(hdr) == 0
