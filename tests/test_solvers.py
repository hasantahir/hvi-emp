"""Tests for the LAMMPS / WarpX / OpenMHD bridges.

Deck generation and unit-conversion tests always run (no solver required).
The live LAMMPS test runs only when the ``lammps`` wheel is importable, and
is kept at true smoke scale (~2e4 atoms, ~1 ps) so it stays under a couple of
minutes on one core; run ``examples/05_lammps_impact.py`` for a physical run.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp import run_scenario
from hvi_emp.constants import AMU, EV
from hvi_emp.solvers import availability, have_lammps
from hvi_emp.solvers.lammps_stage1 import (LammpsImpactConfig, MD_LIBRARY,
                                           generate_input_deck)
from hvi_emp.solvers.openmhd_stage2 import (OpenMHDConfig, cavity_estimates,
                                            write_model_f90, write_run_notes)
from hvi_emp.solvers.warpx_stage3 import (WarpXEMPConfig, write_picmi_script,
                                          write_warpx_inputs)

warnings.simplefilter("ignore")


@pytest.fixture(scope="module")
def scenario():
    return run_scenario("Fe", "Al", mass=1e-12, velocity=50e3)


# ===========================================================================
# Availability report
# ===========================================================================

def test_availability_report_shape():
    av = availability()
    for key in ("lammps", "pywarpx", "picmi", "openpmd",
                "stage1_lammps", "stage3_warpx", "stage2_openmhd"):
        assert key in av


# ===========================================================================
# LAMMPS bridge: deck generation (no solver needed)
# ===========================================================================

def test_fraile_ke_per_atom_identity():
    """The paper's unit check: KE/atom = 0.952 eV x (v [km/s])^2 for W."""
    cfg = LammpsImpactConfig(material="W", velocity=1.0e3)
    assert np.isclose(cfg.ke_per_atom_eV, 0.952, rtol=5e-3)
    cfg9 = LammpsImpactConfig(material="W", velocity=9.0e3)
    assert np.isclose(cfg9.ke_per_atom_eV, 0.952 * 81.0, rtol=5e-3)


def test_deck_contains_fraile_protocol():
    cfg = LammpsImpactConfig(material="W", velocity=9e3,
                             projectile_radius=1.3e-9,
                             timestep=1e-16, damping_walls=True)
    deck = generate_input_deck(cfg, potential_dir="/pot")
    # the protocol elements of Fraile et al. (2022)
    assert "units           metal" in deck
    assert "boundary        p p m" in deck          # PBC x/y, free z
    assert "minimize" in deck
    assert "nvt temp 300.0 300.0" in deck           # 300 K equilibration
    assert "fix             fnve all nve" in deck   # NVE production
    assert "timestep        0.000100" in deck       # 0.1 fs in ps
    assert "fix             fdamp gwalls viscous" in deck
    assert "W_zhou.eam.alloy" in deck               # bundled default
    assert "pair_style      eam/alloy" in deck


def test_deck_velocity_direction_and_magnitude():
    cfg = LammpsImpactConfig(material="W", velocity=9e3)
    deck = generate_input_deck(cfg, potential_dir="/pot")
    # 9 km/s -> -90 Angstrom/ps, downward
    assert "velocity        gproj set 0.0 0.0 -90.0000 units box" in deck


def test_deck_potential_override():
    cfg = LammpsImpactConfig(material="W",
                             potential_file="/potentials/W_marinica.eam.fs",
                             pair_style="eam/fs")
    deck = generate_input_deck(cfg)
    assert "/potentials/W_marinica.eam.fs" in deck
    assert "pair_style      eam/fs" in deck


def test_target_auto_sizing_scales_with_projectile():
    small = LammpsImpactConfig(material="W", projectile_radius=0.5e-9)
    large = LammpsImpactConfig(material="W", projectile_radius=2.0e-9)
    assert large.estimated_atom_count() > 10 * small.estimated_atom_count()
    nx, ny, nz = large.resolved_cells()
    # lateral extent must exceed several projectile diameters
    a0_m = large.md().a0 * 1e-10
    assert nx * a0_m > 5.0 * 2.0 * large.projectile_radius


def test_unknown_material_needs_full_specification():
    with pytest.raises(KeyError, match="not in MD_LIBRARY"):
        LammpsImpactConfig(material="Xx").md()


def test_md_library_masses_match_framework():
    from hvi_emp import get_material
    for sym in ("W", "Al", "Cu", "Fe"):
        assert np.isclose(MD_LIBRARY[sym].mass_amu,
                          get_material(sym).A, rtol=1e-3)


# ===========================================================================
# LAMMPS bridge: live smoke test
# ===========================================================================

@pytest.mark.skipif(not have_lammps(),
                    reason="lammps python module not importable "
                           "(pip install lammps mpich)")
def test_lammps_live_smoke():
    """Tiny live impact: W sphere at 6 km/s, ~1 ps.

    Checks the machinery end to end -- run, ejecta extraction, energy
    accounting, Stage-2 handoff -- not the physics (the system is far too
    small and brief for converged ejecta statistics).
    """
    from hvi_emp.solvers.lammps_stage1 import run_impact

    cfg = LammpsImpactConfig(
        material="W", velocity=6e3, projectile_radius=0.5e-9,
        target_cells=(16, 16, 12),
        t_equilibrate=1e-13, t_production=8e-13, timestep=2e-16)
    res = run_impact(cfg, keep_ejecta_arrays=True)

    assert res.n_atoms == pytest.approx(cfg.estimated_atom_count(), rel=0.3)
    assert res.diagnostics["n_projectile"] > 0
    # NVE energy conservation (loose: 2 fs step at impact speeds)
    assert abs(res.energy_drift) < 0.05
    # the target must have heated
    assert res.T_target_final > 320.0

    handoff = res.to_impact_handoff()
    assert handoff.m_plasma >= 0.0
    assert handoff.v_expansion > 0.0


# ===========================================================================
# WarpX bridge
# ===========================================================================

def test_warpx_config_from_expansion(scenario):
    cfg = WarpXEMPConfig.from_expansion(scenario.expansion,
                                        at_frequency=916e6)
    # initialised at the 916 MHz resonant epoch => f_pe ~ 916 MHz
    assert np.isclose(cfg.omega_pe / (2 * np.pi), 916e6, rtol=0.3)
    assert cfg.R_plume > 1e-4
    assert cfg.m_ion > 1e-26


def test_warpx_electron_drift_assumption(scenario):
    """The Fletcher & Close closure: v_e = sqrt(m_i/m_e) v_i (capped)."""
    from hvi_emp.constants import C_LIGHT, M_ELECTRON
    cfg = WarpXEMPConfig.from_expansion(scenario.expansion,
                                        at_frequency=916e6)
    expected = np.sqrt(cfg.m_ion / M_ELECTRON) * cfg.v_drift
    assert np.isclose(cfg.v_electron_drift, min(expected, 0.25 * C_LIGHT),
                      rtol=1e-12)


def test_warpx_cases_match_paper(scenario):
    """cold: v_t = 0.1 v_d; warm: v_t = 10 v_d (their Sec. III)."""
    cold = WarpXEMPConfig.from_expansion(scenario.expansion,
                                         at_frequency=916e6, case="cold")
    warm = WarpXEMPConfig.from_expansion(scenario.expansion,
                                         at_frequency=916e6, case="warm")
    assert np.isclose(cold.v_thermal_electron,
                      0.1 * cold.v_electron_drift, rtol=1e-12)
    assert np.isclose(warm.v_thermal_electron,
                      10.0 * warm.v_electron_drift, rtol=1e-12)


def test_warpx_grid_constraints(scenario):
    cfg = WarpXEMPConfig.from_expansion(scenario.expansion,
                                        at_frequency=916e6)
    g = cfg.grid()
    # CFL for the 2D Yee solver
    from hvi_emp.constants import C_LIGHT
    assert g["dt"] <= 0.99 * g["dx"] / (C_LIGHT * np.sqrt(2.0))
    # plasma frequency resolved
    assert g["dt"] <= 0.1 / cfg.omega_pe * (1 + 1e-12)
    assert g["n_steps"] > 100


def test_warpx_deck_generation(scenario, tmp_path):
    cfg = WarpXEMPConfig.from_expansion(scenario.expansion,
                                        at_frequency=916e6)
    script = write_picmi_script(cfg, str(tmp_path / "warpx_emp.py"))
    inputs = write_warpx_inputs(cfg, str(tmp_path / "inputs"))
    for text in (script, inputs):
        assert f"{cfg.n_e0:.4e}" in text or f"{cfg.n_e0:.6e}" in text
    assert "picmi.Simulation" in script
    assert "openpmd" in script
    assert "algo.maxwell_solver = yee" in inputs
    assert "boundary.field_lo = pml pml" in inputs
    assert (tmp_path / "warpx_emp.py").exists()
    # the generated PICMI script must at least be valid Python
    compile(script, "warpx_emp.py", "exec")


# ===========================================================================
# OpenMHD bridge
# ===========================================================================

def test_cavity_estimates_scaling():
    """R_c ~ (E/B^2)^(1/3): doubling B at fixed E shrinks the cavity by
    2^(2/3)."""
    a = cavity_estimates(1.0, 3e-5, 30e3)
    b = cavity_estimates(1.0, 6e-5, 30e3)
    assert np.isclose(a["R_cavity"] / b["R_cavity"], 2.0 ** (2.0 / 3.0),
                      rtol=1e-9)
    # pressure balance identity: displaced magnetic energy equals E_kin
    assert np.isclose(a["B_energy_displaced"], 1.0, rtol=1e-9)


def test_openmhd_config_from_expansion(scenario):
    cfg = OpenMHDConfig.from_expansion(scenario.expansion)
    n = cfg.normalised()
    assert n["rho_plume_hat"] > 1.0          # plume denser than ambient
    assert n["v_alfven"] > 1e4               # LEO Alfven speed ~ 100s km/s
    assert cfg.R0 > 1e-4


def test_openmhd_model_f90_generation(scenario, tmp_path):
    cfg = OpenMHDConfig.from_expansion(scenario.expansion)
    f90 = write_model_f90(cfg, str(tmp_path / "model.f90"))
    notes = write_run_notes(cfg, str(tmp_path / "RUN.md"))
    assert "subroutine model" in f90
    assert "call v2u" in f90
    assert "diamagnetic" in f90.lower() or "cavity" in f90.lower()
    # normalisation must be recorded for SI conversion
    assert f"{cfg.normalised()['unit_time']:.6e}" in f90
    assert "SI conversion" in notes
    assert (tmp_path / "model.f90").exists()


def test_openmhd_field_angle():
    cfg = OpenMHDConfig(rho_plume=1e-6, rho_ambient=1e-15, p_plume=1.0,
                        p_ambient=1e-10, v_expansion=3e4, B_angle_deg=90.0)
    f90 = write_model_f90(cfg)
    assert "bx0 = 0.000000d0, by0 = 1.000000d0" in f90


if __name__ == "__main__":       # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))


# ===========================================================================
# Examples: optional-plotting behaviour
# ===========================================================================

def test_examples_run_without_matplotlib(tmp_path):
    """The examples must produce all their numerical output with matplotlib
    absent, skipping only the figure.

    Regression: they used to die at `import matplotlib` before computing
    anything, which made an optional plotting dependency look mandatory.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    blocker = tmp_path / "blocker.py"
    blocker.write_text(
        "import sys\n"
        "class _B:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'matplotlib' or name.startswith('matplotlib.'):\n"
        "            return self\n"
        "    def load_module(self, name):\n"
        "        raise ModuleNotFoundError(name)\n"
        "sys.meta_path.insert(0, _B())\n")

    runner = tmp_path / "run.py"
    runner.write_text(
        f"exec(open({str(blocker)!r}).read())\n"
        "import runpy, sys\n"
        "sys.argv = ['ex']\n"
        f"runpy.run_path({str(root / 'examples' / '02_velocity_sweep.py')!r},"
        " run_name='__main__')\n")

    r = subprocess.run([sys.executable, str(runner)], capture_output=True,
                       text=True, timeout=900, cwd=str(root))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "matplotlib is not installed" in r.stdout
    # the physics table must still be there
    assert "Q_frozen[C]" in r.stdout
    assert "wrote" not in r.stdout          # no figure claimed


def test_plotting_helper_stub_is_transparent():
    """The no-op stand-in must absorb the plotting calls the examples make."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from _plotting import _NoPlot

    plt = _NoPlot()
    fig, ax = plt.subplots(2, 3, figsize=(1, 1))     # unpacking
    ax[0, 0].loglog([1], [1], label="x")             # 2-D indexing + chaining
    ax[1, 2].set(xlabel="t")
    ax[0, 1].get_ylim()[1]                           # subscripting a result
    fig.suptitle("t")
    fig.tight_layout()
    fig.savefig("/dev/null")
    for a in ax:                                     # iteration
        a.grid(alpha=.3)
    assert not plt                                   # falsy, so `if plt:` works


# ===========================================================================
# Solver modules are package modules, not scripts
# ===========================================================================

def test_solver_modules_reject_direct_execution(tmp_path):
    """Copying a solver module out of the package and running it must give an
    actionable message, not a bare relative-import traceback.

    Regression: `python lammps_stage1.py` from the project root produced
    `ImportError: attempted relative import with no known parent package`,
    which says nothing about what to do instead.
    """
    import shutil
    import subprocess

    root = Path(__file__).resolve().parents[1]
    for mod in ("lammps_stage1", "warpx_stage3", "openmhd_stage2"):
        src = root / "hvi_emp" / "solvers" / f"{mod}.py"
        dst = tmp_path / f"{mod}.py"
        shutil.copy(src, dst)
        r = subprocess.run([sys.executable, str(dst)], capture_output=True,
                           text=True, timeout=120, cwd=str(tmp_path))
        assert r.returncode != 0, mod
        msg = r.stdout + r.stderr
        assert "cannot be run as a" in msg, (mod, msg[:400])
        assert "hvi_emp/solvers/" in msg
        assert "attempted relative import" not in msg, (
            f"{mod}: raw ImportError leaked through the guard")


def test_solver_modules_runnable_with_dash_m():
    """`python -m hvi_emp.solvers.<mod>` must work (it is what the guard
    message tells the user to do)."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    checks = {
        "hvi_emp.solvers.openmhd_stage2": ["--energy", "1e-3"],
        "hvi_emp.solvers.warpx_stage3": [],
    }
    for mod, extra in checks.items():
        r = subprocess.run([sys.executable, "-m", mod] + extra,
                           capture_output=True, text=True, timeout=300,
                           cwd=str(root))
        assert r.returncode == 0, (mod, r.stderr[-1500:])
        assert r.stdout.strip(), mod


def test_cli_lammps_subcommand(tmp_path):
    """`python -m hvi_emp lammps --out ...` writes a runnable deck."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "in.impact"
    r = subprocess.run(
        [sys.executable, "-m", "hvi_emp", "--quiet", "lammps",
         "--material", "W", "--velocity", "9e3", "--radius", "5e-10",
         "--out", str(out)],
        capture_output=True, text=True, timeout=300, cwd=str(root))
    assert r.returncode == 0, r.stderr[-1500:]
    deck = out.read_text()
    assert "units           metal" in deck
    assert "fix             fnve all nve" in deck
    assert "Fraile" in deck


# ===========================================================================
# LAMMPS availability diagnostics
# ===========================================================================

def test_lammps_diagnostics_shape():
    from hvi_emp.solvers import lammps_diagnostics
    d = lammps_diagnostics()
    assert set(d) == {"ok", "stage", "reason", "hint"}
    assert d["stage"] in ("ok", "import", "instantiate")
    assert isinstance(d["reason"], str) and d["reason"]
    if d["ok"]:
        assert d["hint"] == ""
    else:
        assert d["hint"], "a failure must come with an actionable hint"


def test_lammps_diagnostics_distinguishes_missing_from_broken(monkeypatch):
    """'Not installed' and 'installed but will not load' need different fixes,
    so they must not be collapsed into one message.

    Regression: `have_lammps()` swallowed every exception, so an installed-
    but-unloadable LAMMPS (the common case -- missing libmpi) was reported as
    'not installed: pip install lammps'.
    """
    import importlib

    import hvi_emp.solvers as S

    real_import = importlib.import_module

    def missing(name, *a, **k):
        if name == "lammps":
            raise ModuleNotFoundError("No module named 'lammps'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(S.importlib, "import_module", missing)
    d = S.lammps_diagnostics()
    assert not d["ok"] and d["stage"] == "import"
    assert "No module named" in d["reason"]
    assert "install" in d["hint"]
    monkeypatch.undo()

    # installed, but the shared library will not load
    def broken():
        raise OSError("libmpi.so.12: cannot open shared object file")

    monkeypatch.setattr(S, "_preload_mpi", broken)
    d2 = S.lammps_diagnostics()
    if d2["stage"] != "ok":            # only meaningful where lammps imports
        assert not d2["ok"]
        assert "libmpi" in d2["reason"] or "OSError" in d2["reason"]
        assert "installed but" in d2["hint"]


def _lammps_missing(monkeypatch):
    """Make ``import lammps`` fail so the hint path is exercised."""
    import importlib

    import hvi_emp.solvers as S

    real_import = importlib.import_module

    def missing(name, *a, **k):
        if name == "lammps":
            raise ModuleNotFoundError("No module named 'lammps'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(S.importlib, "import_module", missing)
    return S


def test_lammps_hint_is_env_aware(monkeypatch):
    S = _lammps_missing(monkeypatch)

    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    assert "pip install lammps" in S.lammps_diagnostics()["hint"]


def test_lammps_hint_names_the_installed_manager(monkeypatch):
    """The hint must name a command this machine actually has.

    The original bug: the hint said "conda install" unconditionally, so a
    micromamba-only workstation was told to run a binary it did not have.
    """
    import shutil

    import hvi_emp.pkgmgr as P
    S = _lammps_missing(monkeypatch)
    monkeypatch.setenv("CONDA_PREFIX", "/opt/envs/hvi")

    for present, expected in (("micromamba", "micromamba install"),
                              ("mamba", "mamba install"),
                              ("conda", "conda install")):
        monkeypatch.setattr(
            P.shutil, "which",
            lambda n, _p=present: f"/usr/bin/{n}" if n == _p else None)
        hint = S.lammps_diagnostics()["hint"]
        assert expected in hint, f"{present}: got {hint!r}"

    # mamba present alongside conda -> mamba wins.
    monkeypatch.setattr(
        P.shutil, "which",
        lambda n: f"/usr/bin/{n}" if n in ("mamba", "conda") else None)
    hint = S.lammps_diagnostics()["hint"]
    assert "mamba install" in hint
    assert "conda install" not in hint

    shutil.which  # keep the import meaningful for linters


def test_install_advice_prefers_mamba_over_conda(monkeypatch):
    """Regression guard: never go back to conda-first advice.

    Conda is kept as a fallback on purpose, but when a mamba-family solver
    is on PATH it must be the one recommended -- that preference is the
    whole point of this module.
    """
    import hvi_emp.pkgmgr as P

    monkeypatch.setattr(
        P.shutil, "which",
        lambda n: f"/usr/bin/{n}" if n in P.PREFERENCE else None)
    pm = P.detect(probe=False)
    assert pm.name == "micromamba"
    assert P.PREFERENCE.index("micromamba") < P.PREFERENCE.index("mamba")
    assert P.PREFERENCE.index("mamba") < P.PREFERENCE.index("conda")


def test_availability_reports_lammps_reason():
    from hvi_emp.solvers import availability
    av = availability()
    assert "lammps_reason" in av
    assert av["stage1_lammps"]


# ===========================================================================
# iSALE bridge (Fletcher 2021 replication)
# ===========================================================================

def test_isale_deck_matches_isale_format():
    from hvi_emp.solvers.isale_stage1 import (ISaleImpactConfig,
                                              write_asteroid_inp,
                                              write_material_inp)
    cfg = ISaleImpactConfig(velocity=32e3, projectile_diameter=1e-3, cppr=20)
    deck = write_asteroid_inp(cfg)
    # iSALE2D v4.1 input keys
    for key in ("VERSION", "GRIDH", "GRIDV", "GRIDSPC", "CYL", "OBJRESH",
                "OBJVEL", "OBJMAT", "LAYMAT", "TEND", "DTSAVE", "TR_VAR",
                "VARLIST"):
        assert key in deck, key
    assert "4.1" in deck
    assert "-3.2000E+04" in deck            # downward, m/s
    mat = write_material_inp(cfg)
    for key in ("MATNAME", "EOSNAME", "EOSTYPE", "STRMOD", "JC_A", "TMELT0"):
        assert key in mat, key


def test_isale_grid_spacing_follows_cppr():
    from hvi_emp.solvers.isale_stage1 import ISaleImpactConfig
    for cppr in (10, 20, 40):
        cfg = ISaleImpactConfig(projectile_diameter=1e-3, cppr=cppr)
        assert np.isclose(cfg.grid_spacing, 0.5e-3 / cppr, rtol=1e-12)


def test_isale_auto_size_contains_the_crater():
    """Regression: the iSALE validation example's 200x240 grid is far too
    small for these impacts -- the crater is tens of projectile diameters
    across, so the mesh must be auto-sized or the run is wasted."""
    from hvi_emp.solvers.isale_stage1 import ISaleImpactConfig

    cfg = ISaleImpactConfig(velocity=62e3, projectile_diameter=1e-3, cppr=20,
                            grid_h=200, grid_v=240)
    assert not cfg.crater_scaling_estimate()["domain_adequate"]
    cfg.auto_size()
    est = cfg.crater_scaling_estimate()
    assert est["domain_adequate"], est
    assert cfg.grid_h > 200


def test_isale_velocity_series_matches_fletcher(tmp_path):
    from hvi_emp.solvers.isale_stage1 import (FLETCHER_VELOCITIES,
                                              write_velocity_series)
    made = write_velocity_series(str(tmp_path / "s"), cores=128)
    assert len(made) == len(FLETCHER_VELOCITIES) == 6
    assert [m["velocity"] for m in made] == list(FLETCHER_VELOCITIES)
    for m in made:
        d = Path(m["dir"])
        assert (d / "asteroid.inp").exists()
        assert (d / "material.inp").exists()
        assert m["scaling"]["domain_adequate"], m["velocity"]
    runner = tmp_path / "s" / "run_all.sh"
    assert runner.exists()
    txt = runner.read_text()
    assert txt.count("&") >= 6          # all six launched concurrently
    # with >= 6 cores the series is the slowest case, not the sum
    assert made[0]["cost"]["concurrent_cases"] == 6


def test_isale_tracers_to_handoff():
    """Tracer peak pressures -> vapour/plasma inventory + Stage-2 handoff."""
    from hvi_emp.solvers.isale_stage1 import isale_tracers_to_handoff

    # a spread of peak pressures spanning the Al vaporisation threshold
    P = np.array([50e9, 200e9, 400e9, 800e9, 1600e9, 3000e9])
    out = isale_tracers_to_handoff(P, tracer_mass=1e-9, material="Al")
    assert 0.0 <= out["vapour_fraction"] <= 1.0
    assert out["m_plasma"] <= out["m_total"]
    assert out["T_eV"] >= 0.0
    h = out["handoff"]
    assert h.m_plasma >= 0 and h.v_expansion > 0
    if h.m_plasma > 0:
        from hvi_emp import simulate_expansion
        exp = simulate_expansion(h, t_end=1e-6)
        assert exp.Q_final >= 0.0


def test_fletcher_mhd_cases_cover_both_field_orientations(tmp_path):
    from hvi_emp.solvers.openmhd_stage2 import (FLETCHER_MHD_CASES,
                                                write_fletcher_mhd_cases)
    assert set(FLETCHER_MHD_CASES) == {"fig6_parallel", "fig7_perpendicular"}
    assert FLETCHER_MHD_CASES["fig6_parallel"]["B_angle_deg"] == 0.0
    assert FLETCHER_MHD_CASES["fig7_perpendicular"]["B_angle_deg"] == 90.0

    out = write_fletcher_mhd_cases(str(tmp_path / "mhd"))
    for name, d in out.items():
        p = Path(d["dir"])
        assert (p / "model.f90").exists()
        assert (p / "RUN.md").exists()
        assert (p / "CASE.md").exists()
        assert d["cavity"]["R_cavity"] > 0
    # the two must differ in the field direction written into model.f90
    a = (Path(out["fig6_parallel"]["dir"]) / "model.f90").read_text()
    b = (Path(out["fig7_perpendicular"]["dir"]) / "model.f90").read_text()
    assert "bx0 = 1.000000d0" in a and "by0 = 0.000000d0" in a
    assert "bx0 = 0.000000d0" in b and "by0 = 1.000000d0" in b


# ===========================================================================
# Environment doctor
# ===========================================================================

def test_doctor_hardware_probe():
    from hvi_emp.doctor import hardware
    hw = hardware()
    assert hw["cores_logical"] >= 1
    assert hw["cores_physical"] >= 1
    assert isinstance(hw["platform"], str) and hw["platform"]
    # memory/disk/gpu may be unavailable, but the keys must exist
    for k in ("memory_GB", "disk_free_GB", "gpu"):
        assert k in hw


def test_doctor_package_and_solver_probes():
    from hvi_emp.doctor import python_packages, solvers
    pkgs = dict((n, s) for n, s, _ in python_packages())
    assert "numpy" in pkgs and "scipy" in pkgs
    names = [n for n, _, _ in solvers()]
    for expect in ("LAMMPS", "WarpX", "iSALE", "OpenMHD", "ParaView"):
        assert any(expect in n for n in names), expect


def test_doctor_self_test_runs_the_real_chain():
    from hvi_emp.doctor import self_test
    r = self_test()
    assert r["scenario_ok"] and r["vti_ok"]
    # Known open discrepancies are not regressions; see validation.Check
    # .known_open. What must hold is that nothing NEW broke.
    assert r["validation_ok"], (
        f"{r.get('regressions', '?')} regression(s); {r['validation']}")
    assert r["T_eV"] > 0 and r["Q_frozen"] > 0
    assert r["vti_MB_per_frame"] > 0


def test_doctor_report_exits_clean():
    import io
    import contextlib

    from hvi_emp.doctor import report
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = report(skip_self_test=True)
    out = buf.getvalue()
    assert rc == 0
    for section in ("environment check", "Python packages", "Solvers",
                    "What you can run now", "Sizing for this machine"):
        assert section in out, section


def test_cli_doctor_subcommand():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-m", "hvi_emp", "doctor", "--quick"],
                       capture_output=True, text=True, timeout=300,
                       cwd=str(root))
    assert r.returncode == 0, r.stderr[-1500:]
    assert "environment check" in r.stdout


# ---------------------------------------------------------------------------
# Environment detection: things that are easy to get subtly wrong
# ---------------------------------------------------------------------------

def test_find_openmhd_looks_for_the_tree_not_the_path(tmp_path):
    """OpenMHD installs no executable, so $PATH lookup can never work.

    The build produces ``a.out``/``ap.out`` inside each problem directory.
    An earlier version searched $PATH for a binary called "OpenMHD" and so
    reported it missing on machines where it was built and working.

    Note the injected `roots`: an earlier version of this test relied on the
    default search, which scans ~/OpenMHD and ~/src/OpenMHD. It passed
    everywhere except on a machine that actually had OpenMHD installed --
    precisely the machine that matters.
    """
    from hvi_emp.doctor import find_openmhd

    assert find_openmhd(roots=[str(tmp_path / "nothing-here")]) is None

    root = tmp_path / "OpenMHD"
    (root / "2D_basic").mkdir(parents=True)
    (root / "param.h").write_text("")

    found = find_openmhd(roots=[str(root)])
    assert found is not None and str(root) in found
    assert "not yet compiled" in found

    (root / "2D_basic" / "ap.out").write_text("")
    assert "built: 2D_basic" in find_openmhd(roots=[str(root)])


def test_explicit_openmhd_env_var_is_authoritative(tmp_path, monkeypatch):
    """A wrong $OPENMHD must be reported, not silently replaced.

    Searching on past an explicit setting is how you end up debugging a
    different build from the one you configured.
    """
    from hvi_emp.doctor import find_openmhd

    # a perfectly good tree that the fallback search *would* find
    decoy = tmp_path / "OpenMHD"
    (decoy / "2D_basic").mkdir(parents=True)
    (decoy / "param.h").write_text("")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setenv("OPENMHD", str(tmp_path / "does-not-exist"))
    assert find_openmhd() is None, (
        "an explicitly-set but invalid $OPENMHD must not fall through to a "
        "different tree")

    monkeypatch.setenv("OPENMHD", str(decoy))
    assert str(decoy) in find_openmhd()

    # unset: now the conventional locations are fair game
    monkeypatch.delenv("OPENMHD")
    assert str(decoy) in find_openmhd()


def test_gpu_toolchains_reports_all_three():
    from hvi_emp.doctor import gpu_toolchains

    rows = gpu_toolchains()
    names = " ".join(n for n, _, _ in rows)
    for tool in ("nvfortran", "nvcc", "cmake"):
        assert tool in names, tool
    assert all(len(r) == 3 for r in rows)


def test_no_document_recommends_the_nonexistent_pywarpx_package():
    """`pip install pywarpx` 404s on PyPI. Guard against it creeping back.

    The module ``pywarpx`` is real -- it comes from a conda-forge, Spack or
    source build of WarpX -- but there has never been a PyPI distribution of
    that name, and advising one sends the user into an unfixable error.
    """
    import re

    root = Path(__file__).resolve().parents[1]
    bad = re.compile(r"pip\s+install\s+[^\n`]*\bpywarpx\b")
    # The string is allowed to appear when the surrounding line is *warning*
    # about it; only unqualified recommendations are a defect.
    disclaimed = re.compile(
        r"\bnot\b|\bno such\b|\bdoes not\b|\bdoesn't\b|404|\bfail",
        re.IGNORECASE)

    offenders = []
    for path in list(root.rglob("*.md")) + list(root.rglob("*.py")):
        if ".git" in path.parts or path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for m in bad.finditer(text):
            i = text[:m.start()].count("\n")
            # look at the sentence, which may wrap: this line and its neighbours
            context = " ".join(lines[max(0, i - 1):i + 2])
            if disclaimed.search(context):
                continue
            offenders.append(f"{path.relative_to(root)}:{i + 1}: {m.group(0)}")
    assert not offenders, (
        "advises a PyPI package that does not exist:\n  "
        + "\n  ".join(offenders))


def test_conda_forge_warpx_is_not_advertised_as_a_gpu_build():
    """conda-forge's warpx is CPU-only (warpx-feedstock issue 89).

    Pairing that command with a GPU promise is the kind of error that costs
    someone a day before they discover the build cannot use their card.
    """
    root = Path(__file__).resolve().parents[1]
    for name in ("docs/INSTALLING_SOLVERS.md", "docs/WORKSTATION_QUICKSTART.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert "conda-forge" in text and "warpx" in text.lower(), name
        assert ("CPU only" in text or "CPU-only" in text), (
            f"{name} mentions conda-forge warpx without saying it is CPU-only")


# ===========================================================================
# LAMMPS upstream version-parsing bug (conda-forge installs)
# ===========================================================================

def test_normalise_lammps_version_trims_extra_components():
    from hvi_emp.solvers import _normalise_lammps_version as n
    assert n("2025.7.22.4.0") == "2025.7.22"
    assert n("2022.6.23.4.0") == "2022.6.23"
    # Already-valid and unexpected shapes are left alone.
    assert n("2025.7.22") == "2025.7.22"
    assert n("20250722") == "20250722"
    assert n("") == ""


def test_upstream_version_logic_fails_then_works_under_the_shim():
    """Reproduce LAMMPS's get_version_number() against real metadata.

    This runs the exact expression from upstream lammps/__init__.py against
    the version actually recorded for the installed distribution, so the
    test fails if upstream changes the format string or the packaging
    changes the version shape -- rather than asserting against a string we
    made up.
    """
    import importlib.metadata as md
    import time

    from hvi_emp.solvers import _lammps_version_shim

    try:
        raw = md.version("lammps")
    except md.PackageNotFoundError:
        pytest.skip("lammps distribution metadata not installed")

    def upstream_parse(v):
        t = time.strptime(v, "%Y.%m.%d")
        return t.tm_year * 10000 + t.tm_mon * 100 + t.tm_mday

    if len(raw.split(".")) > 3:
        # The bug is present in this environment: confirm it, then confirm
        # the shim fixes it.
        with pytest.raises(ValueError, match="unconverted data remains"):
            upstream_parse(raw)

    with _lammps_version_shim():
        shimmed = md.version("lammps")
    assert len(shimmed.split(".")) <= 3
    assert upstream_parse(shimmed) > 19000000


def test_version_shim_is_reverted_afterwards():
    """A permanent monkeypatch would be worse than the bug it fixes."""
    import importlib.metadata as md

    from hvi_emp.solvers import _lammps_version_shim

    before = md.version
    with _lammps_version_shim():
        assert md.version is not before
    assert md.version is before

    # Reverted even when the body raises.
    try:
        with _lammps_version_shim():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert md.version is before


def test_shim_only_touches_lammps():
    import importlib.metadata as md

    from hvi_emp.solvers import _lammps_version_shim

    real_numpy = md.version("numpy")
    with _lammps_version_shim():
        assert md.version("numpy") == real_numpy


def test_import_lammps_works():
    from hvi_emp.solvers import import_lammps
    mod = import_lammps()
    assert hasattr(mod, "lammps")


def test_version_bug_hint_does_not_blame_mpi(monkeypatch):
    """Regression: this failure used to be reported as a missing MPI runtime.

    'pip install mpich' cannot fix a strptime error, and sending someone
    down that path costs an afternoon.
    """
    import importlib

    import hvi_emp.solvers as S

    real_import = importlib.import_module

    def boom(name, *a, **k):
        if name == "lammps":
            raise ValueError("unconverted data remains: .4.0")
        return real_import(name, *a, **k)

    monkeypatch.setattr(S.importlib, "import_module", boom)
    monkeypatch.setattr(S, "import_lammps",
                        lambda: (_ for _ in ()).throw(
                            ValueError("unconverted data remains: .4.0")))

    d = S.lammps_diagnostics()
    assert d["ok"] is False
    hint = d["hint"]
    assert "strptime" in hint
    assert "mpich" not in hint.split("pip install lammps")[0]
    assert "bug in LAMMPS" in hint


# ===========================================================================
# MPI family mismatch (reported from a workstation that gained Open MPI)
# ===========================================================================

_ABORT_SRC = (
    "import os, sys\n"
    "sys.stderr.write('[reznor:1:0] ucp_context.c:2339 UCX WARN UCP API "
    "version is incompatible: required >= 1.20, actual 1.19.1\\n')\n"
    "sys.stderr.write('Abort(201947909) on node 0: Fatal error in "
    "internal_Comm_size: Invalid communicator\\n')\n"
    "os._exit(137)\n"
)


def test_mpi_abort_does_not_kill_the_caller(monkeypatch):
    """MPI_Abort must become a report, not terminate the interpreter.

    Reported: `python -m hvi_emp doctor` died mid-section with
    'Invalid communicator', so no report was produced at all -- not even
    the parts that had already succeeded.
    """
    import hvi_emp.solvers as S

    monkeypatch.setattr(S, "_PROBE_SRC", _ABORT_SRC)
    d = S.lammps_diagnostics()          # must return, not abort
    assert d["ok"] is False
    assert d["stage"] == "instantiate"
    assert "MPI" in d["reason"]


def test_mpi_abort_is_not_blamed_on_lammps(monkeypatch):
    """The hint must not send the user to reinstall LAMMPS."""
    import hvi_emp.solvers as S

    monkeypatch.setattr(S, "_PROBE_SRC", _ABORT_SRC)
    hint = S.lammps_diagnostics()["hint"]
    assert "not a LAMMPS fault" in hint
    assert "mpi=" in hint                    # names the actual lever


def test_doctor_completes_through_an_mpi_abort(monkeypatch, capsys):
    """One broken solver must not cost the user the whole report."""
    import hvi_emp.doctor as D
    import hvi_emp.solvers as S

    monkeypatch.setattr(S, "_PROBE_SRC", _ABORT_SRC)
    D.report()
    out = capsys.readouterr().out
    assert "LAMMPS (MD, Stage 1)" in out
    # sections *after* the solver table must still be present
    assert "Sizing for this machine" in out


def test_preload_mpi_reads_the_soname_instead_of_assuming(monkeypatch):
    """Regression: the soname used to be hardcoded to MPICH's libmpi.so.12.

    Forcing MPICH into a process holding Open MPI communicators is what
    produced 'Invalid communicator'.
    """
    import hvi_emp.solvers as S

    needed = S._required_mpi_sonames()
    if not needed:
        pytest.skip("lammps not installed")
    assert all(n.startswith("libmpi") for n in needed)


def test_preload_refuses_to_mix_mpi_families(monkeypatch):
    """If a different MPI is already mapped, pre-loading must not happen."""
    import hvi_emp.solvers as S

    monkeypatch.setattr(S, "_required_mpi_sonames",
                        lambda: {"libmpi.so.12"})
    monkeypatch.setattr(S, "_loaded_mpi_sonames",
                        lambda: {"libmpi.so.40"})

    def explode(*a, **k):                     # must never be reached
        raise AssertionError("pre-loaded a conflicting MPI")
    monkeypatch.setattr(S.ctypes, "CDLL", explode)

    assert S._preload_mpi() is None


def test_preload_still_loads_a_matching_family(monkeypatch):
    import hvi_emp.solvers as S

    monkeypatch.setattr(S, "_required_mpi_sonames", lambda: {"libmpi.so.12"})
    monkeypatch.setattr(S, "_loaded_mpi_sonames", lambda: {"libmpi.so.12"})
    calls = []
    monkeypatch.setattr(S.ctypes, "CDLL",
                        lambda n, **k: calls.append(n) or object())
    S._preload_mpi()
    assert calls == ["libmpi.so.12"]


def test_mpi_family_names_are_human_readable():
    from hvi_emp.solvers import mpi_family
    assert mpi_family("libmpi.so.12") == "MPICH"
    assert mpi_family("libmpi.so.40") == "Open MPI"
    assert mpi_family("libmpi.so.99") == "libmpi.so.99"
