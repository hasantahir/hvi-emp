"""Tests for the GPU-capable bridges: Idefix (Stage 2) and PIConGPU (Stage 3).

Neither solver is installed here, so these test what we can actually check:
that the generated inputs are internally consistent and syntactically
plausible, that the normalisation agrees with the independent OpenMHD bridge,
that the audits fire on the setups that deserve them, and -- for the one piece
with a real dependency available -- that the openPMD probe reader round-trips
a file with PIConGPU's layout.

What these tests deliberately do NOT claim: that the .param and setup.cpp
files compile.  Only a real Idefix / PIConGPU tree can establish that, and it
is the first thing to do on the workstation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp import run_scenario
from hvi_emp.constants import MU_0
from hvi_emp.solvers.idefix_stage2 import (IdefixConfig, cavity_radius,
                                           cmake_command, fletcher_idefix_config,
                                           read_idefix_vtk,
                                           write_definitions_hpp,
                                           write_idefix_ini,
                                           write_problem_directory,
                                           write_setup_cpp)
from hvi_emp.solvers.openmhd_stage2 import OpenMHDConfig
from hvi_emp.solvers.picongpu_stage3 import (FLOAT32_EPS, PARAM_FILES,
                                             PIConGPUConfig, build_command,
                                             load_probe_timeseries,
                                             write_cfg, write_input_set)


@pytest.fixture(scope="module")
def scenario():
    return run_scenario("Fe", "Al", mass=1e-12, velocity=50e3)


# ===========================================================================
# Idefix
# ===========================================================================

def test_idefix_normalisation_agrees_with_openmhd(scenario):
    """Two independent bridges, same physical normalisation.

    Both reduce the problem to Alfven units.  They were written separately,
    so agreement is a real check on the algebra rather than a tautology --
    and if one is ever changed, this catches the divergence.
    """
    common = dict(rho_plume=1e-9, rho_ambient=2.66e-15, p_plume=1e-2,
                  p_ambient=3.2e-9, v_expansion=2.0e4, B0=3.0e-5, R0=0.01)
    a = IdefixConfig(**common).normalised()
    b = OpenMHDConfig(**common).normalised()
    for key in ("rho_plume_hat", "p_plume_hat", "p_ambient_hat",
                "v_expansion_hat", "beta_ambient", "v_alfven",
                "unit_length", "unit_time", "unit_density", "unit_B",
                "unit_pressure"):
        assert a[key] == pytest.approx(b[key], rel=1e-12), key


def test_idefix_alfven_speed_is_the_textbook_one():
    cfg = IdefixConfig(rho_plume=1e-9, rho_ambient=1e-14, p_plume=1e-2,
                       p_ambient=1e-9, v_expansion=2e4, B0=3e-5, R0=0.01)
    assert cfg.v_alfven == pytest.approx(3e-5 / np.sqrt(MU_0 * 1e-14))


def test_idefix_units_block_is_cgs():
    """Idefix's [Units] wants cm, cm/s and g/cm^3, not SI."""
    cfg = IdefixConfig(rho_plume=1e-9, rho_ambient=1e-14, p_plume=1e-2,
                       p_ambient=1e-9, v_expansion=2e4, B0=3e-5, R0=0.01)
    u = cfg.units_cgs()
    assert u["length"] == pytest.approx(cfg.R0 * 100.0)
    assert u["velocity"] == pytest.approx(cfg.v_alfven * 100.0)
    assert u["density"] == pytest.approx(cfg.rho_ambient * 1e-3)


def test_idefix_run_time_tracks_cavity_formation_not_alfven_crossing():
    """A fixed few Alfven crossings stops before anything happens.

    t_formation / t_alfven is typically ~1e2, so defaulting t_end to 3 Alfven
    crossings (as an earlier version did) ended the run two orders of
    magnitude before the cavity formed.
    """
    cfg = IdefixConfig(rho_plume=1e-9, rho_ambient=2.66e-15, p_plume=1e-2,
                       p_ambient=3.2e-9, v_expansion=2.0e4, B0=3.0e-5,
                       R0=0.01)
    t_alfven = cfg.R0 / cfg.v_alfven
    expected = 3.0 * cfg.cavity()["t_formation"] / t_alfven
    assert cfg.t_end_alfven == pytest.approx(expected)
    assert cfg.t_end_alfven > 10.0


def test_idefix_domain_holds_the_expected_cavity():
    cfg = IdefixConfig(rho_plume=1e-9, rho_ambient=2.66e-15, p_plume=1e-2,
                       p_ambient=3.2e-9, v_expansion=2.0e4, B0=3.0e-5,
                       R0=0.01)
    assert cfg.domain_R > cfg.cavity()["R_cavity"] / cfg.R0


def test_idefix_ini_grid_line_has_idefixs_five_field_syntax():
    """X1-grid <nblocks> <start> <npoints> <spacing> <end>."""
    cfg = fletcher_idefix_config("fig6_parallel")
    ini = write_idefix_ini(cfg)
    m = re.search(r"^X1-grid\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$",
                  ini, re.M)
    assert m, "no X1-grid entry in the generated ini"
    nblocks, start, npts, spacing, end = m.groups()
    assert int(nblocks) == 1
    assert float(start) == pytest.approx(cfg.r_min_frac)
    assert int(npts) == cfg.n_r
    assert spacing in ("u", "l", "s+", "s-")
    assert float(end) == pytest.approx(cfg.domain_R)


def test_idefix_spherical_uses_axis_boundary_over_the_full_polar_range():
    """Idefix's `axis` condition requires X2 to span exactly 0..pi."""
    cfg = fletcher_idefix_config("fig6_parallel")
    ini = write_idefix_ini(cfg)
    m = re.search(r"^X2-grid\s+\S+\s+(\S+)\s+\S+\s+\S+\s+(\S+)\s*$", ini,
                  re.M)
    assert float(m.group(1)) == pytest.approx(0.0)
    assert float(m.group(2)) == pytest.approx(np.pi, rel=1e-9)
    assert re.search(r"^X2-beg\s+axis", ini, re.M)
    assert re.search(r"^X2-end\s+axis", ini, re.M)


def test_idefix_definitions_match_the_geometry():
    sph = write_definitions_hpp(fletcher_idefix_config("fig6_parallel"))
    assert "GEOMETRY        SPHERICAL" in sph
    assert "DIMENSIONS      2" in sph

    cart = write_definitions_hpp(fletcher_idefix_config("fig7_perpendicular"))
    assert "GEOMETRY        CARTESIAN" in cart
    assert "DIMENSIONS      3" in cart


def test_idefix_setup_cpp_has_the_required_entry_points():
    for case in ("fig6_parallel", "fig7_perpendicular"):
        src = write_setup_cpp(fletcher_idefix_config(case))
        assert "Setup::Setup(Input &input, Grid &grid, DataBlock &data," in src
        assert "void Setup::InitFlow(DataBlock &data)" in src
        assert "DataBlockHost d(data);" in src
        assert "d.SyncToDevice();" in src
        # balanced braces is a weak but real syntax check
        assert src.count("{") == src.count("}")


def test_idefix_setup_initialises_field_by_both_routes():
    """Vector potential when available, face-centred B otherwise."""
    src = write_setup_cpp(fletcher_idefix_config("fig6_parallel"))
    assert "#ifdef EVOLVE_VECTOR_POTENTIAL" in src
    assert "AX3e" in src and "BX1s" in src and "BX2s" in src


def test_idefix_cmake_enables_mhd_and_optionally_cuda():
    cfg = fletcher_idefix_config("fig6_parallel")
    cpu = cmake_command(cfg)
    assert "-DIdefix_MHD=ON" in cpu
    assert "Kokkos_ENABLE_CUDA" not in cpu

    gpu = cmake_command(cfg, gpu_arch="AMPERE80", mpi=True)
    assert "-DKokkos_ENABLE_CUDA=ON" in gpu
    assert "-DKokkos_ARCH_AMPERE80=ON" in gpu
    assert "-DIdefix_MPI=ON" in gpu


def test_idefix_resistivity_only_appears_when_requested():
    base = dict(rho_plume=1e-9, rho_ambient=2.66e-15, p_plume=1e-2,
                p_ambient=3.2e-9, v_expansion=2.0e4, B0=3.0e-5, R0=0.01)
    ideal = write_idefix_ini(IdefixConfig(**base, eta_ohm=0.0))
    assert "resistivity" not in ideal

    resistive = write_idefix_ini(IdefixConfig(**base, eta_ohm=1.0e3))
    m = re.search(r"^resistivity\s+(\S+)\s+(\S+)\s+(\S+)", resistive, re.M)
    assert m and m.group(1) in ("explicit", "rkl")
    assert m.group(2) == "constant"
    assert float(m.group(3)) > 0


def test_idefix_rejects_bad_geometry_and_inner_radius():
    base = dict(rho_plume=1e-9, rho_ambient=1e-14, p_plume=1e-2,
                p_ambient=1e-9, v_expansion=2e4)
    with pytest.raises(ValueError, match="spherical"):
        IdefixConfig(**base, geometry="cylindrical")
    with pytest.raises(ValueError, match="r_min_frac"):
        IdefixConfig(**base, r_min_frac=0.0)


def test_idefix_epoch_selection(scenario):
    early = IdefixConfig.from_expansion(scenario.expansion, epoch="early")
    late = IdefixConfig.from_expansion(scenario.expansion, epoch="late")
    # the plume grows monotonically, so 'early' must be the smaller one
    assert early.R0 < late.R0
    with pytest.raises(ValueError, match="epoch must be"):
        IdefixConfig.from_expansion(scenario.expansion, epoch="whenever")


def test_idefix_reports_when_the_plume_never_couples_to_the_field(scenario):
    """A picogram plume in the geomagnetic field carves no cavity.

    Silently returning the last sample would look like a valid cavity setup.
    The bridge has to say that the regime does not apply.
    """
    cfg = IdefixConfig.from_expansion(scenario.expansion, epoch="auto")
    assert any("never falls to within" in n for n in cfg.notes), cfg.notes


def test_idefix_flags_spitzer_used_outside_its_validity(scenario):
    cfg = IdefixConfig.from_expansion(scenario.expansion, epoch="late",
                                      resistive=True)
    assert any("outside its range of validity" in n for n in cfg.notes)

    ideal = IdefixConfig.from_expansion(scenario.expansion, epoch="late",
                                        resistive=False)
    assert ideal.eta_ohm == 0.0
    assert not any("Spitzer" in n for n in ideal.notes)


def test_idefix_flags_a_domain_too_small_for_its_cavity():
    cfg = IdefixConfig(rho_plume=1e-9, rho_ambient=2.66e-15, p_plume=1e-2,
                       p_ambient=3.2e-9, v_expansion=2.0e4, B0=3.0e-5,
                       R0=0.01, domain_R=2.0)
    cfg._audit()
    assert any("hit the boundary" in n for n in cfg.notes)


def test_idefix_flags_a_gas_pressure_dominated_ambient():
    cfg = IdefixConfig(rho_plume=1e-9, rho_ambient=2.66e-15, p_plume=1e-2,
                       p_ambient=1.0, v_expansion=2.0e4, B0=3.0e-5, R0=0.01)
    cfg._audit()
    assert any("no well-defined diamagnetic" in n for n in cfg.notes)


def test_idefix_problem_directory_is_complete(tmp_path):
    cfg = fletcher_idefix_config("fig6_parallel")
    out = write_problem_directory(cfg, str(tmp_path / "prob"),
                                  gpu_arch="AMPERE80")
    for name in ("definitions.hpp", "setup.cpp", "idefix.ini", "README.txt"):
        p = Path(out["paths"][name])
        assert p.is_file() and p.stat().st_size > 0, name
    assert "cmake $IDEFIX_DIR" in out["cmake"]


def test_idefix_vtk_reader_says_what_to_do_when_it_cannot_find_idefix(
        monkeypatch, tmp_path):
    monkeypatch.delenv("IDEFIX_DIR", raising=False)
    with pytest.raises(RuntimeError, match="IDEFIX_DIR"):
        read_idefix_vtk(str(tmp_path / "data.0001.vtk"))


def test_cavity_radius_finds_a_synthetic_cavity():
    """|B| suppressed inside R_c, compressed in a shell, ambient outside."""
    r = np.linspace(0.2, 20.0, 400)
    R_true, R_shell = 5.0, 6.0
    B = np.ones_like(r)
    B[r < R_true] = 0.05
    shell = np.abs(r - R_shell) < 0.5
    B[shell] = 3.0
    out = cavity_radius(np.ones_like(r), B, r)
    assert out["R_cavity"] == pytest.approx(R_true, abs=0.1)
    assert out["R_shell"] == pytest.approx(R_shell, abs=0.6)
    assert out["compression"] == pytest.approx(3.0)


def test_cavity_radius_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="does not match"):
        cavity_radius(np.ones(10), np.ones(10), np.linspace(0, 1, 7))


# ===========================================================================
# PIConGPU
# ===========================================================================

def test_picongpu_electron_drift_ratio_is_the_fletcher_close_assumption():
    from hvi_emp.constants import AMU, M_ELECTRON
    cfg = PIConGPUConfig(n_e0=1e18, T_eV=1.0, m_ion=27 * AMU, Zbar=1.0)
    assert cfg.electron_drift_ratio == pytest.approx(
        np.sqrt(27 * AMU / M_ELECTRON))
    assert cfg.v_electron_drift == pytest.approx(
        cfg.v_drift * cfg.electron_drift_ratio)


def test_picongpu_timestep_respects_both_courant_and_the_plasma_period():
    cfg = PIConGPUConfig(n_e0=1e18, T_eV=1.0)
    g = cfg.grid()
    assert g["dt"] <= g["dt_courant"] * (1 + 1e-12)
    assert g["dt"] <= g["dt_plasma"] * (1 + 1e-12)
    assert g["dt"] == pytest.approx(min(g["dt_courant"], g["dt_plasma"]))


def test_picongpu_grid_is_honest_about_unresolved_debye_lengths():
    """A dense plume cannot be Debye-resolved out to a 0.3 m probe.

    The bridge must cap the grid and say so, not return an impossible one.
    """
    dense = PIConGPUConfig(n_e0=1e26, T_eV=2.0, probe_distance=0.30)
    g = dense.grid()
    assert not g["debye_resolved"]
    assert g["n_cells"] <= dense.max_cells
    assert g["cells_per_debye_actual"] < 1.0
    dense._audit()
    assert any("heat numerically" in n for n in dense.notes)

    thin = PIConGPUConfig(n_e0=1e15, T_eV=1.0, probe_distance=0.30)
    assert thin.grid()["debye_resolved"]


def test_picongpu_warns_when_the_ion_drift_vanishes_in_single_precision():
    """gamma - 1 for a 20 km/s ion is ~2e-9, below float32 epsilon.

    PIConGPU builds single-precision by default, so this drift would be
    silently lost.  The bridge must catch it and switch the build command.
    """
    cfg = PIConGPUConfig(n_e0=1e15, T_eV=1.0, v_drift=2.0e4)
    assert cfg.gamma_ion - 1.0 < FLOAT32_EPS
    cfg._audit()
    assert any("float32 epsilon" in n for n in cfg.notes)
    assert "precision64Bit" in build_command(cfg)

    fast = PIConGPUConfig(n_e0=1e15, T_eV=1.0, v_drift=3.0e7)
    assert fast.gamma_ion - 1.0 > FLOAT32_EPS
    assert "precision64Bit" not in build_command(fast)


def test_picongpu_gamma_is_the_relativistic_one():
    cfg = PIConGPUConfig(n_e0=1e15, T_eV=1.0, v_drift=1.0e8)
    from hvi_emp.constants import C_LIGHT
    beta = 1.0e8 / C_LIGHT
    assert cfg.gamma_ion == pytest.approx(1.0 / np.sqrt(1 - beta**2))


def test_picongpu_param_files_are_syntactically_plausible(scenario):
    cfg = PIConGPUConfig.from_expansion(scenario.expansion,
                                        at_frequency=916e6)
    for name, writer in PARAM_FILES.items():
        src = writer(cfg)
        assert src.startswith("/* Generated by"), name
        assert "#pragma once" in src, name
        assert src.count("{") == src.count("}"), f"unbalanced braces in {name}"
        assert src.count("(") == src.count(")"), f"unbalanced parens in {name}"
        assert "namespace picongpu" in src, name


def test_picongpu_density_param_uses_an_anisotropic_gaussian(scenario):
    """The Anisimov ellipsoid maps onto GaussianCloudImpl's per-axis sigma."""
    cfg = PIConGPUConfig.from_expansion(scenario.expansion,
                                        at_frequency=916e6,
                                        sigma=(1e-3, 5e-3, 1e-3))
    src = PARAM_FILES["density.param"](cfg)
    assert "GaussianCloudImpl<PlumeParam>" in src
    assert "gasFactor = -0.5" in src and "gasPower = 2.0" in src
    m = re.search(r"sigma_SI = float3_64\(\s*([\d.e+-]+),\s*([\d.e+-]+),"
                  r"\s*([\d.e+-]+)", src)
    assert m
    assert [float(g) for g in m.groups()] == pytest.approx([1e-3, 5e-3, 1e-3])


def test_picongpu_init_pipeline_starts_quasi_neutral(scenario):
    """Ions derived from electrons means charge separation is emergent.

    If both species were created independently from the density profile they
    would not be co-located, and the run would begin with an imposed charge
    separation that has nothing to do with the physics under study.
    """
    cfg = PIConGPUConfig.from_expansion(scenario.expansion)
    src = PARAM_FILES["speciesInitialization.param"](cfg)
    i_create = src.index("CreateDensity<densityProfiles::Plume")
    i_derive = src.index("Derive<PIC_Electrons, PIC_Ions>")
    i_drift = src.index("AssignIonDrift")
    assert i_create < i_derive < i_drift


def test_picongpu_defines_probe_species_with_field_attributes(scenario):
    cfg = PIConGPUConfig.from_expansion(scenario.expansion)
    species = PARAM_FILES["speciesDefinition.param"](cfg)
    assert "particles::pusher::Probe" in species
    assert "probeE" in species and "probeB" in species
    assert "VectorAllSpecies" in species

    out = PARAM_FILES["fileOutput.param"](cfg)
    assert "FileOutputParticles" in out and "Probes" in out


def test_picongpu_cfg_is_absorbing_not_periodic(scenario):
    """A radiated pulse must leave the box, not wrap into it."""
    cfg = PIConGPUConfig.from_expansion(scenario.expansion)
    text = write_cfg(cfg)
    assert "--periodic 0 0 0" in text
    assert "TBG_steps=" in text and "TBG_gridSize=" in text
    assert "openPMD.period" in text


def test_picongpu_2d_cfg_has_unit_depth(scenario):
    cfg = PIConGPUConfig.from_expansion(scenario.expansion, dim=2)
    assert write_cfg(cfg).count(" 1\"") >= 1     # gridSize ends "... 1"


def test_picongpu_rejects_bad_dimensions_and_state():
    with pytest.raises(ValueError, match="dim must be"):
        PIConGPUConfig(n_e0=1e18, T_eV=1.0, dim=1)
    with pytest.raises(ValueError, match="must be positive"):
        PIConGPUConfig(n_e0=0.0, T_eV=1.0)


def test_picongpu_input_set_lands_in_picongpus_directory_layout(tmp_path,
                                                                scenario):
    cfg = PIConGPUConfig.from_expansion(scenario.expansion, dim=2,
                                        max_cells=256)
    written = write_input_set(cfg, str(tmp_path / "in"), devices=(1, 1, 1))
    param_dir = tmp_path / "in" / "include" / "picongpu" / "param"
    assert (param_dir / "simulation.param").is_file()
    assert (tmp_path / "in" / "etc" / "picongpu" / "1.cfg").is_file()
    assert set(PARAM_FILES) <= set(written)


def test_picongpu_build_command_targets_the_named_gpu():
    cfg = PIConGPUConfig(n_e0=1e15, T_eV=1.0, v_drift=3e7)
    assert 'pic-build -b "cuda:80"' in build_command(cfg, gpu="A100")
    assert 'pic-build -b "cuda"' in build_command(cfg, gpu="NoSuchCard")


def test_picongpu_cold_and_warm_cases_differ_only_in_thermal_speed(scenario):
    cold = PIConGPUConfig.from_expansion(scenario.expansion,
                                         thermal_ratio=0.1)
    warm = PIConGPUConfig.from_expansion(scenario.expansion,
                                         thermal_ratio=10.0)
    assert warm.v_thermal_electron == pytest.approx(
        100.0 * cold.v_thermal_electron)
    assert warm.temperature_keV == pytest.approx(
        1e4 * cold.temperature_keV, rel=1e-9)
    assert cold.v_drift == warm.v_drift


# ---------------------------------------------------------------------------
# openPMD readback -- the one piece with a real dependency available
# ---------------------------------------------------------------------------

openpmd = pytest.importorskip("openpmd_api")


def _write_synthetic_probe_series(path: str, n_t: int = 5,
                                  n_probe: int = 4) -> dict:
    """Write a file with PIConGPU's probe layout, for the reader to parse."""
    series = openpmd.Series(path, openpmd.Access.create)
    positions = np.stack([np.linspace(0.0, 0.3, n_probe),
                          np.zeros(n_probe), np.zeros(n_probe)], axis=-1)
    E_true = np.zeros((n_t, n_probe, 3))
    B_true = np.zeros((n_t, n_probe, 3))

    for it in range(n_t):
        iteration = series.iterations[it]
        iteration.time = float(it) * 1e-12
        iteration.time_unit_SI = 1.0
        probe = iteration.particles["probe"]
        for name, arr in (("probeE", E_true), ("probeB", B_true)):
            for j, comp in enumerate("xyz"):
                vals = (np.arange(n_probe, dtype=np.float64)
                        + 10.0 * it + 100.0 * j
                        + (0.0 if name == "probeE" else 1000.0))
                arr[it, :, j] = vals
                rec = probe[name][comp]
                rec.reset_dataset(openpmd.Dataset(vals.dtype, vals.shape))
                rec.store_chunk(vals)
                rec.unit_SI = 1.0
        for j, comp in enumerate("xyz"):
            vals = np.ascontiguousarray(positions[:, j])
            rec = probe["position"][comp]
            rec.reset_dataset(openpmd.Dataset(vals.dtype, vals.shape))
            rec.store_chunk(vals)
            rec.unit_SI = 1.0
        series.flush()
    del series
    return {"positions": positions, "E": E_true, "B": B_true}


def test_probe_reader_round_trips_a_picongpu_style_file(tmp_path):
    path = str(tmp_path / "simData_%T.h5")
    truth = _write_synthetic_probe_series(path)

    all_probes = load_probe_timeseries(path)
    assert all_probes["t"].shape == (5,)
    assert all_probes["E"].shape == (4, 5, 3)
    np.testing.assert_allclose(all_probes["E"],
                               np.moveaxis(truth["E"], 0, 1))
    np.testing.assert_allclose(all_probes["B"],
                               np.moveaxis(truth["B"], 0, 1))


def test_probe_reader_selects_the_nearest_probe(tmp_path):
    path = str(tmp_path / "simData_%T.h5")
    truth = _write_synthetic_probe_series(path)

    ts = load_probe_timeseries(path, position=(0.3, 0.0, 0.0))
    assert ts["E"].shape == (5, 3)
    np.testing.assert_allclose(ts["E"], truth["E"][:, -1, :])
    assert ts["distance"] == pytest.approx(0.0, abs=1e-12)


def test_probe_reader_refuses_a_probe_outside_the_tolerance(tmp_path):
    path = str(tmp_path / "simData_%T.h5")
    _write_synthetic_probe_series(path)
    with pytest.raises(ValueError, match="tolerance"):
        load_probe_timeseries(path, position=(5.0, 0.0, 0.0),
                              tolerance=0.01)


def test_probe_reader_explains_a_missing_probe_species(tmp_path):
    path = str(tmp_path / "noprobe_%T.h5")
    series = openpmd.Series(path, openpmd.Access.create)
    it = series.iterations[0]
    rec = it.particles["e"]["position"]["x"]
    rec.reset_dataset(openpmd.Dataset(np.dtype("float64"), (2,)))
    rec.store_chunk(np.zeros(2))
    series.flush()
    del series

    with pytest.raises(KeyError, match="speciesDefinition"):
        load_probe_timeseries(path)


def test_probe_reader_accepts_a_directory(tmp_path):
    d = tmp_path / "simOutput"
    d.mkdir()
    _write_synthetic_probe_series(str(d / "simData_%T.h5"))
    ts = load_probe_timeseries(str(d))
    assert ts["t"].shape == (5,)


# ===========================================================================
# GPU detection: "no card" and "card with no driver" need different advice
# ===========================================================================

def test_gpu_states_are_distinguished(monkeypatch):
    """nvidia-smi being absent does not mean there is no GPU.

    A card present on the PCI bus with no driver loaded reports identically to
    a CPU-only machine if you only probe for nvidia-smi -- but the fix is
    completely different, and building for CUDA first wastes hours.
    """
    import hvi_emp.doctor as D

    monkeypatch.setattr(D.shutil, "which",
                        lambda n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(D, "_run",
                        lambda c, timeout=10: (True, "NVIDIA A100, 40960 MiB, 550.54"))
    g = D.gpu()
    assert g["state"] == "ready" and "A100" in g["name"]

    monkeypatch.setattr(D, "_run", lambda c, timeout=10:
                        (False, "Failed to initialize NVML: "
                                "Driver/library version mismatch"))
    g = D.gpu()
    assert g["state"] == "driver_error" and g["name"] is None
    assert "reboot" in g["detail"]

    monkeypatch.setattr(D.shutil, "which", lambda n: None)
    monkeypatch.setattr(D, "nvidia_pci_devices",
                        lambda: ["0000:65:00.0 (device 0x20b2)"])
    g = D.gpu()
    assert g["state"] == "driver_missing" and g["name"] is None
    assert "ubuntu-drivers" in g["detail"]

    monkeypatch.setattr(D, "nvidia_pci_devices", lambda: [])
    g = D.gpu()
    assert g["state"] == "absent" and g["name"] is None
    assert "CPU-only" in g["detail"]


def test_pci_probe_needs_no_driver_and_no_lspci(tmp_path):
    """The hardware probe reads sysfs, so it works with no driver loaded."""
    import hvi_emp.doctor as D

    root = tmp_path / "devices"
    for slot, vendor, cls, dev in (
            ("0000:65:00.0", "0x10de", "0x030000", "0x20b2"),  # NVIDIA GPU
            ("0000:00:02.0", "0x8086", "0x030000", "0x1234"),  # Intel iGPU
            ("0000:01:00.0", "0x10de", "0x010802", "0x5678")):  # NVIDIA, not a GPU
        d = root / slot
        d.mkdir(parents=True)
        (d / "vendor").write_text(vendor + "\n")
        (d / "class").write_text(cls + "\n")
        (d / "device").write_text(dev + "\n")

    found = D.nvidia_pci_devices(root=str(root))
    assert len(found) == 1, found        # only the NVIDIA display controller
    assert "0000:65:00.0" in found[0] and "0x20b2" in found[0]

    assert D.nvidia_pci_devices(root=str(tmp_path / "nope")) == []


def test_doctor_puts_the_driver_first_when_a_card_has_no_driver(monkeypatch,
                                                               capsys):
    """Advising a CUDA build before the driver exists is the wrong order."""
    import hvi_emp.doctor as D

    monkeypatch.setattr(D, "hardware", lambda: {
        "platform": "Linux-test", "python": "3.11.0", "machine": "x86_64",
        "cores_logical": 48, "cores_physical": 48, "memory_GB": 540.0,
        "disk_free_GB": 416.0, "gpu": None, "gpu_state": "driver_missing",
        "gpu_detail": "1 NVIDIA device(s) on the PCI bus but no nvidia-smi: "
                      "the card is there, the driver is not.\n"
                      "      Ubuntu: sudo ubuntu-drivers install"})
    D.report(skip_self_test=True)
    out = capsys.readouterr().out
    assert "PRESENT BUT NO DRIVER" in out
    i_driver = out.index("the GPU driver")
    assert "ubuntu-drivers install" in out
    # the driver must be listed before any other suggestion
    for later in ("iSALE", "WarpX", "ParaView"):
        if later in out[out.index("To go further"):]:
            assert i_driver < out.index(later, out.index("To go further"))


def _fake_nvidia_env(monkeypatch, *, modules="", dkms=None, secureboot=None,
                     kernel="6.17.0-35-generic", headers=True):
    """Pretend nvidia-smi is installed but cannot reach the driver.

    ``headers`` controls whether /lib/modules/<kernel>/build exists.  It
    defaults to present so that each test exercises the cause it is named
    for: missing headers is a blocking cause and would otherwise mask the
    others.
    """
    import io

    import hvi_emp.doctor as D

    monkeypatch.setattr(D.platform, "release", lambda: kernel)
    real_isdir = D.os.path.isdir
    monkeypatch.setattr(
        D.os.path, "isdir",
        lambda p: headers if "/lib/modules/" in p else real_isdir(p))
    present = {"nvidia-smi": "/usr/bin/nvidia-smi",
               "dkms": "/usr/bin/dkms" if dkms is not None else None,
               "mokutil": "/usr/bin/mokutil" if secureboot is not None
               else None}
    monkeypatch.setattr(D.shutil, "which", lambda n: present.get(n))

    def run(cmd, timeout=10):
        if cmd[0].endswith("dkms"):
            return True, dkms
        if cmd[0].endswith("mokutil"):
            return True, secureboot
        return False, ("NVIDIA-SMI has failed because it couldn't "
                       "communicate with the NVIDIA driver.")
    monkeypatch.setattr(D, "_run", run)

    real_open = open

    def fake_open(path, *a, **k):
        if path == "/proc/modules":
            return io.StringIO(modules)
        return real_open(path, *a, **k)
    monkeypatch.setattr("builtins.open", fake_open)
    return D


def test_driver_diagnosis_spots_a_dkms_kernel_mismatch(monkeypatch):
    """The usual cause: kernel upgraded, DKMS module not rebuilt for it.

    "Try rebooting" is wrong here -- a reboot into the same kernel changes
    nothing, because the module was never built for it.
    """
    D = _fake_nvidia_env(
        monkeypatch, kernel="6.17.0-35-generic",
        dkms="nvidia/550.120, 6.14.0-27-generic, x86_64: installed")
    g = D.gpu()
    assert g["state"] == "driver_error" and g["name"] is None
    assert "6.14.0-27-generic" in g["detail"]
    assert "6.17.0-35-generic" in g["detail"]
    assert "dkms autoinstall" in g["detail"]
    assert "ubuntu-drivers install" in g["detail"]


def test_driver_diagnosis_spots_a_leftover_userspace_binary(monkeypatch):
    D = _fake_nvidia_env(monkeypatch, dkms="")
    detail = D.gpu()["detail"]
    assert "no NVIDIA kernel module is built or loaded" in detail
    assert "ubuntu-drivers install" in detail


def test_driver_diagnosis_spots_a_built_but_unloaded_module(monkeypatch):
    D = _fake_nvidia_env(
        monkeypatch, kernel="6.17.0-35-generic",
        dkms="nvidia/580.65, 6.17.0-35-generic, x86_64: installed")
    detail = D.gpu()["detail"]
    assert "not loaded" in detail
    assert "modprobe nvidia" in detail


def test_driver_diagnosis_recommends_reboot_only_when_module_is_loaded(
        monkeypatch):
    """A reboot is the right answer for exactly one of these causes."""
    D = _fake_nvidia_env(
        monkeypatch, modules="nvidia 12345 0 - Live 0x0\n",
        kernel="6.17.0-35-generic",
        dkms="nvidia/580.65, 6.17.0-35-generic, x86_64: installed")
    detail = D.gpu()["detail"]
    assert "version mismatch" in detail
    assert "sudo reboot" in detail

    # ...and is not the headline advice when the module was never built
    D2 = _fake_nvidia_env(monkeypatch, dkms="")
    headline = D2.gpu()["detail"].split("Most likely cause:")[1]
    assert "reboot" in headline          # appears as part of the install line
    assert "version mismatch" not in headline


def test_driver_diagnosis_flags_secure_boot_as_an_additional_cause(monkeypatch):
    D = _fake_nvidia_env(monkeypatch, dkms="", secureboot="SecureBoot enabled")
    detail = D.gpu()["detail"]
    assert "secure boot         ENABLED" in detail
    assert "Secure Boot is enabled" in detail
    assert "MOK" in detail


def test_driver_diagnosis_survives_a_machine_without_dkms_or_mokutil(
        monkeypatch):
    D = _fake_nvidia_env(monkeypatch)      # dkms=None, secureboot=None
    detail = D.gpu()["detail"]
    assert "dkms                not installed" in detail
    assert "Most likely cause" in detail


# ===========================================================================
# Blackwell / sm_120: the arch tables must not stop at Hopper
# ===========================================================================

def test_blackwell_consumer_cards_map_to_sm_120():
    """RTX 50-series is sm_120, and every table here must know it.

    An earlier version topped out at Hopper (90), so a 5090 silently got a
    Hopper build that nvcc rejects.
    """
    from hvi_emp.solvers.picongpu_stage3 import CUDA_ARCH, GPU_MEMORY_GB

    for card in ("RTX5090", "RTX5080", "RTX5070Ti", "RTXPRO6000"):
        assert CUDA_ARCH[card] == "120", card
        assert card in GPU_MEMORY_GB, card
    assert CUDA_ARCH["B200"] == "100"       # datacentre Blackwell is distinct
    assert GPU_MEMORY_GB["RTX5090"] == 32


def test_picongpu_build_command_targets_blackwell():
    cfg = PIConGPUConfig(n_e0=1e15, T_eV=1.0, v_drift=3e7)
    assert 'pic-build -b "cuda:120"' in build_command(cfg, gpu="RTX5090")


def test_gpu_build_flags_identifies_an_rtx_5090_from_its_pci_id():
    """Works with a broken driver, which is the whole point."""
    import hvi_emp.doctor as D

    f = D.gpu_build_flags(device_id="0x2b85")
    assert not f["unknown"]
    assert "5090" in f["name"]
    assert f["compute_capability"] == "120"
    assert f["kokkos_arch"] == "BLACKWELL120"
    assert f["cuda_min"] == "12.8"
    assert "-DKokkos_ARCH_BLACKWELL120=ON" in f["idefix"]
    assert 'cuda:120' in f["picongpu"]
    assert "cc120" in f["openmhd"]


def test_compute_capability_dotted_form_splits_on_the_last_digit():
    """120 is 12.0, not 1.20 -- AMReX takes the dotted form."""
    import hvi_emp.doctor as D

    assert D.gpu_build_flags(device_id="0x2b85")[
        "compute_capability_dotted"] == "12.0"
    assert D.gpu_build_flags(device_id="0x2684")[
        "compute_capability_dotted"] == "8.9"
    assert "AMReX_CUDA_ARCH=12.0" in D.gpu_build_flags(
        device_id="0x2b85")["warpx"]
    assert "AMReX_CUDA_ARCH=8.9" in D.gpu_build_flags(
        device_id="0x2684")["warpx"]


def test_gpu_build_flags_admits_when_it_does_not_know_the_card():
    import hvi_emp.doctor as D

    f = D.gpu_build_flags(device_id="0xdead")
    assert f["unknown"] and "compute_cap" in f["hint"]


def test_gpu_build_flags_returns_none_with_no_nvidia_device(monkeypatch):
    import hvi_emp.doctor as D
    monkeypatch.setattr(D, "nvidia_pci_devices", lambda root=None: [])
    assert D.gpu_build_flags() is None


def test_cuda_minimum_toolkit_is_recorded_for_every_arch():
    """Targeting sm_120 with CUDA < 12.8 fails at compile time."""
    import hvi_emp.doctor as D
    for _, cc, _ in D.GPU_DEVICE_IDS.values():
        assert cc in D.CUDA_MIN_TOOLKIT, cc
    assert D.CUDA_MIN_TOOLKIT["120"] == "12.8"


def test_driver_diagnosis_blames_missing_kernel_headers(monkeypatch):
    """Hasan's actual failure: nvidia-dkms installed, no headers, no module.

    modprobe reports "Module nvidia not found in /lib/modules/<kernel>",
    dkms status is empty, and the reason is that DKMS had no headers to build
    against.  "Reboot" and "reinstall the driver" both waste time here.
    """
    D = _fake_nvidia_env(monkeypatch, kernel="6.17.0-35-generic", dkms="",
                         secureboot="SecureBoot disabled", headers=False)
    detail = D.gpu()["detail"]
    assert "kernel headers      MISSING" in detail
    headline = detail.split("Most likely cause:")[1]
    assert "linux-headers-6.17.0-35-generic" in headline
    assert "dkms autoinstall" in headline
    # Secure Boot is disabled here, so it must not be blamed
    assert "Secure Boot is enabled" not in detail


def test_memory_warning_uses_the_actual_device_size():
    """32 GB on a 5090 is a different verdict from 80 GB on an A100."""
    common = dict(n_e0=1e15, T_eV=1.0, probe_distance=0.30, dim=3,
                  max_cells=512)
    small = PIConGPUConfig(**common, gpu_memory_GB=32.0)
    big = PIConGPUConfig(**common, gpu_memory_GB=180.0)
    small._audit()
    big._audit()

    need = small.cost_estimate()["gpu_memory_GB"]
    if need > 0.8 * 32.0:
        assert any("GB device" in n for n in small.notes)
        if need <= 0.8 * 180.0:
            assert not any("GB device" in n for n in big.notes)


# ===========================================================================
# MD post-processing: crater -> inferred ionisation -> plasma
# ===========================================================================

def _synthetic_dump(path, n_side=8, hot_fraction=0.3, T_hot=120000.0,
                    n_frames=3):
    """A tungsten bcc slab with a hot blob at the top, in LAMMPS dump format."""
    from hvi_emp.constants import AMU, K_B

    a0 = 3.165                                    # W bcc, Angstrom
    cells = np.array([[i, j, k] for i in range(n_side)
                      for j in range(n_side) for k in range(max(n_side // 2, 2))])
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    pos = ((cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a0)
    n = len(pos)
    m = 183.84 * AMU
    rng = np.random.default_rng(0)

    with open(path, "w") as fh:
        for f in range(n_frames):
            v = rng.normal(0, np.sqrt(K_B * 300 / m), (n, 3)) / 1e2
            if f > 0:
                c = pos.mean(axis=0)
                c[2] = pos[:, 2].max()
                d = np.linalg.norm(pos - c, axis=1)
                hot = d < hot_fraction * d.max()
                v[hot] = rng.normal(0, np.sqrt(K_B * T_hot / m),
                                    (int(hot.sum()), 3)) / 1e2
            ke = 0.5 * m * np.sum((v * 1e2) ** 2, axis=1) / 1.602176634e-19
            fh.write(f"ITEM: TIMESTEP\n{f * 1000}\n"
                     f"ITEM: NUMBER OF ATOMS\n{n}\n"
                     "ITEM: BOX BOUNDS pp pp mm\n")
            for d3 in range(3):
                fh.write(f"{pos[:, d3].min():.4f} {pos[:, d3].max():.4f}\n")
            fh.write("ITEM: ATOMS id type x y z vx vy vz c_ke c_pe c_coord\n")
            for i in range(n):
                fh.write(f"{i + 1} 1 {pos[i, 0]:.4f} {pos[i, 1]:.4f} "
                         f"{pos[i, 2]:.4f} {v[i, 0]:.6f} {v[i, 1]:.6f} "
                         f"{v[i, 2]:.6f} {ke[i]:.6e} -8.9 8.0\n")
    return path, n


def test_lammps_dump_reader_round_trips(tmp_path):
    from hvi_emp.solvers.lammps_postprocess import read_lammps_dump

    path, n = _synthetic_dump(str(tmp_path / "impact.dump"))
    frames = read_lammps_dump(path)
    assert len(frames) == 3
    assert frames[0].n_atoms == n
    assert frames[0].columns[:5] == ("id", "type", "x", "y", "z")
    assert frames[1].timestep == 1000
    assert frames[0].positions().shape == (n, 3)
    with pytest.raises(KeyError, match="not in this dump"):
        frames[0].col("c_stress")


def test_temperature_is_dispersion_not_kinetic_energy(tmp_path):
    """A cold lump moving fast is not hot, and must not appear ionised.

    Using per-atom kinetic energy as a temperature is the classic way to
    fabricate plasma that is not there: the ejecta are fast but cold.
    """
    from hvi_emp.solvers.lammps_postprocess import (analyse_frame,
                                                    read_lammps_dump)

    path, _ = _synthetic_dump(str(tmp_path / "cold.dump"), n_frames=1)
    frames = read_lammps_dump(path)
    fr = frames[0]

    # give every atom a huge *uniform* drift: enormous KE, zero temperature
    for c in ("vx", "vy", "vz"):
        fr.data[:, fr.columns.index(c)] += 1.0e5      # A/ps = 10 km/s

    st = analyse_frame(fr, material="W", cell=8e-10)
    assert st.T_eV.max() < 0.1, (
        f"a uniformly drifting cold lattice registered {st.T_eV.max():.3f} eV")
    assert st.Zbar.max() < 1e-3
    assert st.atom_speed.max() > 1e6            # it really is moving fast


def test_ionisation_appears_only_where_the_material_is_hot(tmp_path):
    from hvi_emp.solvers.lammps_postprocess import (analyse_frame,
                                                    read_lammps_dump)

    path, _ = _synthetic_dump(str(tmp_path / "hot.dump"))
    frames = read_lammps_dump(path)

    cold = analyse_frame(frames[0], material="W", cell=8e-10)
    hot = analyse_frame(frames[-1], material="W", cell=8e-10)

    assert cold.Zbar.max() < 0.01, "pristine 300 K lattice must not ionise"
    assert hot.Zbar.max() > 0.5, "the hot blob should be substantially ionised"
    assert hot.n_e.max() > hot.Zbar.max() * 1e27
    # and the ionisation must be localised, not smeared over the whole slab
    assert (hot.Zbar > 0.5 * hot.Zbar.max()).mean() < 0.5


def test_postprocessor_states_that_ionisation_is_inferred(tmp_path):
    """Classical MD has no electrons. The output must say so, every time."""
    from hvi_emp.solvers.lammps_postprocess import (analyse_frame,
                                                    read_lammps_dump)

    path, _ = _synthetic_dump(str(tmp_path / "x.dump"))
    st = analyse_frame(read_lammps_dump(path)[-1], material="W", cell=8e-10)
    assert any("INFERRED, NOT SIMULATED" in n for n in st.notes)


def test_convert_dump_writes_both_representations(tmp_path):
    from hvi_emp.solvers.lammps_postprocess import convert_dump

    path, n = _synthetic_dump(str(tmp_path / "impact.dump"))
    out = convert_dump(path, str(tmp_path / "out"), material="W",
                       cell=8e-10, verbose=False)
    assert out["frames"] == 3
    assert len(out["atoms"]) == 3 and len(out["grid"]) == 3
    assert Path(out["atoms_pvd"]).is_file()
    assert Path(out["grid_pvd"]).is_file()
    # the .pvd must carry the caveat to whoever opens it in ParaView
    assert "INFERRED, NOT SIMULATED" in Path(out["atoms_pvd"]).read_text()
    # ionisation should rise across the series
    z = [p["Zbar_max"] for p in out["peaks"]]
    assert z[-1] > z[0]


def test_vtp_is_valid_polydata(tmp_path):
    """Independent parse: one vertex cell per atom, scalars intact."""
    import re
    import struct
    import zlib

    from hvi_emp.viz.vti import write_vtp

    n = 50
    pts = np.random.default_rng(1).normal(size=(n, 3))
    vals = np.arange(n, dtype=np.float32)
    p = str(tmp_path / "a.vtp")
    write_vtp(p, pts, {"Zbar": vals})

    raw = Path(p).read_bytes()
    head = raw[:raw.index(b"<AppendedData")].decode("ascii")
    data = raw[raw.index(b"_", raw.index(b"<AppendedData")) + 1:]
    assert f'NumberOfPoints="{n}"' in head
    assert f'NumberOfVerts="{n}"' in head

    NPT = {"Float32": np.float32, "Int32": np.int32}
    arrays = {}
    for typ, name, nc, off in re.findall(
            r'<DataArray type="(\w+)" Name="([^"]+)" '
            r'NumberOfComponents="(\d+)" format="appended" offset="(\d+)"/>',
            head):
        off, nc = int(off), int(nc)
        nb = struct.unpack_from("<QQQ", data, off)[0]
        sizes = struct.unpack_from(f"<{nb}Q", data, off + 24)
        pos, buf = off + 24 + 8 * nb, b""
        for cs in sizes:
            buf += zlib.decompress(data[pos:pos + cs])
            pos += cs
        a = np.frombuffer(buf, dtype=NPT[typ])
        arrays[name] = a.reshape(-1, nc) if nc > 1 else a

    np.testing.assert_allclose(arrays["Points"], pts.astype(np.float32))
    np.testing.assert_allclose(arrays["Zbar"], vals)
    assert arrays["connectivity"].tolist() == list(range(n))
    assert arrays["offsets"].tolist() == list(range(1, n + 1))


def test_vtp_rejects_mismatched_scalars(tmp_path):
    from hvi_emp.viz.vti import write_vtp

    with pytest.raises(ValueError, match=r"points must be"):
        write_vtp(str(tmp_path / "b.vtp"), np.zeros((5, 2)))
    with pytest.raises(ValueError, match="values for"):
        write_vtp(str(tmp_path / "c.vtp"), np.zeros((5, 3)),
                  {"bad": np.zeros(4)})


def test_lammps_deck_dumps_what_the_postprocessor_needs():
    """Positions alone give a pretty picture and nothing else."""
    from hvi_emp.solvers.lammps_stage1 import (LammpsImpactConfig,
                                               generate_input_deck)

    deck = generate_input_deck(LammpsImpactConfig(material="W", velocity=9e3,
                                                  dump_every=100))
    dump = [ln for ln in deck.splitlines() if ln.startswith("dump ")][0]
    for col in ("id", "type", "x", "y", "z", "vx", "vy", "vz",
                "c_ke", "c_pe", "c_coord"):
        assert col in dump.split(), f"{col} missing from the dump"
    assert "compute         coord all coord/atom cutoff" in deck
    assert "dump_modify     d1 sort id" in deck
