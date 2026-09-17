"""Tests for the configuration, schema, cluster and workflow layer.

The point of this layer is that a bad configuration fails immediately and
loudly instead of three hours later and quietly. So most of these tests
assert that something is **rejected**, and that the message says why.

Two rules that get their own tests because breaking them is silent:

* the YAML material library must reproduce the previous hardcoded values
  exactly -- moving data into config is only safe if the data did not move;
* the pydantic and no-pydantic paths must agree on what is valid, because a
  file accepted on a laptop and rejected on a cluster is worse than either.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp import get_material, list_materials                  # noqa: E402
from hvi_emp.cluster import (ClusterSpec, LocalScheduler,         # noqa: E402
                             SlurmScheduler, estimate_memory_mb,
                             estimate_walltime, get_scheduler,
                             write_job_script)
from hvi_emp.config import (CONF_DIR, apply_overrides,            # noqa: E402
                            compose_config, config_to_yaml, load_config,
                            to_run_spec)
from hvi_emp.materials import (load_material_file,                # noqa: E402
                               load_material_library)
from hvi_emp.schema import (ConfigError, MATERIAL_BOUNDS,         # noqa: E402
                            SweepSpec, USING_PYDANTIC,
                            _check_material_fields, require_number,
                            validate_material, validate_run,
                            validate_scenario)

warnings.simplefilter("ignore")
EV = 1.602176634e-19


# ---------------------------------------------------------------------------
# The materials moved to YAML -- they must not have changed value
# ---------------------------------------------------------------------------

#: Verbatim from the pre-YAML `Material(...)` literals in materials.py.
LEGACY = {
    "Al": dict(rho0=2700.0, c0=5386.0, s=1.339, gamma0=2.14, cv_solid=897.0,
               T_melt=933.5, T_vap=2792.0, L_fusion=3.97e5, L_vap=1.05e7,
               E_cohesive_eV=3.39, A=26.9815, Z=13, work_function_eV=4.08,
               us_up_valid_to=2.0e4, E_ion_eV=(5.9858, 18.8286, 28.4477),
               g_ion=(6.0, 1.0, 2.0, 1.0)),
    "Fe": dict(rho0=7870.0, c0=3955.0, s=1.580, gamma0=1.69, cv_solid=449.0,
               T_melt=1811.0, T_vap=3134.0, L_fusion=2.47e5, L_vap=6.09e6,
               E_cohesive_eV=4.28, A=55.845, Z=26, work_function_eV=4.67,
               us_up_valid_to=1.5e4, E_ion_eV=(7.9025, 16.1992, 30.651),
               g_ion=(25.0, 30.0, 25.0, 6.0)),
    "W": dict(rho0=19240.0, c0=4029.0, s=1.237, gamma0=1.54, cv_solid=132.0,
              T_melt=3695.0, T_vap=5828.0, L_fusion=1.91e5, L_vap=4.48e6,
              E_cohesive_eV=8.90, A=183.84, Z=74, work_function_eV=4.55,
              us_up_valid_to=1.5e4, E_ion_eV=(7.8640, 16.37, 26.0),
              g_ion=(5.0, 6.0, 9.0, 10.0)),
}


@pytest.mark.parametrize("name", sorted(LEGACY))
def test_yaml_materials_match_the_previous_hardcoded_values(name):
    """Moving data into config must not move the data."""
    m = get_material(name)
    ref = LEGACY[name]
    for key, want in ref.items():
        if key.endswith("_eV"):
            got = getattr(m, key[:-3])
            got = tuple(x / EV for x in got) if isinstance(got, tuple) \
                else got / EV
        else:
            got = getattr(m, key)
        assert got == pytest.approx(want, rel=1e-12), f"{name}.{key}"


def test_all_eight_materials_still_present():
    assert list_materials() == ["Al", "Cu", "Dolomite", "Fe", "Kapton",
                                "Olivine", "SiO2", "W"]


def test_every_shipped_material_file_validates():
    """The library is the config tree; it must load cleanly from disk."""
    names = load_material_library(CONF_DIR / "materials")
    assert len(names) == 8


# ---------------------------------------------------------------------------
# Schema: what must be rejected
# ---------------------------------------------------------------------------

def _valid_material() -> dict:
    return dict(name="Test", rho0=2700.0, c0=5386.0, s=1.339, gamma0=2.14,
                cv_solid=897.0, T_melt=933.5, T_vap=2792.0, L_fusion=3.97e5,
                L_vap=1.05e7, E_cohesive_eV=3.39, A=26.9815, Z=13,
                work_function_eV=4.08, us_up_valid_to=2.0e4,
                E_ion_eV=[5.9858, 18.8286, 28.4477], g_ion=[6.0, 1.0, 2.0, 1.0])


def test_valid_material_round_trips():
    spec = validate_material(_valid_material())
    assert spec.rho0 == 2700.0
    assert spec.E_ion[0] == pytest.approx(5.9858 * EV)


def test_g_ion_off_by_one_is_rejected():
    """The classic Saha bug: one statistical weight short.

    Silently drops the last ion stage from the partition function.
    """
    d = _valid_material()
    d["g_ion"] = [6.0, 1.0, 2.0]                # one short
    with pytest.raises(ConfigError, match="len\\(E_ion\\)\\+1"):
        validate_material(d)


def test_non_monotonic_ionisation_ladder_is_rejected():
    d = _valid_material()
    d["E_ion_eV"] = [5.9858, 4.0, 28.4477]      # second stage cheaper
    with pytest.raises(ConfigError, match="not greater than"):
        validate_material(d)


def test_melt_above_boil_is_rejected():
    d = _valid_material()
    d["T_melt"], d["T_vap"] = 3000.0, 2792.0
    with pytest.raises(ConfigError, match="below T_vap"):
        validate_material(d)


def test_density_in_g_per_cm3_is_rejected():
    """2.7 instead of 2700: the single most likely material typo."""
    d = _valid_material()
    d["rho0"] = 2.7
    with pytest.raises(ConfigError, match="physical range"):
        validate_material(d)


def test_cohesive_energy_inconsistent_with_latent_heat_is_rejected():
    d = _valid_material()
    d["E_cohesive_eV"] = 40.0                   # 10x too large
    with pytest.raises(ConfigError, match="latent heat"):
        validate_material(d)


def test_molecular_bonding_relaxes_the_cohesive_energy_check():
    """Kapton is the real case: a polymer depolymerises, it does not
    vaporise to atoms, so the two energies legitimately differ ~10x."""
    d = _valid_material()
    d["E_cohesive_eV"] = 40.0
    d["bonding"] = "molecular"
    assert validate_material(d).E_cohesive == pytest.approx(40.0 * EV)
    d["bonding"] = "metallic"                   # not a known classification
    with pytest.raises(ConfigError, match="atomic"):
        validate_material(d)


def test_unknown_material_key_is_rejected_not_ignored():
    """A typo'd key must not be silently dropped, leaving the default."""
    d = _valid_material()
    d["rho_0"] = 2700.0
    with pytest.raises(ConfigError, match="unknown key"):
        validate_material(d)


def test_missing_material_key_is_reported():
    d = _valid_material()
    del d["gamma0"]
    with pytest.raises(ConfigError, match="missing required key"):
        validate_material(d)


@pytest.mark.parametrize("text", ["50.0e3", "5e+06", "1e-12"])
def test_yaml_string_floats_are_rejected(text):
    """YAML 1.1 needs a point and a signed exponent, so these parse as text.

    Three shipped material files had this and nothing complained, because
    pydantic coerces silently while the fallback path does not -- the two
    backends disagreed about the same file.
    """
    with pytest.raises(ConfigError, match="string, not a number"):
        require_number(text, "test field")


def test_material_rules_are_shared_by_both_backends():
    """`_check_material_fields` is the single rule set.

    If pydantic and the fallback ever validated separately they would drift,
    and a file valid on one machine would fail on another.
    """
    d = validate_material(_valid_material()).model_dump()
    assert _check_material_fields(dict(d)) is not None
    d["T_melt"] = 9000.0
    with pytest.raises(ConfigError):
        _check_material_fields(d)


def test_material_bounds_all_carry_a_reason():
    for key, (lo, hi, unit, why) in MATERIAL_BOUNDS.items():
        assert lo < hi, key
        assert unit and why, f"{key} has no unit or no stated reason"


# ---------------------------------------------------------------------------
# Scenario schema
# ---------------------------------------------------------------------------

def test_velocity_in_km_per_second_is_rejected():
    """`velocity: 50` meaning 50 km/s is the unit error this layer exists
    to catch: 50 m/s runs happily and produces no plasma."""
    with pytest.raises(ConfigError, match="km/s"):
        validate_scenario({"velocity": 50.0})


def test_grazing_angle_is_rejected():
    with pytest.raises(ConfigError, match="grazing"):
        validate_scenario({"angle_deg": 90.0})


def test_unknown_expansion_model_is_rejected():
    with pytest.raises(ConfigError, match="1T"):
        validate_scenario({"expansion_model": "3T"})


def test_valid_scenario_accepts_both_models():
    for m in ("1T", "2T"):
        assert validate_scenario({"expansion_model": m}).expansion_model == m


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def test_sweep_linear_and_log():
    lin = SweepSpec(parameter="velocity", start=40e3, stop=60e3, num=3)
    assert lin.resolve() == pytest.approx([40e3, 50e3, 60e3])
    log = SweepSpec(parameter="mass", start=1e-15, stop=1e-9, num=3, log=True)
    assert log.resolve() == pytest.approx([1e-15, 1e-12, 1e-9])


def test_sweep_explicit_values_win():
    s = SweepSpec(parameter="velocity", values=(1e4, 2e4))
    assert s.resolve() == [1e4, 2e4]


def test_underspecified_sweep_is_rejected():
    with pytest.raises(ConfigError, match="start"):
        SweepSpec(parameter="velocity", start=40e3).resolve()


def test_log_sweep_through_zero_is_rejected():
    with pytest.raises(ConfigError, match="positive"):
        SweepSpec(parameter="mass", start=0.0, stop=1.0, num=3,
                  log=True).resolve()


# ---------------------------------------------------------------------------
# Config composition
# ---------------------------------------------------------------------------

def test_default_config_composes_and_validates():
    cfg = load_config()
    assert cfg["scenario"]["projectile"] == "Fe"
    assert isinstance(cfg["scenario"]["velocity"], float)
    assert to_run_spec(cfg).sweep is None


def test_group_selection_and_field_override():
    cfg = load_config(["scenario=fletcher_w_al", "sweep=velocity",
                       "cluster=slurm_array", "scenario.velocity=62e3"])
    assert cfg["scenario"]["projectile"] == "W"
    assert cfg["scenario"]["expansion_model"] == "2T"   # from the group
    assert cfg["scenario"]["velocity"] == 62e3          # from the override
    assert cfg["cluster"]["backend"] == "slurm"
    assert len(to_run_spec(cfg).sweep.resolve()) == 14


def test_group_defaults_are_inherited():
    """fletcher_w_al inherits from default, so it gets every base field."""
    cfg = load_config(["scenario=fletcher_w_al"])
    assert "bands" in cfg["scenario"] and "closure" in cfg["scenario"]


def test_typo_in_an_override_key_is_rejected_with_a_suggestion():
    with pytest.raises(ConfigError, match="velocity"):
        load_config(["scenario.veloctiy=62e3"])


def test_override_without_equals_is_rejected():
    with pytest.raises(ConfigError, match="key=value"):
        apply_overrides({}, ["nonsense"])


def test_unknown_config_group_lists_what_exists():
    with pytest.raises(ConfigError, match="Available"):
        load_config(["scenario=does_not_exist"])


def test_config_serialises_round_trip():
    import yaml
    cfg = load_config(["sweep=velocity"])
    assert yaml.safe_load(config_to_yaml(cfg))["sweep"]["num"] == 14


def test_compose_without_validation_still_parses():
    cfg = compose_config(["scenario.velocity=50"])   # invalid physics
    assert cfg["scenario"]["velocity"] == 50
    with pytest.raises(ConfigError):
        to_run_spec(cfg)


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------

def test_walltime_scales_with_work_and_model():
    fast = estimate_walltime(10, "1T", workers=4)
    slow = estimate_walltime(10, "2T", workers=4)
    assert slow > fast
    assert estimate_walltime(100, "1T", 4) > estimate_walltime(10, "1T", 4)
    assert all(len(t.split(":")) == 3 for t in (fast, slow))


def test_memory_scales_with_workers():
    """Workers are processes with their own table cache, so memory scales."""
    assert estimate_memory_mb(16) > estimate_memory_mb(1)
    assert estimate_memory_mb(8, "2T") > estimate_memory_mb(8, "1T")


def test_invalid_walltime_is_rejected():
    with pytest.raises(ConfigError, match="walltime"):
        ClusterSpec(time="2 hours")
    for good in ("01:00:00", "1-00:00:00", "30:00", "90"):
        ClusterSpec(time=good)


def test_unknown_backend_names_the_extension_point():
    with pytest.raises(ConfigError, match="Scheduler"):
        ClusterSpec(backend="pbs")


def test_slurm_array_directive_respects_throttle():
    s = SlurmScheduler(ClusterSpec(backend="slurm", array_throttle=8))
    assert s.array_directive(14) == "--array=0-13%8"
    s2 = SlurmScheduler(ClusterSpec(backend="slurm"))
    assert s2.array_directive(14) == "--array=0-13"


def test_job_script_pins_thread_pools_and_pool_size(tmp_path):
    """The oversubscription guard is the whole point of the script.

    Without OMP_NUM_THREADS=1 a 16-CPU job runs 16 processes x 16 BLAS
    threads on 16 cores. Without HVI_EMP_WORKERS the pool sizes itself to
    the node, not the allocation.
    """
    run = validate_run({"scenario": {}, "sweep": None,
                        "cluster": {"backend": "slurm", "cpus_per_task": 16}})
    text = write_job_script(run, tmp_path / "job.sh")
    assert "export OMP_NUM_THREADS=1" in text
    assert "export MKL_NUM_THREADS=1" in text
    assert "export HVI_EMP_WORKERS=16" in text
    assert "--cpus-per-task=16" in text
    assert "set -euo pipefail" in text
    assert (tmp_path / "job.sh").stat().st_mode & 0o111


def test_job_script_becomes_an_array_when_there_is_a_sweep(tmp_path):
    run = validate_run({
        "scenario": {}, "cluster": {"backend": "slurm"},
        "sweep": {"parameter": "velocity", "start": 40e3, "stop": 66e3,
                  "num": 14}})
    text = write_job_script(run, tmp_path / "job.sh")
    assert "--array=0-13" in text
    assert "--sweep-index" in text
    assert "${SLURM_ARRAY_TASK_ID}" in text


def test_single_scenario_script_has_no_array(tmp_path):
    run = validate_run({"scenario": {}, "cluster": {"backend": "slurm"}})
    text = write_job_script(run, tmp_path / "job.sh")
    assert "--array" not in text and "--sweep-index" not in text


def test_gpu_request_only_appears_when_asked(tmp_path):
    plain = validate_run({"scenario": {}, "cluster": {"backend": "slurm"}})
    assert "--gres" not in write_job_script(plain, tmp_path / "a.sh")
    gpu = validate_run({"scenario": {},
                        "cluster": {"backend": "slurm", "gpus": 2,
                                    "gpu_type": "a100"}})
    assert "--gres=gpu:a100:2" in write_job_script(gpu, tmp_path / "b.sh")


def test_scheduler_lookup():
    assert isinstance(get_scheduler(ClusterSpec(backend="slurm")),
                      SlurmScheduler)
    assert isinstance(get_scheduler(ClusterSpec(backend="local")),
                      LocalScheduler)


def test_cpu_count_honours_the_allocation(monkeypatch):
    """A job allocated 4 CPUs must not fork a pool sized to the node."""
    from hvi_emp import parallel
    monkeypatch.setenv("HVI_EMP_WORKERS", "4")
    assert parallel.cpu_count() == 4
    monkeypatch.delenv("HVI_EMP_WORKERS")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    assert parallel.cpu_count() == 6
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    assert parallel.cpu_count() >= 1          # malformed: fall through


# ---------------------------------------------------------------------------
# The run entry point -- the command both sbatch and Snakemake call
# ---------------------------------------------------------------------------

def _run_cli(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "hvi_emp.run", *args],
        capture_output=True, text=True, cwd=str(cwd), timeout=300)


@pytest.fixture(scope="module")
def repo_root():
    return Path(__file__).resolve().parents[1]


def test_print_config_resolves_without_running(repo_root):
    p = _run_cli("--print-config", "sweep=velocity", cwd=repo_root)
    assert p.returncode == 0, p.stderr
    assert "resolves to 14 scenario(s)" in p.stdout


def test_bad_config_exits_two_with_a_reason(repo_root):
    p = _run_cli("--print-config", "scenario.velocity=50", cwd=repo_root)
    assert p.returncode == 2
    assert "km/s" in p.stderr
    assert "Traceback" not in p.stderr, "should be a message, not a crash"


def test_sweep_index_out_of_range_exits_two(repo_root, tmp_path):
    p = _run_cli("sweep=velocity", "--sweep-index", "99",
                 "--output-dir", str(tmp_path), cwd=repo_root)
    assert p.returncode == 2
    assert "outside 0..13" in p.stderr


def test_sweep_index_without_a_sweep_is_rejected(repo_root, tmp_path):
    p = _run_cli("--sweep-index", "0", "--output-dir", str(tmp_path),
                 cwd=repo_root)
    assert p.returncode == 2
    assert "no sweep is configured" in p.stderr


def test_one_sweep_point_writes_a_traceable_result(repo_root, tmp_path):
    """Every result must carry the config and provenance that produced it."""
    p = _run_cli("sweep=velocity", "--sweep-index", "0", "tag=t",
                 "--output-dir", str(tmp_path), cwd=repo_root)
    assert p.returncode == 0, p.stderr
    out = tmp_path / "t" / "point_0000"
    assert (out / "config.yaml").is_file()
    result = json.loads((out / "result.json").read_text())
    assert result["velocity_m_s"] == pytest.approx(40e3)
    assert result["Q_frozen_C"] > 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["versions"]["hvi_emp"]
    assert manifest["n_results"] == 1


def test_submit_is_dry_by_default(repo_root, tmp_path):
    """A function that queues a job by accident is worse than one that asks."""
    p = _run_cli("--submit", "sweep=velocity", "cluster=slurm_array",
                 "--output-dir", str(tmp_path),
                 "--script", str(tmp_path / "s.sh"), cwd=repo_root)
    assert p.returncode == 0, p.stderr
    assert "nothing submitted" in p.stdout
    assert (tmp_path / "s.sh").is_file()
    assert "--array=0-13%16" in (tmp_path / "s.sh").read_text()


# ---------------------------------------------------------------------------
# Snakemake
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Path(__file__).resolve().parents[1]
                    .joinpath("workflow/Snakefile").is_file(),
                    reason="no Snakefile")
def test_snakefile_and_profile_are_present_and_consistent(repo_root):
    """The DAG's resources must exist for every rule the profile names."""
    import yaml
    snake = (repo_root / "workflow" / "Snakefile").read_text()
    rules = {ln.split()[1].rstrip(":") for ln in snake.splitlines()
             if ln.startswith("rule ")}
    assert {"all", "validate_config", "scenario_point", "aggregate",
            "provenance"} <= rules
    profile = yaml.safe_load(
        (repo_root / "workflow/profiles/slurm/config.yaml").read_text())
    for named in profile.get("set-resources", {}):
        assert named in rules, f"profile sets resources for unknown rule "\
                               f"{named!r}"
    for named in profile.get("set-threads", {}):
        assert named in rules
