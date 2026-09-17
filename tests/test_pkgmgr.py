"""Tests for the package-manager detector.

The behaviour worth pinning is not "prefers mamba" -- that is one line. It
is the conditional advice: a current conda must NOT be told to install
mamba, an old conda MUST be told about the solver flag, and a machine with
no conda at all must not be handed a conda command.
"""

import subprocess

import pytest

import hvi_emp.pkgmgr as P


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def only(monkeypatch, *present):
    """Pretend exactly ``present`` are on PATH."""
    monkeypatch.setattr(
        P.shutil, "which",
        lambda n: f"/usr/bin/{n}" if n in present else None)


def conda_says(monkeypatch, version_text):
    """Pretend ``conda --version`` prints ``version_text``."""
    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(argv, 0, version_text, "")
    monkeypatch.setattr(P.subprocess, "run", fake_run)


# ---------------------------------------------------------------------------
# preference order
# ---------------------------------------------------------------------------

def test_preference_order_is_fastest_first():
    assert P.PREFERENCE == ("micromamba", "mamba", "conda")


@pytest.mark.parametrize("present,expected", [
    (("micromamba", "mamba", "conda"), "micromamba"),
    (("mamba", "conda"), "mamba"),
    (("conda",), "conda"),
])
def test_detect_picks_the_fastest_available(monkeypatch, present, expected):
    only(monkeypatch, *present)
    conda_says(monkeypatch, "conda 24.9.2")
    assert P.detect().name == expected


def test_no_manager_is_reported_not_guessed(monkeypatch):
    only(monkeypatch)
    pm = P.detect()
    assert pm.name is None
    assert pm.install_argv("lammps") == []
    assert pm.install("lammps") == ""
    # It must point somewhere real rather than naming a missing command.
    assert "miniforge" in pm.speedup_hint().lower()


def test_install_hint_falls_back_to_pip_without_a_manager(monkeypatch):
    only(monkeypatch)
    out = P.install_hint("lammps", pip_fallback="pip install lammps mpich")
    assert "pip install lammps mpich" in out
    assert "conda" not in out
    assert "mamba" not in out


# ---------------------------------------------------------------------------
# the conda nuance -- the point of the module
# ---------------------------------------------------------------------------

def test_modern_conda_is_not_told_to_install_mamba(monkeypatch):
    """conda >= 23.10 already defaults to libmamba. Nagging it is wrong."""
    only(monkeypatch, "conda")
    conda_says(monkeypatch, "conda 24.9.2")
    pm = P.detect()
    assert pm.solver == "libmamba"
    assert pm.fast is True
    assert pm.speedup_hint() is None
    assert "libmamba" in pm.why()


def test_conda_with_the_solver_key_is_told_to_opt_in(monkeypatch):
    """22.11 <= v < 23.10: the key exists, so offer the cheap fix."""
    only(monkeypatch, "conda")
    conda_says(monkeypatch, "conda 23.5.0")
    pm = P.detect()
    assert pm.solver == "classic"
    assert pm.fast is False
    tip = pm.speedup_hint()
    assert "conda config --set solver libmamba" in tip


def test_conda_4_10_is_not_told_to_set_a_key_it_lacks(monkeypatch):
    """Regression: conda 4.10.3 has no `solver` config key.

    Reported from a real workstation. The advice used to be
    `conda config --set solver libmamba`, which on this version fails with
    "CondaValueError: Key 'solver' is not a known primitive parameter."
    The key arrived in conda 22.11 (conda-libmamba-solver 22.12.0), and
    current plugin releases require conda >= 23.3, so there is no route to
    libmamba through a conda this old.
    """
    only(monkeypatch, "conda")
    conda_says(monkeypatch, "conda 4.10.3")
    pm = P.detect()

    assert pm.version == (4, 10, 3)
    assert pm.solver == "classic"
    assert pm.fast is False

    tip = pm.speedup_hint()
    assert "conda config --set solver" not in tip
    assert "conda-libmamba-solver" not in tip
    # It must point at the route that actually works.
    assert "--bootstrap-mamba" in tip
    # And say why, so the reader is not left guessing.
    assert "22.11" in tip

    assert "too old to switch" in pm.why()


@pytest.mark.parametrize("version,expect_solver_key", [
    ("conda 4.10.3", False),
    ("conda 22.9.0", False),
    ("conda 22.11.0", True),
    ("conda 23.5.0", True),
])
def test_solver_key_threshold(monkeypatch, version, expect_solver_key):
    """The `solver` key exists from 22.11 on -- and not before."""
    only(monkeypatch, "conda")
    conda_says(monkeypatch, version)
    tip = P.detect().speedup_hint()
    assert ("conda config --set solver libmamba" in tip) is expect_solver_key


def test_the_boundary_release_counts_as_fast(monkeypatch):
    only(monkeypatch, "conda")
    conda_says(monkeypatch, "conda 23.10.0")
    assert P.detect().solver == "libmamba"
    conda_says(monkeypatch, "conda 23.9.0")
    assert P.detect().solver == "classic"


def test_explicit_classic_override_beats_a_new_version(monkeypatch):
    """A pinned classic solver is slow however new conda is."""
    only(monkeypatch, "conda")
    conda_says(monkeypatch, "conda 24.9.2")
    pm = P.detect(environ={"CONDA_SOLVER": "classic"})
    assert pm.solver == "classic"
    assert "pinned to the classic resolver" in pm.speedup_hint()


def test_unprobed_conda_does_not_claim_to_know(monkeypatch):
    """probe=False means we never asked. Silence beats a guess."""
    only(monkeypatch, "conda")
    pm = P.detect(probe=False, environ={})
    assert pm.version is None
    assert pm.solver is None
    assert pm.speedup_hint() is None


def test_unparseable_conda_version_is_not_called_old(monkeypatch):
    only(monkeypatch, "conda")
    conda_says(monkeypatch, "conda: command mangled")
    pm = P.detect()
    assert pm.version is None
    assert pm.solver is None


def test_conda_that_will_not_run_is_survivable(monkeypatch):
    only(monkeypatch, "conda")

    def boom(*a, **k):
        raise OSError("exec format error")
    monkeypatch.setattr(P.subprocess, "run", boom)
    pm = P.detect()
    assert pm.name == "conda"
    assert pm.version is None


def test_version_probe_timeout_is_not_fatal(monkeypatch):
    only(monkeypatch, "conda")

    def slow(*a, **k):
        raise subprocess.TimeoutExpired("conda", 15)
    monkeypatch.setattr(P.subprocess, "run", slow)
    assert P.detect().version is None


# ---------------------------------------------------------------------------
# command construction
# ---------------------------------------------------------------------------

def test_micromamba_gets_an_explicit_prefix(monkeypatch):
    """micromamba reads MAMBA_ROOT_PREFIX, not CONDA_PREFIX.

    Inside an activated conda env a bare `micromamba install` resolves
    against the wrong target and only fails later, at import time.
    """
    only(monkeypatch, "micromamba")
    pm = P.detect(environ={"CONDA_PREFIX": "/opt/envs/hvi"})
    argv = pm.install_argv("lammps")
    assert argv[:4] == ["micromamba", "install", "-p", "/opt/envs/hvi"]
    assert argv[-1] == "lammps"
    assert "-c" in argv and "conda-forge" in argv


def test_mamba_and_conda_do_not_get_a_prefix_flag(monkeypatch):
    """They already honour CONDA_PREFIX; -p would be redundant noise."""
    for name in ("mamba", "conda"):
        only(monkeypatch, name)
        conda_says(monkeypatch, "conda 24.9.2")
        pm = P.detect(environ={"CONDA_PREFIX": "/opt/envs/hvi"})
        assert "-p" not in pm.install_argv("lammps")


def test_micromamba_without_an_active_env_omits_the_prefix(monkeypatch):
    only(monkeypatch, "micromamba")
    pm = P.detect(environ={})
    assert "-p" not in pm.install_argv("lammps")


def test_multiple_packages_in_one_solve(monkeypatch):
    only(monkeypatch, "mamba")
    pm = P.detect(environ={})
    argv = pm.install_argv("lammps", "paraview")
    assert argv[-2:] == ["lammps", "paraview"]
    # One solve, not two: that is most of the speed win.
    assert argv.count("install") == 1


def test_channel_is_overridable(monkeypatch):
    only(monkeypatch, "mamba")
    pm = P.detect(environ={})
    assert "nvidia" in pm.install_argv("cuda", channel="nvidia")


def test_install_is_always_noninteractive(monkeypatch):
    """Without -y an unattended install hangs on a prompt forever."""
    for name in P.PREFERENCE:
        only(monkeypatch, name)
        conda_says(monkeypatch, "conda 24.9.2")
        assert "-y" in P.detect(environ={}).install_argv("lammps")


# ---------------------------------------------------------------------------
# micromamba bootstrap platform token
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("system,machine,expected", [
    ("Linux", "x86_64", "linux-64"),
    ("Linux", "aarch64", "linux-aarch64"),
    ("Linux", "ppc64le", "linux-ppc64le"),
    ("Darwin", "arm64", "osx-arm64"),
    ("Darwin", "x86_64", "osx-64"),
])
def test_micromamba_platform_tokens(system, machine, expected):
    assert P.micromamba_platform(system, machine) == expected


def test_micromamba_url_is_the_official_endpoint():
    url = P.MICROMAMBA_URL.format(platform="linux-64")
    assert url.startswith("https://micro.mamba.pm/api/micromamba/")
