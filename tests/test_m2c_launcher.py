"""The MPI launcher pre-flight, and the summary's third state.

`--scale full` reported two independent-looking tier failures when there was
one problem -- no `mpirun` on PATH -- and the summary printed `[ ok ] saha`
for a check that never ran because the solver was not installed. Both are
pinned here.
"""

import importlib.util
import stat
import sys
from pathlib import Path

import pytest

# Load scripts/test_m2c.py by PATH, under a name of its own.
#
# `import test_m2c` does NOT get the script: tests/test_m2c.py is already in
# sys.modules under that name, and pytest's rootdir import puts it first. The
# tests then run against the wrong module and fail with AttributeError --
# which is how this was caught, after the commit rather than before it.
_SRC = Path(__file__).resolve().parents[1] / "scripts" / "test_m2c.py"
_spec = importlib.util.spec_from_file_location("m2c_test_script", _SRC)
T = importlib.util.module_from_spec(_spec)
sys.modules["m2c_test_script"] = T
_spec.loader.exec_module(T)


def _exe(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


# --------------------------------------------------------------------------
# find_launcher
# --------------------------------------------------------------------------

def test_launcher_found_on_path_is_used(tmp_path, monkeypatch):
    mpirun = _exe(tmp_path / "bin" / "mpirun")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setattr(T, "linked_mpi", lambda e: ("libmpi.so.40", None))
    got, advice = T.find_launcher("/nonexistent/m2c")
    assert got == str(mpirun)
    assert any("libmpi.so.40" in a for a in advice)


def test_launcher_recovered_from_the_linked_library(tmp_path, monkeypatch):
    """The case Hasan hit: MPI is installed, its bin/ is just not on PATH.

    The binary resolves libmpi to <prefix>/lib, and mpirun lives in
    <prefix>/bin. Telling him to install MPI would have been wrong advice
    about a machine that already has it.
    """
    prefix = tmp_path / "opt" / "ompi"
    lib = prefix / "lib" / "libmpi.so.40"
    lib.parent.mkdir(parents=True)
    lib.write_text("")
    mpirun = _exe(prefix / "bin" / "mpirun")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(T, "linked_mpi",
                        lambda e: ("libmpi.so.40", str(lib)))

    got, advice = T.find_launcher("/nonexistent/m2c")
    assert got == str(mpirun)
    joined = "\n".join(advice)
    assert "not on PATH" in joined
    # The advice must be runnable as written, not a description of a fix.
    assert f"export PATH={prefix / 'bin'}:$PATH" in joined


def test_no_launcher_says_mpi_is_installed_when_it_is(tmp_path, monkeypatch):
    lib = tmp_path / "lib" / "libmpi.so.12"
    lib.parent.mkdir(parents=True)
    lib.write_text("")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(T, "linked_mpi", lambda e: ("libmpi.so.12", str(lib)))
    got, advice = T.find_launcher("/nonexistent/m2c")
    assert got is None
    joined = "\n".join(advice)
    assert "an MPI IS installed" in joined
    assert "module load" in joined or "activate" in joined


def test_no_launcher_and_no_linked_mpi(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(T, "linked_mpi", lambda e: (None, None))
    got, advice = T.find_launcher("/nonexistent/m2c")
    assert got is None
    assert any("no MPI launcher found" in a for a in advice)


def test_family_mismatch_is_named(monkeypatch, tmp_path):
    """Launching MPICH's binary with Open MPI's mpirun gives a useless error.

    It surfaced once in this project as 'Invalid communicator', which names
    neither MPI nor the mismatch. The soname has to be stated up front.
    """
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(T, "linked_mpi", lambda e: ("libmpi.so.12", None))
    _got, advice = T.find_launcher("/nonexistent/m2c")
    joined = "\n".join(advice)
    assert "libmpi.so.12" in joined
    assert "Invalid communicator" in joined


def test_srun_is_tried_last():
    """`srun` on PATH does not mean a bare srun runs outside an allocation."""
    assert T._LAUNCHERS[-1] == "srun"
    assert T._LAUNCHERS[0] == "mpirun"


# --------------------------------------------------------------------------
# the summary's third state
# --------------------------------------------------------------------------

def test_skipped_is_not_reported_as_passed(capsys):
    rc = T._summary({"grammar": True, "saha": None})
    out = capsys.readouterr().out
    assert rc == 0
    assert "skipped" in out
    # The exact regression: a skipped tier printed with the pass marker.
    saha_line = [ln for ln in out.splitlines() if "saha" in ln][0]
    assert "[ ok ]" not in saha_line


def test_failure_still_dominates_a_skip(capsys):
    rc = T._summary({"grammar": True, "saha": None, "shipped": False})
    out = capsys.readouterr().out
    assert rc == 1
    assert "first failure: shipped" in out


def test_launcher_failure_is_reported_once_not_per_tier(capsys):
    """One missing launcher is one problem, not one per tier."""
    rc = T._summary({"grammar": True, "launcher": False})
    out = capsys.readouterr().out
    assert rc == 1
    assert out.count("[FAIL]") == 1


# --------------------------------------------------------------------------
# refusing a run that cannot finish
# --------------------------------------------------------------------------

class _Cfg:
    def __init__(self, hours):
        self._h = hours
        self.target = "Al"

    def estimate_resources(self, cores):
        return {"cells": 3.55e5, "n_steps": 61700,
                "wall_hours_estimate": self._h}


def test_declines_a_run_the_timeout_would_kill(tmp_path, monkeypatch, capsys):
    """`--scale full` estimates ~38 days; a 20-minute kill proves nothing.

    Reported as [FAIL], it reads as a broken bridge. It is not: the deck
    never got near finishing.
    """
    monkeypatch.setattr(T, "write_problem_directory",
                        lambda cfg, out, cores: {})
    launched = []
    monkeypatch.setattr(T, "_run_deck",
                        lambda *a, **k: launched.append(a) or True)

    got = T.tier2_smoke(_Cfg(902.5), tmp_path, "/bin/true", 8, 20.0)
    out = capsys.readouterr().out
    assert got is None, "must skip, not pass and not fail"
    assert not launched, "must not launch a run it knows will be killed"
    assert "declining to launch" in out
    assert "--scale smoke" in out


def test_runs_when_the_estimate_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "write_problem_directory",
                        lambda cfg, out, cores: {})
    launched = []
    monkeypatch.setattr(T, "_run_deck",
                        lambda *a, **k: launched.append(a) or True)
    got = T.tier2_smoke(_Cfg(0.05), tmp_path, "/bin/true", 8, 20.0)
    assert got is True and launched
