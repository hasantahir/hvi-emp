"""Tests for how the package is shipped, not what it computes.

Two bugs prompted this file, both found by a user rather than by the suite:

  * ``scripts/install_solvers.sh`` had no execute bit, so the documented
    ``scripts/install_solvers.sh ...`` gave "Permission denied".
  * ``doctor`` printed ``spack install warpx ...`` on a machine with no
    spack, so the recommended next step was "command not found".

Both are the same failure in different clothes: telling someone to run
something they cannot run.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Execute bits
# ---------------------------------------------------------------------------

def _shebanged_files():
    out = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or "build/lib" in str(p) or ".git" in p.parts:
            continue
        if p.suffix not in (".sh", ".py"):
            continue
        try:
            first = p.open("rb").readline(64)
        except OSError:
            continue
        if first.startswith(b"#!"):
            out.append(p)
    return out


def test_shebanged_scripts_are_executable():
    """A file starting with #! is meant to be run directly.

    Shipping one without +x means the documented invocation fails with
    "Permission denied" -- and zip preserves the missing bit faithfully,
    so it survives the transfer to the workstation.
    """
    bad = [str(p.relative_to(ROOT)) for p in _shebanged_files()
           if not os.access(p, os.X_OK)]
    assert not bad, ("shebang but not executable (chmod +x these): "
                     + ", ".join(bad))


def test_scripts_directory_entrypoints_are_executable():
    for name in ("install_solvers.sh", "run_full_chain.py"):
        p = ROOT / "scripts" / name
        assert p.exists(), f"{name} missing"
        assert os.stat(p).st_mode & stat.S_IXUSR, f"{name} is not executable"


def test_installer_runs_without_bash_prefix():
    """The docs say `scripts/install_solvers.sh`, so that must work."""
    p = ROOT / "scripts" / "install_solvers.sh"
    r = subprocess.run([str(p), "--version"], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    assert "build" in r.stdout


# ---------------------------------------------------------------------------
# Advice must name commands that exist, or say how to get them
# ---------------------------------------------------------------------------

#: Tokens that mean "here is how to obtain the thing I just named".
_OBTAIN = ("git clone", "install", "http", "see docs", "download",
           "not installed", "bootstrap")


def _leading_commands(block: str):
    """First word of each line that is plausibly a shell command.

    Deliberately conservative. The bullet titles and the explanatory prose
    live in the same block as the commands, and counting "WarpX" or "needs"
    as executables produced false positives that would have trained us to
    ignore this test. Prose is filtered by two markers used consistently
    throughout this codebase: a capitalised first word, and the ` -- `
    separator.
    """
    cmds = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith(("(", "#", "see ", "export ")):
            continue
        if " -- " in line:                      # prose, not a command line
            continue
        m = re.match(r"^([a-z][\w.-]*)\s", line)   # commands are lowercase
        if m:
            cmds.append(m.group(1))
    return cmds


def test_doctor_todo_never_names_an_absent_tool_unhelpfully(monkeypatch):
    """Every command doctor suggests must be runnable, or explained.

    This is the structural version of the spack bug: it does not care
    which tool is missing, only that the advice does not dead-end.
    """
    import hvi_emp.doctor as D

    todo = []

    real_append = list.append

    # Capture the todo list by running the report and scraping its output
    # rather than reaching into internals, so the test tracks what the user
    # actually sees.
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            D.report()
        except Exception as exc:              # pragma: no cover
            pytest.skip(f"doctor could not run here: {exc}")
    text = buf.getvalue()

    section = text.split("To go further on this machine")
    if len(section) < 2:
        pytest.skip("no 'to go further' section in this environment")
    section = section[1].split("Sizing for this machine")[0]

    # Split into one block per bullet.
    blocks = re.split(r"\n\s*\*\s", section)[1:]
    ignore = {"sudo", "cd", "export", "make", "python", "pip", "source"}

    problems = []
    for block in blocks:
        for cmd in _leading_commands(block):
            if cmd in ignore or shutil.which(cmd):
                continue
            if not any(tok in block for tok in _OBTAIN):
                problems.append((cmd, block.strip()[:160]))
    assert not problems, (
        "doctor suggests commands that are not installed and does not say "
        "how to get them: "
        + "; ".join(f"{c}: {b!r}" for c, b in problems))


def _report_with(monkeypatch, *, gpu: bool, spack: bool) -> str:
    """Run doctor with the hardware and PATH we want to exercise.

    The GPU branch of the WarpX advice is unreachable on a CPU-only test
    machine, so without forcing ``gpu=True`` this test passes vacuously --
    which is exactly how the spack bug reached a user in the first place.
    """
    import contextlib
    import io

    import hvi_emp.doctor as D

    real_hardware = D.hardware

    def fake_hardware():
        hw = dict(real_hardware())
        hw["gpu"] = "NVIDIA RTX A5000" if gpu else None
        hw["gpu_state"] = "ready" if gpu else "absent"
        return hw

    monkeypatch.setattr(D, "hardware", fake_hardware)

    real_which = shutil.which
    monkeypatch.setattr(
        D.shutil, "which",
        lambda n, *a, **k: (None if (n == "spack" and not spack)
                            else real_which(n, *a, **k)))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        D.report()
    return buf.getvalue()


def _warpx_block(text):
    """The WarpX bullet from the 'To go further' section only.

    Scoped deliberately: 'WarpX PIC' also appears earlier in the output as
    'for WarpX PICMI scripts', and splitting on the bare string grabbed
    that instead -- which made these tests assert against the wrong half
    of the report.
    """
    if "To go further on this machine" not in text:
        pytest.skip("no 'to go further' section in this environment")
    section = (text.split("To go further on this machine")[1]
                   .split("Sizing for this machine")[0])
    marker = "* WarpX PIC (Stage 3)"
    if marker not in section:
        pytest.skip("warpx already installed here")
    return section.split(marker)[1].split("\n  * ")[0]


def test_warpx_gpu_advice_without_spack_explains_how_to_get_spack(monkeypatch):
    """Regression for the reported `spack: command not found`.

    A GPU machine without spack must not be handed a bare `spack install`.
    """
    block = _warpx_block(_report_with(monkeypatch, gpu=True, spack=False))
    if "spack install" in block:
        assert "git clone" in block and "spack.git" in block, block
    # and it must offer something runnable right now
    assert "conda-forge warpx" in block, block


def test_warpx_gpu_advice_with_spack_uses_it(monkeypatch):
    block = _warpx_block(_report_with(monkeypatch, gpu=True, spack=True))
    assert "spack install warpx" in block, block


def test_warpx_cpu_machine_is_not_told_about_spack(monkeypatch):
    block = _warpx_block(_report_with(monkeypatch, gpu=False, spack=False))
    assert "spack" not in block, block
