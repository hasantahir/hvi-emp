"""Which conda-family package manager to use, and what to tell the user.

Three commands install conda-forge packages, and they differ by roughly an
order of magnitude in how long a solve takes:

  ``micromamba``  a single static binary; no base environment at all
  ``mamba``       the same libmamba solver, installed into a conda base env
  ``conda``       Python; since 23.10 it *defaults* to that same solver

That last line is what stops this module from being "conda bad, mamba good".
A current conda is **not** the slow case -- it is libmamba underneath, and
swapping it for mamba buys you process startup time and little else. The slow
case is specifically conda older than 23.10, or any conda pinned back to the
classic resolver, and that distinction matters because the fix there is one
config line rather than a fresh installation.

So the advice this module produces is conditional on what is actually on the
machine. Hardcoding ``conda install ...`` into a hint string was the original
bug: it told someone with micromamba to use a command they did not have, and
told someone on conda 22.9 nothing about why their solve had been running for
eleven minutes.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field

__all__ = [
    "PackageManager",
    "detect",
    "install_hint",
    "LIBMAMBA_DEFAULT",
    "PREFERENCE",
    "micromamba_platform",
    "MICROMAMBA_URL",
]

#: conda made libmamba the default solver in this release. Below it, a solve
#: uses the classic resolver unless the user opted in explicitly.
LIBMAMBA_DEFAULT = (23, 10)

#: conda 22.11 added the CEP-4 solver plugin hook, and with it the ``solver``
#: config key and ``--solver`` flag (conda-libmamba-solver 22.12.0).
#: **Below this, ``conda config --set solver libmamba`` is not a valid key**
#: and fails with "Key 'solver' is not a known primitive parameter."
LIBMAMBA_SOLVER_KEY = (22, 11)

#: conda-libmamba-solver 22.3 (the first public release) worked with conda
#: 4.12+ via the older ``experimental_solver`` key. Below 4.12 there is no
#: route to libmamba through conda at all; conda itself has to be replaced.
LIBMAMBA_REACHABLE = (4, 12)

#: Search order. Fastest first; every one of these speaks ``-c conda-forge``.
PREFERENCE = ("micromamba", "mamba", "conda")

MICROMAMBA_URL = "https://micro.mamba.pm/api/micromamba/{platform}/latest"


def micromamba_platform(system: str | None = None,
                        machine: str | None = None) -> str:
    """The platform token micro.mamba.pm expects, e.g. ``linux-64``."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    arm = machine in ("arm64", "aarch64")
    if system == "darwin":
        return "osx-arm64" if arm else "osx-64"
    if arm:
        return "linux-aarch64"
    if machine in ("ppc64le",):
        return "linux-ppc64le"
    return "linux-64"


def _version_of(exe: str, timeout: float = 15.0) -> tuple[int, ...] | None:
    """``(24, 9, 2)`` from ``conda --version``, or None if it will not say."""
    try:
        out = subprocess.run([exe, "--version"], capture_output=True,
                             text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", out.stdout or out.stderr or "")
    if not m:
        return None
    return tuple(int(g) for g in m.groups() if g is not None)


@dataclass
class PackageManager:
    """A resolved choice of installer, plus why it was chosen.

    ``name`` is None when nothing in :data:`PREFERENCE` is on PATH. Callers
    must handle that: it is the normal state on a machine that has only
    system Python, and the right answer there is pip, not a conda command
    the user cannot run.
    """

    name: str | None = None
    path: str | None = None
    version: tuple[int, ...] | None = None
    #: "libmamba", "classic", or None when it is not a conda-solver question.
    solver: str | None = None
    #: True only when we positively know solves are fast.
    fast: bool = False
    #: Environment prefix to install into, when one is active.
    prefix: str | None = None
    candidates: tuple[str, ...] = field(default_factory=tuple)

    # -- commands ---------------------------------------------------------

    def install_argv(self, *packages: str,
                     channel: str = "conda-forge") -> list[str]:
        """Argv that installs ``packages``, or ``[]`` if there is no manager.

        micromamba gets an explicit ``-p $CONDA_PREFIX``. It keys off
        ``MAMBA_ROOT_PREFIX``, not ``CONDA_PREFIX``, so inside an activated
        *conda* env a bare ``micromamba install`` would resolve against the
        wrong target -- silently, and only visibly wrong later when the
        import fails.
        """
        if not self.name:
            return []
        argv = [self.name, "install", "-y", "-c", channel]
        if self.name == "micromamba" and self.prefix:
            # After the subcommand: -p is an option of `install`, not a
            # global flag, and micromamba rejects it in front.
            argv[2:2] = ["-p", self.prefix]
        return argv + list(packages)

    def install(self, *packages: str, channel: str = "conda-forge") -> str:
        """The install command as a copy-pasteable string."""
        argv = self.install_argv(*packages, channel=channel)
        return " ".join(argv)

    # -- advice -----------------------------------------------------------

    @property
    def is_slow_conda(self) -> bool:
        """conda that will use the classic resolver for this solve."""
        return self.name == "conda" and self.solver == "classic"

    def why(self) -> str:
        """One line: what was picked and what that means for solve time."""
        if not self.name:
            return ("no micromamba, mamba or conda on PATH")
        if self.name in ("micromamba", "mamba"):
            return f"{self.name} (libmamba solver)"
        ver = ".".join(str(p) for p in self.version) if self.version else "?"
        if self.solver == "libmamba":
            return (f"conda {ver} -- already defaults to the libmamba solver, "
                    f"so this is as fast as mamba bar process startup")
        if self.solver == "classic":
            if self.version and self.version < LIBMAMBA_SOLVER_KEY:
                return (f"conda {ver} -- classic resolver, and too old to "
                        f"switch (no `solver` key before 22.11)")
            return (f"conda {ver} -- classic resolver, solves can take "
                    f"minutes")
        return f"conda {ver}"

    def speedup_hint(self) -> str | None:
        """How to make installs fast, or None when they already are.

        Deliberately offers the config line before the new installation:
        ``conda config --set solver libmamba`` takes a second and needs no
        download, and on conda >= 23.10 it is all that was ever wrong.
        """
        if self.name in ("micromamba", "mamba"):
            return None
        if self.name == "conda":
            if self.solver == "libmamba":
                return None
            if self.solver is None:
                # detect(probe=False): we never asked conda its version, so
                # we do not know whether this is the slow case. Saying
                # "your conda is old" here would be a guess presented as a
                # diagnosis -- exactly the failure this module exists to
                # stop. Stay silent; callers that care pass probe=True.
                return None
            v = self.version
            # Band 1: new enough to default to libmamba, but pinned off it.
            if v and v >= LIBMAMBA_DEFAULT:
                return ("conda is pinned to the classic resolver. Undo that "
                        "-- it is a one-line fix, no download:\n"
                        "    conda config --set solver libmamba")
            # Band 2: the `solver` key exists; opt in (plugin may be needed).
            if v and v >= LIBMAMBA_SOLVER_KEY:
                return ("this conda predates the libmamba default (23.10) "
                        "but can opt in:\n"
                        "    conda install -n base -c conda-forge "
                        "conda-libmamba-solver\n"
                        "    conda config --set solver libmamba")
            # Band 3: too old for the solver plugin interface entirely.
            # `conda config --set solver libmamba` fails outright here with
            # "Key 'solver' is not a known primitive parameter" -- the key
            # arrived in 22.11 (plugin 22.12.0), and current plugin releases
            # want conda >= 23.3. Suggesting it would waste the reader's
            # time on a command that cannot work.
            ver = ".".join(str(p) for p in v) if v else "?"
            return (f"conda {ver} cannot use libmamba at all: the `solver` "
                    f"config key did not exist until 22.11. Upgrading conda "
                    f"in place from here is itself a slow classic solve, so "
                    f"the cheap route is a standalone micromamba, which "
                    f"ignores conda entirely:\n"
                    f"    scripts/install_solvers.sh --bootstrap-mamba\n"
                    f"  It installs into the active env via -p $CONDA_PREFIX, "
                    f"so the env you are in keeps working.")
        return ("install Miniforge -- it ships conda-forge as the default "
                "channel and mamba alongside conda:\n"
                "    https://github.com/conda-forge/miniforge#install")


def detect(probe: bool = True, environ: dict | None = None) -> PackageManager:
    """Find the best available manager.

    Parameters
    ----------
    probe
        Run ``--version`` on conda to tell the fast case from the slow one.
        Costs about a second of conda startup. Pass False on hot paths; the
        result then reports ``solver=None`` rather than guessing, because a
        guess here is exactly the thing that produced wrong advice before.
    """
    env = os.environ if environ is None else environ
    found = [n for n in PREFERENCE if shutil.which(n)]
    pm = PackageManager(candidates=tuple(found),
                        prefix=env.get("CONDA_PREFIX") or None)
    if not found:
        return pm

    pm.name = found[0]
    pm.path = shutil.which(pm.name)

    if pm.name in ("micromamba", "mamba"):
        pm.solver = "libmamba"
        pm.fast = True
        if probe:
            pm.version = _version_of(pm.name)
        return pm

    # conda: the answer depends on the version and on any explicit override.
    if probe:
        pm.version = _version_of("conda")
    override = (env.get("CONDA_SOLVER") or "").strip().lower()
    if override in ("libmamba", "classic"):
        pm.solver = override
    elif pm.version is not None:
        pm.solver = ("libmamba" if pm.version >= LIBMAMBA_DEFAULT
                     else "classic")
    # A condarc `solver:` key can override the version default and is not
    # read here; `why()` and `speedup_hint()` both stay true either way
    # because the config line they suggest is idempotent.
    pm.fast = pm.solver == "libmamba"
    return pm


def install_hint(*packages: str, pip_fallback: str = "",
                 indent: str = "    ", probe: bool = True) -> str:
    """The install advice to print, matched to this machine.

    Falls back to ``pip_fallback`` when no conda-family manager exists, which
    is the honest answer rather than naming a command the reader cannot run.
    """
    pm = detect(probe=probe)
    if not pm.name:
        return f"{indent}{pip_fallback}" if pip_fallback else (
            f"{indent}# {pm.speedup_hint()}")
    lines = [f"{indent}{pm.install(*packages)}"]
    tip = pm.speedup_hint()
    if tip:
        lines.append(f"{indent}# note: {tip}".replace("\n", f"\n{indent}#   "))
    return "\n".join(lines)
