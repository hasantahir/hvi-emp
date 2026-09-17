"""Bridges to open-source production solvers.

The reduced-order chain in the parent package is the connective tissue; each
stage can be replaced by an established open-source code:

    Stage 1  LAMMPS    (molecular dynamics)      `solvers.lammps_stage1`
    Stage 1  iSALE     (continuum shock physics) `solvers.isale_stage1`
    Stage 1  M2C       (multi-material + Saha)   `solvers.m2c_stage1`
    Stage 2  OpenMHD   (magnetohydrodynamics)    `solvers.openmhd_stage2`
    Stage 2  Idefix    (MHD, GPU, spherical)     `solvers.idefix_stage2`
    Stage 3  WarpX     (electromagnetic PIC)     `solvers.warpx_stage3`
    Stage 3  PIConGPU  (EM PIC, GPU-native)      `solvers.picongpu_stage3`

Where two codes cover one stage they are alternatives, not a pipeline: pick
whichever suits the machine and the question.  For Stage 2, Idefix is the
GPU-capable one and works in spherical geometry with non-ideal MHD; for
Stage 3, PIConGPU is GPU-native with probe particles that map onto a physical
antenna.  See docs/SOLVERS.md for the comparison.

For Stage 1, M2C is the one to reach for first.  It is GPLv3 on GitHub with
no application (iSALE needs one), it is 3-D so it can represent an oblique
impact -- which the reduced chain and axisymmetric iSALE2D structurally
cannot -- and it carries the two pieces of physics this framework is still
missing: tabular EOS (Tillotson, ANEOS) above the linear Us-Up fit's ceiling,
and a non-ideal Saha solver with continuum lowering.  It stops at the plasma
state, so Stage 3 remains ours.

Design rule: every bridge can always *generate* complete, runnable input decks
and can always *post-process* the solver's output back into the framework's
result objects, whether or not the solver is installed.  Actually *running*
the solver in-process is supported where a Python interface exists (LAMMPS
via its official wheel, WarpX via pywarpx).  `availability()` reports what
the current environment supports.

Why bridges and not a pure-MD/PIC rewrite
-----------------------------------------
The scales make a single first-principles code impossible.  Fraile et al.
(Nucl. Fusion 62, 026034, 2022) simulate the largest MD impacts in this
literature -- 40 million atoms -- and that is a projectile of radius ~13 nm
for 100 ps.  A 1 pg micrometeoroid is ~1e13 atoms and the EMP forms over
microseconds; a full-scale MD or kinetic treatment is ~1e6 times beyond
reach.  The viable use of these codes, and what this subpackage implements,
is: LAMMPS resolves what the shock-and-release closures approximate
(Stage 1, at nm scale, extrapolated by the scaling laws the reduced model
provides); WarpX resolves the charge-separation radiation mechanism that
Stage 3 closes with a single parameter; OpenMHD resolves the
magnetised-plume physics (diamagnetic cavity) that the reduced Stage 2
omits entirely.
"""

from __future__ import annotations

import ctypes
import glob
import contextlib
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys


#: DT_NEEDED entries look like libmpi.so.12 (MPICH) or libmpi.so.40 (Open MPI).
_MPI_SONAME_RE = re.compile(rb"libmpi(?:_[a-z0-9]+)?\.so\.\d+")

#: Sonames by family, for turning a soname into a name a human recognises.
_MPI_FAMILY = {"libmpi.so.12": "MPICH", "libmpi.so.40": "Open MPI"}


def mpi_family(soname: str) -> str:
    """"MPICH" / "Open MPI" / the soname itself when unrecognised."""
    return _MPI_FAMILY.get(soname, soname)


def _required_mpi_sonames() -> set[str]:
    """Which libmpi the *installed* LAMMPS was actually linked against.

    Read from the shared library rather than assumed. Assuming MPICH is what
    broke this: the wheels use MPICH, conda-forge ships both variants, and
    force-loading the wrong one is worse than loading none at all.
    """
    try:
        spec = importlib.util.find_spec("lammps")
        if spec is None or not spec.origin:
            return set()
        root = os.path.dirname(spec.origin)
    except Exception:
        return set()
    out: set[str] = set()
    for so in glob.glob(os.path.join(root, "**", "*.so*"), recursive=True):
        try:
            with open(so, "rb") as fh:
                out |= {m.decode() for m in _MPI_SONAME_RE.findall(fh.read())}
        except OSError:
            continue
    return out


def _loaded_mpi_sonames() -> set[str]:
    """MPI libraries already mapped into this process (Linux only)."""
    try:
        with open("/proc/self/maps", "rb") as fh:
            return {m.decode() for m in _MPI_SONAME_RE.findall(fh.read())}
    except OSError:
        return set()


def _preload_mpi() -> str | None:
    """Load the MPI runtime LAMMPS needs, if the loader cannot find it.

    The PyPI LAMMPS wheels link against MPICH's ``libmpi.so.12``, which the
    ``mpich`` wheel installs somewhere not on the default loader path, so a
    pre-load with RTLD_GLOBAL saves the user setting LD_LIBRARY_PATH.

    **The soname is now read from the installed library, not assumed.** The
    previous version hardcoded ``libmpi.so.12``. On an environment that had
    gained Open MPI (typically by installing PETSc for M2C alongside a
    conda-forge LAMMPS), that forced MPICH's symbols into a process holding
    Open MPI's communicators, and the run died with::

        Abort(...) internal_Comm_size(41): Invalid communicator

    which reads like a LAMMPS or cluster fault and is neither. If a
    *different* MPI is already mapped, this now refuses to pre-load and
    returns None, leaving the loader's own resolution alone.

    Returns the path loaded, or None if nothing was needed or safe.
    """
    needed = _required_mpi_sonames() or {"libmpi.so.12"}
    already = _loaded_mpi_sonames()

    # Mixing families in one process is the bug, not the fix. Bail out.
    if already and not (already & needed):
        return None

    for soname in sorted(needed):
        try:
            ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
            return None                              # already resolvable
        except OSError:
            pass

    candidates: list[str] = []
    bases = {os.path.dirname(os.path.dirname(p)) for p in sys.path
             if p.endswith("site-packages")}
    for soname in sorted(needed):
        for base in bases:
            candidates += glob.glob(os.path.join(base, "lib", soname))
        candidates += glob.glob(os.path.expanduser(f"~/.local/lib/{soname}"))
    for lib in candidates:
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            return lib
        except OSError:
            continue
    return None


def mpi_conflict() -> dict | None:
    """Report an MPI family mismatch around LAMMPS, or None if consistent.

    Returns ``{"needed": [...], "found": [...], "hint": str}``.
    """
    needed = _required_mpi_sonames()
    if not needed:
        return None
    prefix = os.environ.get("CONDA_PREFIX")
    present: set[str] = set()
    if prefix:
        for pat in ("lib/libmpi.so.*", "lib/libmpi_*.so.*"):
            for p in glob.glob(os.path.join(prefix, pat)):
                name = os.path.basename(p)
                if _MPI_SONAME_RE.fullmatch(name.encode()):
                    present.add(name)
    if not present or (present & needed):
        return None
    return {
        "needed": sorted(needed),
        "found": sorted(present),
        "hint": (
            f"LAMMPS is linked against "
            f"{', '.join(mpi_family(s) for s in sorted(needed))} but this "
            f"environment provides "
            f"{', '.join(mpi_family(s) for s in sorted(present))}.\n"
            "  Two MPI implementations in one process give "
            "'Invalid communicator'.\n"
            "  Pin one family for the whole environment, e.g.:\n"
            "    micromamba install -p $CONDA_PREFIX -c conda-forge "
            "'mpi=*=openmpi'\n"
            "  or keep them apart -- the chain stages are separate "
            "processes and\n"
            "  do not need to share an environment."),
    }


#: Marker for the upstream version-parsing crash described in
#: :func:`_lammps_version_shim`. Matched against the ValueError text.
_LAMMPS_VERSION_BUG = "unconverted data remains"


def _normalise_lammps_version(vstring: str) -> str:
    """Trim a LAMMPS distribution version to the ``Y.m.d`` it claims to be.

    ``"2025.7.22.4.0"`` -> ``"2025.7.22"``. Anything already parseable, or
    not of this shape, is returned untouched.
    """
    parts = vstring.split(".")
    return ".".join(parts[:3]) if len(parts) > 3 else vstring


@contextlib.contextmanager
def _lammps_version_shim():
    """Work around LAMMPS's own import-time version parsing.

    Upstream ``lammps/__init__.py`` computes ``__version__`` like this::

        vstring = importlib.metadata.version('lammps')
        t = time.strptime(vstring, "%Y.%m.%d")

    but the distribution metadata LAMMPS itself ships carries a four- or
    five-component version -- ``2025.7.22.4.0`` on both the PyPI wheel and
    the conda-forge package. ``strptime`` consumes ``2025.7.22``, chokes on
    the remainder, and raises::

        ValueError: unconverted data remains: .4.0

    at *import* time, so the whole package is unusable even though the
    library underneath is fine. The PyPI wheel dodges this by hardcoding
    ``__version__`` instead of calling the function; conda-forge ships the
    upstream file, so conda-forge installs are the ones that break.

    This patches ``importlib.metadata.version`` for the duration of the
    import only. ``get_version_number`` does ``from importlib.metadata
    import version`` *inside* the function body, at call time, so it picks
    up the patched attribute. Nothing on disk is modified, and the shim is
    removed again immediately -- a global monkeypatch left in place would
    be a far worse bug than the one it fixes.
    """
    import importlib.metadata as _md

    real = _md.version

    def patched(name, *a, **k):
        v = real(name, *a, **k)
        return _normalise_lammps_version(v) if name == "lammps" else v

    _md.version = patched
    try:
        yield
    finally:
        _md.version = real


def import_lammps():
    """Import and return the ``lammps`` module, working around the above.

    Use this instead of a bare ``import lammps`` anywhere in the package.
    """
    with _lammps_version_shim():
        return importlib.import_module("lammps")


#: Set in the child so the probe does not recurse into another subprocess.
_PROBE_ENV = "HVI_EMP_LAMMPS_PROBE"

_PROBE_SRC = """
import json, sys
try:
    from hvi_emp.solvers import _preload_mpi, import_lammps
    lib = _preload_mpi()
    lmp = import_lammps().lammps(
        cmdargs=["-log", "none", "-screen", "none"])
    v = lmp.version()
    lmp.close()
    sys.stdout.write("@@" + json.dumps(
        {"ok": True, "version": v, "lib": lib}))
except BaseException as exc:
    sys.stdout.write("@@" + json.dumps(
        {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}))
"""

#: Fragments that mean "two MPI stacks in one process", not "LAMMPS broken".
_MPI_MISMATCH_MARKERS = ("Invalid communicator", "MPI_Comm_size",
                         "API version is incompatible", "MPI_Init",
                         "internal_Comm_size")


def _probe_lammps_subprocess(timeout: float = 120.0) -> dict | None:
    """Start LAMMPS in a child process. None = caller should do it inline.

    Returns a dict with ``ok`` and either ``version`` or ``error``/``hint``.
    A child killed by MPI_Abort or a signal is reported, not re-raised.
    """
    if os.environ.get(_PROBE_ENV):
        return None                       # we *are* the child
    env = dict(os.environ, **{_PROBE_ENV: "1"})
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    env["PYTHONPATH"] = os.pathsep.join(
        [pkg_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    try:
        out = subprocess.run([sys.executable, "-c", _PROBE_SRC],
                             capture_output=True, text=True,
                             timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError):
        return None                       # cannot spawn; fall back inline

    blob = out.stdout.rsplit("@@", 1)[-1] if "@@" in out.stdout else ""
    if blob:
        try:
            res = json.loads(blob)
        except ValueError:
            res = None
        if isinstance(res, dict) and (res.get("ok") or out.returncode == 0):
            return res
        if isinstance(res, dict):
            return res

    # No verdict: the child died before it could report.
    noise = (out.stderr or "") + (out.stdout or "")
    conflict = mpi_conflict()
    if conflict or any(m in noise for m in _MPI_MISMATCH_MARKERS):
        detail = conflict["hint"] if conflict else (
            "LAMMPS aborted inside MPI startup. The usual cause is two MPI\n"
            "  implementations in one process.\n"
            "  Pin one family for the environment, e.g.:\n"
            "    micromamba install -p $CONDA_PREFIX -c conda-forge "
            "'mpi=*=openmpi'")
        first = next((ln.strip() for ln in noise.splitlines()
                      if any(m in ln for m in _MPI_MISMATCH_MARKERS)), "")
        return {"ok": False,
                "error": f"MPI aborted at startup ({first[:120]})"
                         if first else "MPI aborted at startup",
                "hint": "This is an MPI mismatch, not a LAMMPS fault.\n  "
                        + detail}
    return {"ok": False,
            "error": f"probe exited {out.returncode}",
            "hint": (noise.strip()[-600:] or "no output from the probe")}


def lammps_diagnostics() -> dict:
    """Why LAMMPS is or is not usable, with a platform-appropriate hint.

    "Not installed" and "installed but will not load" need completely
    different fixes, and the second is the common one: the LAMMPS wheels link
    against MPICH's ``libmpi.so.12``, so importing without the ``mpich`` wheel
    (or a system MPI) raises ``OSError: libmpi.so.12: cannot open shared
    object file`` even though ``pip show lammps`` looks fine.

    Returns
    -------
    dict with ``ok`` (bool), ``stage`` ("ok" | "import" | "instantiate"),
    ``reason`` (str) and ``hint`` (multi-line str).
    """
    import platform

    from ..pkgmgr import detect

    conda = bool(os.environ.get("CONDA_PREFIX"))
    mac = platform.system() == "Darwin"
    # Name the command this machine actually has. Hardcoding "conda install"
    # here used to hand micromamba users a command they could not run.
    pm = detect(probe=False)

    def _hint(kind: str) -> str:
        if kind == "missing":
            if conda and pm.name:
                out = (f"    {pm.install('lammps')}\n"
                       "  or, to use the pip wheel inside the env:\n"
                       "    pip install lammps mpich")
                tip = pm.speedup_hint()
                return out + (f"\n  ({tip.splitlines()[0]})" if tip else "")
            return "    pip install lammps mpich"
        # loaded-but-broken
        base = ("The module is installed but its shared library will not "
                "load.\n"
                "  Usually the MPI runtime it links against is missing:\n"
                "    pip install mpich\n")
        if conda and pm.name:
            base += ("  Inside a conda-family env, prefer a consistent "
                     "stack:\n"
                     f"    {pm.install('lammps')}\n")
        if kind == "version_bug":
            return (
                "This is a bug in LAMMPS's own package, not in your "
                "install.\n"
                "  lammps/__init__.py computes __version__ with\n"
                "    time.strptime(metadata_version, \"%Y.%m.%d\")\n"
                "  but the metadata it ships reads like '2025.7.22.4.0',\n"
                "  so the import dies on the trailing '.4.0'. The library\n"
                "  itself is fine.\n"
                "  hvi_emp works around it automatically; if you want a\n"
                "  permanent fix, either use the PyPI wheel, which\n"
                "  hardcodes __version__ instead of parsing it:\n"
                "    pip install lammps mpich\n"
                "  or edit the last lines of\n"
                "    $CONDA_PREFIX/lib/python*/site-packages/lammps/"
                "__init__.py\n"
                "  to read: __version__ = 0\n")
        if mac:
            base += ("  On macOS also check the wheel matches your Python's\n"
                     "  architecture (arm64 vs x86_64 under Rosetta):\n"
                     "    python -c \"import platform; "
                     "print(platform.machine())\"\n")
        return base

    # The shim is applied on both attempts. If a bare import fails with the
    # upstream version-parsing ValueError but the shimmed one succeeds, we
    # report that we worked around it rather than silently hiding a
    # third-party bug the user may want to report.
    shimmed = False
    try:
        importlib.import_module("lammps")
    except ModuleNotFoundError as exc:
        return {"ok": False, "stage": "import", "reason": str(exc),
                "hint": _hint("missing")}
    except ValueError as exc:
        if _LAMMPS_VERSION_BUG not in str(exc):
            return {"ok": False, "stage": "import",
                    "reason": f"ValueError: {exc}", "hint": _hint("broken")}
        try:
            import_lammps()
            shimmed = True
        except Exception as exc2:
            return {"ok": False, "stage": "import",
                    "reason": f"{type(exc2).__name__}: {exc2}",
                    "hint": _hint("version_bug")}
    except Exception as exc:
        return {"ok": False, "stage": "import", "reason": f"{type(exc).__name__}: {exc}",
                "hint": _hint("broken")}

    # importable -- can it actually start?
    #
    # Out of process by default. Starting LAMMPS initialises MPI, and a
    # mismatched MPI stack calls MPI_Abort, which terminates the *whole*
    # interpreter. In-process, that meant one broken solver killed `doctor`
    # outright and the user got no report at all -- not even the sections
    # that had already succeeded. A crash here must be data, not death.
    probe = _probe_lammps_subprocess()
    if probe is not None:
        if probe.get("ok"):
            notes = []
            if probe.get("lib"):
                notes.append(f"MPI preloaded from {probe['lib']}")
            if shimmed:
                notes.append("upstream version-parse bug worked around")
            return {"ok": True, "stage": "ok",
                    "reason": f"LAMMPS {probe['version']} ready"
                              + (f" ({'; '.join(notes)})" if notes else ""),
                    "hint": ""}
        return {"ok": False, "stage": "instantiate",
                "reason": probe.get("error", "probe failed"),
                "hint": probe.get("hint") or _hint("broken")}

    try:
        lib = _preload_mpi()
        lammps = import_lammps()
        lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])
        version = lmp.version()
        lmp.close()
        notes = []
        if lib:
            notes.append(f"MPI preloaded from {lib}")
        if shimmed:
            notes.append("upstream version-parse bug worked around")
        return {"ok": True, "stage": "ok",
                "reason": f"LAMMPS {version} ready"
                          + (f" ({'; '.join(notes)})" if notes else ""),
                "hint": ""}
    except Exception as exc:
        return {"ok": False, "stage": "instantiate",
                "reason": f"{type(exc).__name__}: {exc}",
                "hint": _hint("broken")}


def have_lammps() -> bool:
    """True if the LAMMPS Python module can be imported and instantiated.

    See `lammps_diagnostics()` when this is False and you want to know why.
    """
    return lammps_diagnostics()["ok"]


def have_pywarpx() -> bool:
    """True if WarpX's Python interface is importable."""
    try:
        importlib.import_module("pywarpx")
        return True
    except Exception:
        return False


def have_picmi() -> bool:
    """True if the PICMI standard interface is importable."""
    try:
        importlib.import_module("picmistandard")
        return True
    except Exception:
        return False


def have_openpmd() -> bool:
    """True if openPMD-api (WarpX output reader) is importable."""
    try:
        importlib.import_module("openpmd_api")
        return True
    except Exception:
        return False


def _m2c_status() -> str:
    """One line on M2C: what the bridge can do, and whether M2C is here.

    Imported lazily so `availability()` costs nothing extra when the caller
    does not have M2C and does not care.
    """
    try:
        from .m2c_stage1 import find_m2c
        found = find_m2c()
    except Exception as exc:                      # pragma: no cover
        return f"deck generation and readback (bridge import failed: {exc})"
    if found:
        return f"deck generation and readback (M2C at {found})"
    return ("deck generation and readback (M2C not found -- "
            "git clone https://github.com/kevinwgy/m2c; GPLv3, no "
            "application needed, unlike iSALE)")


def availability() -> dict:
    """What the current environment can do, bridge by bridge."""
    diag = lammps_diagnostics()
    lmp = diag["ok"]
    pwx, pic, pmd = have_pywarpx(), have_picmi(), have_openpmd()
    return {
        "lammps": lmp,
        "lammps_reason": diag["reason"],
        "pywarpx": pwx,
        "picmi": pic,
        "openpmd": pmd,
        "stage1_lammps": ("run in-process" if lmp else
                          "deck generation only -- " + diag["reason"]),
        "stage3_warpx": ("run in-process" if pwx else
                         "input generation only (install WarpX; "
                         "see docs/SOLVERS.md)"),
        "stage2_openmhd": "input generation only (Fortran source; "
                          "see docs/SOLVERS.md)",
        "stage1_isale": "input generation only (iSALE is free to academics "
                        "on application; see docs/REPLICATION_FLETCHER2021.md)",
        "stage1_m2c": _m2c_status(),
        "stage2_idefix": ("input generation only (compiles the setup into "
                          "the binary; see docs/INSTALLING_SOLVERS.md)"),
        "stage3_picongpu": ("input generation only (compiles .param files "
                            "into the binary; openPMD readback needs "
                            + ("openpmd-api, which is installed)" if pmd
                               else "openpmd-api, which is NOT installed)")),
    }


__all__ = ["availability", "have_lammps", "lammps_diagnostics",
           "have_pywarpx", "have_picmi", "have_openpmd"]
