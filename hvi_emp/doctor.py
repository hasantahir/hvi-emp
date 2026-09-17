"""Environment check: what this machine can actually run.

    python -m hvi_emp doctor

Reports the hardware, which of the optional solvers are present, and runs a
live self-test of the core chain -- then prints a prioritised list of what to
do next.  Written so that "can I run this here?" is answered by the machine
rather than by guesswork.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

OK, WARN, BAD, INFO = "  OK  ", " WARN ", " MISS ", " ---- "


def _hr(title: str = "") -> str:
    return f"\n{'-' * 76}\n{title}\n{'-' * 76}" if title else "-" * 76


def _run(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode == 0, (r.stdout or r.stderr).strip()
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int) -> list:
    """Minimal greedy word wrap (no textwrap import for one call site)."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines


def hardware() -> dict:
    """CPU, memory, disk and GPU."""
    info = {"platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine()}

    info["cores_logical"] = os.cpu_count() or 1
    try:
        info["cores_physical"] = len(os.sched_getaffinity(0))
    except AttributeError:
        info["cores_physical"] = info["cores_logical"]

    # memory
    mem = None
    try:
        mem = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1e9
    except (ValueError, AttributeError, OSError):
        pass
    info["memory_GB"] = mem

    # disk where output would go
    try:
        du = shutil.disk_usage(os.getcwd())
        info["disk_free_GB"] = du.free / 1e9
    except Exception:
        info["disk_free_GB"] = None

    g = gpu()
    info["gpu"] = g["name"]          # truthy only when actually usable
    info["gpu_state"] = g["state"]
    info["gpu_detail"] = g["detail"]
    return info


def cuda_toolkit_version() -> str | None:
    """CUDA toolkit major.minor from `nvcc --version`, or None."""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    ok, txt = _run([nvcc, "--version"])
    if not ok:
        return None
    import re
    m = re.search(r"release (\d+)\.(\d+)", txt)
    return f"{m.group(1)}.{m.group(2)}" if m else None


def cupy_package() -> str:
    """The CuPy wheel that matches this machine's CUDA toolkit.

    CuPy ships one binary wheel per CUDA *major* version and they are not
    interchangeable -- `cupy-cuda12x` links against the CUDA 12 runtime and
    will not work against a 13.x install. Advising the wrong one costs a
    failed install and a confusing error, so it is derived rather than
    hardcoded. CUDA 13 support arrived in CuPy 13.6 as `cupy-cuda13x`.
    """
    ver = cuda_toolkit_version()
    if not ver:
        return "cupy-cuda12x"      # no toolkit found; 12.x is the common case
    return f"cupy-cuda{ver.split('.')[0]}x"


def compute_capability() -> dict:
    """What acceleration is actually available to the Python framework.

    Distinct from `gpu()`, which reports the *card and driver* for the
    external solvers. This reports what `hvi_emp` itself can use, which is a
    different question: the framework's own speed comes from process
    parallelism, and CUDA only enters for the bulk array stages.
    """
    from .accel import GPU_MIN_ELEMENTS, gpu_info
    from .parallel import cpu_count

    cores = cpu_count()
    g = gpu_info()
    advice = []
    if cores > 1:
        advice.append(
            f"{cores} cores usable: velocity_sweep() and ionisation-table "
            f"builds run across them automatically. A single expansion is a "
            f"sequential ODE and cannot use more than one.")
    if not g.get("available"):
        ctk = cuda_toolkit_version()
        pkg = cupy_package()
        where = (f"Your CUDA toolkit is {ctk}, so the matching wheel is "
                 f"`{pkg}`" if ctk else
                 f"No CUDA toolkit found; `{pkg}` is the usual choice")
        advice.append(
            "No CuPy/CUDA. This costs nothing for ordinary runs -- the GPU "
            "paths only cover MD post-processing and volume sampling, which "
            f"need >= {GPU_MIN_ELEMENTS:,} elements before a GPU beats the "
            f"PCIe transfer. {where}: `pip install {pkg}`.")
    return {"cores": cores, "cupy": g, "cuda_toolkit": cuda_toolkit_version(),
            "cupy_package": cupy_package(), "advice": advice}



def find_openmhd(roots=None) -> str | None:
    """Locate an OpenMHD source tree or a build of it.

    OpenMHD is *not* installed to a prefix and produces no executable called
    ``OpenMHD``: each problem directory is compiled in place by its own
    Makefile into ``a.out`` (serial) and ``ap.out`` (MPI).  Looking for it on
    $PATH therefore always fails, so we look for the tree instead.

    ``$OPENMHD``, when set, is **authoritative**: if you point it somewhere
    that is not an OpenMHD tree, that is reported rather than silently
    papered over by finding a different copy in your home directory. Guessing
    past an explicit setting is how you end up debugging the wrong build.

    ``roots`` overrides the search entirely, for tests.
    """
    def describe(root):
        if not root or not os.path.isdir(root):
            return None
        if not os.path.isfile(os.path.join(root, "param.h")):
            return None
        built = [d for d in ("2D_basic", "2D_basic_gpu", "2D_reconnection")
                 if os.path.isfile(os.path.join(root, d, "ap.out"))
                 or os.path.isfile(os.path.join(root, d, "a.out"))]
        state = ("built: " + ", ".join(built) if built
                 else "source only, not yet compiled")
        return f"{root} ({state})"

    if roots is None:
        env = os.environ.get("OPENMHD")
        if env:
            found = describe(env)
            if found:
                return found
            # Explicitly set and wrong: say so instead of searching elsewhere.
            return None
        roots = [os.path.join(os.getcwd(), "OpenMHD"),
                 os.path.expanduser("~/OpenMHD"),
                 os.path.expanduser("~/src/OpenMHD")]

    for root in roots:
        found = describe(root)
        if found:
            return found
    return None


#: NVIDIA's PCI vendor ID, and the PCI classes for display controllers.
_NVIDIA_VENDOR = "0x10de"
_DISPLAY_CLASSES = ("0x030000", "0x030200", "0x030100", "0x038000")


def nvidia_pci_devices(root: str = "/sys/bus/pci/devices") -> list:
    """NVIDIA display/compute devices visible on the PCI bus.

    Read straight from sysfs, so this works with no driver loaded and no
    ``lspci`` installed -- which is exactly the situation we need to detect.
    A card that is physically present but has no driver is a completely
    different problem from no card at all, and ``nvidia-smi`` alone cannot
    tell them apart: it is absent in both cases.

    ``root`` is a parameter so this can be tested against a synthetic sysfs
    tree instead of the real one.
    """
    found = []
    if not os.path.isdir(root):
        return found                                    # not Linux
    for slot in sorted(os.listdir(root)):
        base = os.path.join(root, slot)
        try:
            with open(os.path.join(base, "vendor")) as fh:
                vendor = fh.read().strip().lower()
            with open(os.path.join(base, "class")) as fh:
                pci_class = fh.read().strip().lower()
        except OSError:
            continue
        if vendor != _NVIDIA_VENDOR:
            continue
        if not pci_class.startswith(("0x0300", "0x0302", "0x0301", "0x0380")):
            continue
        device = ""
        try:
            with open(os.path.join(base, "device")) as fh:
                device = fh.read().strip()
        except OSError:
            pass
        found.append(f"{slot} (device {device})" if device else slot)
    return found


#: PCI device ID -> (marketing name, compute capability, Kokkos arch macro).
#: Deliberately small: this table exists only for the case where the driver is
#: broken, so nvidia-smi cannot be asked.  Once the driver works,
#: ``nvidia-smi --query-gpu=compute_cap`` is authoritative and should be
#: preferred over anything here.
GPU_DEVICE_IDS = {
    "0x2b85": ("GeForce RTX 5090 (GB202)", "120", "BLACKWELL120"),
    "0x2c02": ("GeForce RTX 5080 (GB203)", "120", "BLACKWELL120"),
    "0x2f04": ("GeForce RTX 5070 Ti (GB203)", "120", "BLACKWELL120"),
    "0x2bb1": ("RTX PRO 6000 Blackwell (GB202)", "120", "BLACKWELL120"),
    "0x2901": ("B200 (GB100)", "100", "BLACKWELL100"),
    "0x2330": ("H100 (GH100)", "90", "HOPPER90"),
    "0x2331": ("H100 PCIe (GH100)", "90", "HOPPER90"),
    "0x26b9": ("L40S (AD102)", "89", "ADA89"),
    "0x26b5": ("L40 (AD102)", "89", "ADA89"),
    "0x2684": ("GeForce RTX 4090 (AD102)", "89", "ADA89"),
    "0x20b2": ("A100 80GB (GA100)", "80", "AMPERE80"),
    "0x20b0": ("A100 40GB (GA100)", "80", "AMPERE80"),
    "0x2235": ("A40 (GA102)", "86", "AMPERE86"),
    "0x2204": ("GeForce RTX 3090 (GA102)", "86", "AMPERE86"),
    # Professional Ampere workstation cards -- the common case for a desk-side
    # research box, and the reason this table needed extending.
    "0x2231": ("RTX A5000 (GA102GL)", "86", "AMPERE86"),
    "0x2232": ("RTX A4500 (GA102GL)", "86", "AMPERE86"),
    "0x2233": ("RTX A5500 (GA102GL)", "86", "AMPERE86"),
    "0x2230": ("RTX A6000 (GA102GL)", "86", "AMPERE86"),
    "0x24b0": ("RTX A4000 (GA104GL)", "86", "AMPERE86"),
    "0x1eb8": ("Tesla T4 (TU104)", "75", "TURING75"),
    "0x1db6": ("V100 32GB (GV100)", "70", "VOLTA70"),
}

#: Minimum CUDA toolkit that can *target* each compute capability.  Getting
#: this wrong produces "nvcc fatal: Unsupported gpu architecture", which is
#: an easy hour to lose.
CUDA_MIN_TOOLKIT = {"70": "9.0", "72": "9.0", "75": "10.0",
                    "80": "11.0", "86": "11.1", "87": "11.4", "89": "11.8",
                    "90": "11.8", "100": "12.8", "120": "12.8"}


def _smi_card() -> tuple | None:
    """``(name, compute_capability)`` from nvidia-smi, or None.

    This is the authoritative source whenever the driver is working -- the
    PCI table below is a small hand-maintained fallback for the case where
    it is *not*, and it will always lag new hardware.
    """
    ok, out = _run(["nvidia-smi", "--query-gpu=name,compute_cap",
                    "--format=csv,noheader"])
    if not ok or not out.strip():
        return None
    first = out.strip().splitlines()[0]
    if "," not in first:
        return None
    name, cc = (p.strip() for p in first.split(",", 1))
    cc = cc.replace(".", "")                    # "8.6" -> "86"
    return (name, cc) if cc.isdigit() else None


def _kokkos_arch_for(cc: str) -> str:
    """Kokkos ``Kokkos_ARCH_*`` macro for a compute capability.

    Derived rather than tabulated, so a card newer than the table still gets
    the right flag as long as its generation is known.
    """
    return {"70": "VOLTA70", "72": "VOLTA72", "75": "TURING75",
            "80": "AMPERE80", "86": "AMPERE86", "87": "AMPERE86",
            "89": "ADA89", "90": "HOPPER90", "100": "BLACKWELL100",
            "120": "BLACKWELL120"}.get(cc, f"SM{cc}")


def gpu_build_flags(device_id: str | None = None) -> dict | None:
    """The exact GPU build flags for the card in this machine.

    Prefers ``nvidia-smi`` when the driver is working, and falls back to the
    PCI device ID when it is not -- which is the whole point of having a
    table at all, since a card with a broken driver is invisible to
    nvidia-smi but still on the bus.

    The order used to be the other way round. On a working machine with an
    RTX A5000 that was not in the table, this printed "card not in this
    table. Once the driver works: ..." while the driver was demonstrably
    working and nvidia-smi could have answered immediately. The table's own
    comment said nvidia-smi should be preferred; the code did not do it.

    Returns None when no NVIDIA device is present at all, and a dict with
    ``unknown=True`` only when both routes fail.
    """
    if device_id is None:
        smi = _smi_card()
        if smi is not None:
            name, cc = smi
            return _build_flag_dict(name, cc, _kokkos_arch_for(cc),
                                    device_id="(from nvidia-smi)")
        devices = nvidia_pci_devices()
        if not devices:
            return None
        # nvidia_pci_devices() formats as "<slot> (device 0x....)"
        device_id = devices[0].split("device ")[-1].rstrip(")").lower()

    entry = GPU_DEVICE_IDS.get(device_id)
    if entry is None:
        return {"unknown": True, "device_id": device_id,
                "hint": ("card not in this table, and nvidia-smi could not "
                         "be asked (no working driver). Once it works:\n"
                         "        nvidia-smi --query-gpu=name,compute_cap "
                         "--format=csv,noheader\n"
                         "      then use that compute capability below "
                         "(e.g. 8.9 -> 89 / ADA89).")}
    name, cc, kokkos = entry
    return _build_flag_dict(name, cc, kokkos, device_id)


def _build_flag_dict(name: str, cc: str, kokkos: str,
                     device_id: str) -> dict:
    # Dotted form for tools that want it (AMReX). The *last* digit is the
    # minor version, so 89 -> 8.9 but 120 -> 12.0, not 1.20.
    cc_dotted = f"{cc[:-1]}.{cc[-1]}"
    return {
        "unknown": False, "device_id": device_id, "name": name,
        "compute_capability": cc, "compute_capability_dotted": cc_dotted,
        "kokkos_arch": kokkos,
        "cuda_min": CUDA_MIN_TOOLKIT.get(cc, "?"),
        "idefix": f"-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_{kokkos}=ON",
        "picongpu": f'pic-build -b "cuda:{cc}"',
        "warpx": f'-DWarpX_COMPUTE=CUDA -DAMReX_CUDA_ARCH={cc_dotted}',
        "openmhd": f"nvfortran -cuda -gpu=cc{cc}",
        "lammps": 'mamba install -c conda-forge "lammps=*=cuda*"',
    }


def nvidia_driver_diagnosis() -> str:
    """Work out *why* a present nvidia-smi cannot reach the driver.

    There are four common causes and they need different fixes, so guessing
    "try rebooting" is not good enough:

    1. The DKMS module was never rebuilt for the running kernel.  Very common
       after a kernel upgrade, and the usual cause on a brand-new kernel.
    2. The installed driver branch is too old to build against the running
       kernel at all -- no amount of rebuilding helps, the driver must be
       newer.
    3. Secure Boot is rejecting the unsigned module.
    4. Module built and loaded, but userspace libraries are from a different
       version (the classic "Driver/library version mismatch"), which a reboot
       does fix.
    """
    lines, causes = [], []

    running = platform.release()
    lines.append(f"      running kernel      {running}")

    loaded = False
    try:
        with open("/proc/modules") as fh:
            loaded = any(ln.startswith("nvidia ") or ln.startswith("nvidia_")
                         for ln in fh)
    except OSError:
        pass
    lines.append(f"      nvidia kernel module {'loaded' if loaded else 'NOT LOADED'}")

    # Which kernels does DKMS have the module built for?
    built_for = []
    if shutil.which("dkms"):
        ok, out = _run(["dkms", "status"])
        if ok and out:
            for ln in out.splitlines():
                if "nvidia" in ln.lower():
                    lines.append(f"      dkms                {ln.strip()}")
                    if "installed" in ln:
                        # "nvidia/580.x, 6.14.0-27-generic, x86_64: installed"
                        for part in ln.replace(":", ",").split(","):
                            part = part.strip()
                            if "-generic" in part or part.startswith("6."):
                                built_for.append(part)
        else:
            lines.append("      dkms                no nvidia module registered")
    else:
        lines.append("      dkms                not installed")

    # Kernel headers: DKMS cannot build anything without them, and this is the
    # commonest reason the build silently never happened.
    headers = os.path.isdir(f"/lib/modules/{running}/build")
    lines.append(f"      kernel headers      "
                 f"{'present' if headers else 'MISSING'}"
                 f"  (/lib/modules/{running}/build)")

    if not headers:
        causes.append(
            f"the kernel headers for {running} are not installed, so DKMS "
            "cannot build the module\n        at all -- which is why "
            "/lib/modules has no nvidia.ko.\n"
            f"        sudo apt install linux-headers-{running}\n"
            "        sudo dkms autoinstall && sudo reboot")

    if not headers:
        pass                                   # headline already set above
    elif built_for and not any(running in b for b in built_for):
        causes.append(
            f"the DKMS module is built for {', '.join(sorted(set(built_for)))} "
            f"but you are running {running}.\n"
            "        sudo dkms autoinstall && sudo reboot\n"
            "        If that fails to build, the driver branch is too old for "
            "this kernel:\n"
            "        sudo ubuntu-drivers install   # or boot the older kernel "
            "from GRUB")
    elif not loaded and not built_for:
        causes.append(
            "no NVIDIA kernel module is built or loaded, so nvidia-smi is a "
            "leftover userspace binary.\n"
            "        sudo ubuntu-drivers install && sudo reboot")
    elif not loaded:
        causes.append(
            "the module is built for this kernel but is not loaded.\n"
            "        sudo modprobe nvidia    # read the error it prints\n"
            "        sudo dmesg | grep -i nvidia | tail -20")
    else:
        causes.append(
            "the module is loaded, so this is most likely a userspace/kernel "
            "version mismatch from an update applied without a reboot.\n"
            "        sudo reboot")

    # What the package manager thinks is installed, which is often at odds
    # with what is actually built.
    if shutil.which("dpkg-query"):
        ok, pkgs = _run(["dpkg-query", "-W", "-f=${Package} ${Version}\n",
                         "nvidia-dkms-*", "nvidia-driver-*"])
        for ln in (pkgs or "").splitlines():
            if ln.strip() and "no packages found" not in ln.lower():
                lines.append(f"      apt package         {ln.strip()}")

    ok, sb = _run(["mokutil", "--sb-state"]) if shutil.which("mokutil") \
        else (False, "")
    if ok and "enabled" in sb.lower():
        lines.append("      secure boot         ENABLED")
        causes.append(
            "Secure Boot is enabled, which refuses unsigned DKMS modules. "
            "Either enrol the\n        MOK key when prompted during driver "
            "install, or disable Secure Boot in the BIOS.")

    text = "\n".join(lines)
    text += "\n\n      Most likely cause:\n        " + causes[0]
    for extra in causes[1:]:
        text += "\n      Also:\n        " + extra
    return text


def gpu() -> dict:
    """GPU state, distinguishing 'none' from 'present but unusable'.

    Returns a dict with:
      ``state``  -- "ready" | "driver_error" | "driver_missing" | "absent"
      ``name``   -- the nvidia-smi description, or None unless usable
      ``detail`` -- what to do about it
    """
    smi = shutil.which("nvidia-smi")
    if smi:
        ok, txt = _run([smi, "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader"])
        if ok and txt:
            return {"state": "ready", "name": txt, "detail": smi}
        # nvidia-smi exists but will not talk to a device.  This has several
        # distinct causes with different fixes, so diagnose rather than guess.
        first = txt.splitlines()[0] if txt else "no output"
        return {"state": "driver_error", "name": None,
                "detail": f"nvidia-smi runs but cannot reach the driver:\n"
                          f"      {first}\n" + nvidia_driver_diagnosis()}

    devices = nvidia_pci_devices()
    if devices:
        return {"state": "driver_missing", "name": None,
                "detail": (f"{len(devices)} NVIDIA device(s) on the PCI bus "
                           f"({devices[0]}) but no nvidia-smi: the card is "
                           "there, the driver is not.\n"
                           "      Ubuntu: sudo ubuntu-drivers install\n"
                           "      then reboot and re-run this check.\n"
                           "      If this is a container, the host driver "
                           "must be passed through (--gpus all).")}
    return {"state": "absent", "name": None,
            "detail": ("no NVIDIA device on the PCI bus either, so this is a "
                       "CPU-only machine.\n      Everything here runs on CPU; "
                       "only the GPU builds of Idefix, PIConGPU, WarpX,\n"
                       "      OpenMHD and LAMMPS are unavailable.")}


# ---------------------------------------------------------------------------
# Software
# ---------------------------------------------------------------------------

def python_packages() -> list:
    """(name, status, detail) for the Python dependencies."""
    out = []
    for name, required in (("numpy", True), ("scipy", True),
                           ("matplotlib", False), ("pytest", False)):
        try:
            m = __import__(name)
            out.append((name, OK, getattr(m, "__version__", "?")))
        except ImportError:
            out.append((name, BAD if required else WARN,
                        "required" if required else "optional"))
    return out


def gpu_toolchains() -> list:
    """Compilers that could target the GPU, for the solvers that can use it."""
    out = []
    nvf = shutil.which("nvfortran")
    out.append(("nvfortran (NVIDIA HPC SDK)", OK if nvf else WARN,
                nvf or "not found -- needed only for OpenMHD's *_gpu problems"))
    nvcc = shutil.which("nvcc")
    ver = ""
    if nvcc:
        ok, txt = _run([nvcc, "--version"])
        if ok:
            ver = next((ln.strip() for ln in txt.splitlines()
                        if "release" in ln), "")
    out.append(("nvcc (CUDA toolkit)", OK if nvcc else WARN,
                ver or nvcc or "not found -- needed to build WarpX with CUDA"))
    cm = shutil.which("cmake")
    out.append(("cmake", OK if cm else WARN,
                cm or "not found -- needed to build WarpX from source"))
    return out


def solvers() -> list:
    """(name, status, detail) for the production-solver bridges."""
    from .solvers import (have_openpmd, have_picmi, have_pywarpx,
                          lammps_diagnostics)

    out = []
    d = lammps_diagnostics()
    out.append(("LAMMPS (MD, Stage 1)", OK if d["ok"] else WARN, d["reason"]))

    out.append(("WarpX / pywarpx (PIC, Stage 3)",
                OK if have_pywarpx() else WARN,
                "importable" if have_pywarpx() else
                "not installed -- deck generation still works"))
    out.append(("picmistandard", OK if have_picmi() else WARN,
                "for WarpX PICMI scripts"))
    out.append(("openpmd-api", OK if have_openpmd() else WARN,
                "to read WarpX output back"))

    # external executables
    isale = (os.environ.get("ISALE")
             or shutil.which("iSALE2D") or shutil.which("isale2d"))
    out.append(("iSALE (hydro, Stage 1)", OK if isale else INFO,
                isale or "optional, not pursued -- access requests went "
                         "unanswered; M2C covers this stage"))

    from .solvers.m2c_stage1 import find_m2c
    m2c = find_m2c()
    out.append(("M2C (multi-material, Stage 1)", OK if m2c else WARN,
                m2c or "not found -- git clone github.com/kevinwgy/m2c, set "
                       "$M2C_HOME; GPLv3, no application needed"))

    idefix = os.environ.get("IDEFIX_DIR")
    if idefix and not os.path.isfile(os.path.join(idefix, "CMakeLists.txt")):
        idefix = None
    out.append(("Idefix (MHD/GPU, Stage 2)", OK if idefix else WARN,
                idefix or "not found -- git clone --recurse-submodules "
                          "github.com/idefix-code/idefix, set $IDEFIX_DIR"))

    picsrc = os.environ.get("PICSRC") or (
        os.path.dirname(os.path.dirname(shutil.which("pic-build")))
        if shutil.which("pic-build") else None)
    out.append(("PIConGPU (PIC/GPU, Stage 3)", OK if picsrc else WARN,
                picsrc or "not found -- git clone github.com/"
                          "ComputationalRadiationPhysics/picongpu, set "
                          "$PICSRC and add $PICSRC/bin to PATH"))

    omhd = find_openmhd()
    out.append(("OpenMHD (MHD, Stage 2)", OK if omhd else WARN,
                omhd or "not found -- git clone github.com/zenitani/OpenMHD, "
                        "then set $OPENMHD to the source tree"))

    pv = shutil.which("paraview") or shutil.which("pvpython")
    out.append(("ParaView (.vti viewer)", OK if pv else WARN,
                pv or "not found -- needed only to *view* the output"))
    if pv:
        # pvbatch is what renders a movie headlessly; on a remote workstation
        # its absence is the difference between a GPU animation and none.
        pvb = shutil.which("pvbatch")
        out.append(("  pvbatch (headless render)", OK if pvb else WARN,
                    pvb or "not found -- needed for offscreen GPU animation; "
                           "ships with most ParaView builds"))

    mpi = shutil.which("mpirun") or shutil.which("mpiexec")
    out.append(("MPI runtime", OK if mpi else WARN,
                mpi or "not found -- needed for M2C, OpenMHD and iSALE3D"))
    return out


# ---------------------------------------------------------------------------
# Live self-test
# ---------------------------------------------------------------------------

def self_test(verbose: bool = True) -> dict:
    """Run the core chain and time it. This is the real check."""
    import warnings

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        from .pipeline import run_scenario
        t0 = time.time()
        sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3)
        results["scenario_s"] = time.time() - t0
        results["scenario_ok"] = (sc.expansion is not None
                                  and sc.emp is not None)
        results["T_eV"] = sc.impact.plasma.T_eV
        results["Q_frozen"] = sc.expansion.Q_final if sc.expansion else 0.0

        from .validation import run_all
        t0 = time.time()
        val = run_all(verbose=False)
        results["validation_s"] = time.time() - t0
        results["validation"] = f"{val['n_pass']}/{val['n_total']}"
        # A documented open discrepancy is not a regression. Requiring 100%
        # green would only invite widening tolerances until it is.
        results["validation_ok"] = val.get("ok", True)
        results["regressions"] = val.get("n_regressions", 0)
        results["open_issues"] = [c.name for c in val.get("open_issues", [])]

        # a small .vti series, which exercises the writer end to end
        import tempfile

        from .viz import write_impact_series
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            r = write_impact_series(sc, td, n_frames=4, shape=(32, 32, 32),
                                    verbose=False)
            results["vti_s"] = time.time() - t0
            results["vti_ok"] = len(r["files"]) == 4
            results["vti_MB_per_frame"] = r["bytes"] / 1e6 / 4
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(skip_self_test: bool = False) -> int:
    """Print the whole diagnosis. Returns a shell exit code."""
    from . import __version__

    print(_hr(f"HVI-EMP {__version__} -- environment check"))

    hw = hardware()
    print(f"\n{'platform':<22}{hw['platform']}")
    print(f"{'python':<22}{hw['python']} ({hw['machine']})")
    print(f"{'cores':<22}{hw['cores_physical']} available "
          f"/ {hw['cores_logical']} logical")
    print(f"{'memory':<22}"
          + (f"{hw['memory_GB']:.0f} GB" if hw["memory_GB"] else "unknown"))
    print(f"{'disk free (cwd)':<22}"
          + (f"{hw['disk_free_GB']:.0f} GB" if hw["disk_free_GB"] else "unknown"))
    label = {"ready": "", "driver_error": "PRESENT BUT BROKEN -- ",
             "driver_missing": "PRESENT BUT NO DRIVER -- ",
             "absent": "none -- "}[hw["gpu_state"]]
    print(f"{'GPU':<22}{label}"
          + (hw["gpu"] if hw["gpu"] else hw["gpu_detail"].split("\n")[0]))
    if hw["gpu_state"] != "ready":
        for line in hw["gpu_detail"].split("\n")[1:]:
            print(line if line.startswith(" ") else f"                      {line}")

    print(_hr("Acceleration available to hvi_emp itself"))
    cc = compute_capability()
    print(f"{'parallel workers':<22}{cc['cores']}")
    cu = cc["cupy"]
    if cu.get("available"):
        print(f"{'CuPy / CUDA':<22}{cu['name']} "
              f"({cu['sm_arch']}, {cu['total_GB']:.0f} GB, "
              f"{cu['free_GB']:.0f} GB free, CuPy {cu['cupy_version']})")
        print(f"{'':<22}run `python -c \"from hvi_emp.accel import "
              f"accel_selftest; print(accel_selftest())\"`")
        print(f"{'':<22}to verify GPU/CPU agreement -- the CUDA paths ship "
              f"untested.")
    else:
        print(f"{'CuPy / CUDA':<22}not available")
    for line in cc["advice"]:
        for i, chunk in enumerate(_wrap(line, 56)):
            print(f"{'':<22}{chunk}" if i else f"{'':<22}{chunk}")

    print(_hr("Python packages"))
    pkgs = python_packages()
    for name, status, detail in pkgs:
        print(f"[{status}] {name:<28}{detail}")

    print(_hr("Solvers and tools (all optional)"))
    for name, status, detail in solvers():
        print(f"[{status}] {name:<32}{detail}")

    flags = gpu_build_flags()
    if flags is not None:
        print(_hr("GPU build flags for this card"))
        if flags["unknown"]:
            print(f"  device {flags['device_id']}: {flags['hint']}")
        else:
            print(f"  {flags['name']}, compute capability {flags['compute_capability']}")
            print(f"  needs CUDA toolkit >= {flags['cuda_min']} to target it "
                  "at all\n")
            for solver, key in (("Idefix", "idefix"),
                                ("PIConGPU", "picongpu"),
                                ("WarpX", "warpx"),
                                ("OpenMHD", "openmhd"),
                                ("LAMMPS", "lammps")):
                print(f"    {solver:<10}{flags[key]}")
        for name, status, detail in gpu_toolchains():
            print(f"\n[{status}] {name:<32}{detail}", end="")
        print()

    missing_required = [n for n, s, _ in pkgs if s == BAD]
    if missing_required:
        print(f"\n[{BAD}] cannot run: missing {', '.join(missing_required)}")
        print("       pip install numpy scipy")
        return 1

    if not skip_self_test:
        print(_hr("Self-test (running the real chain)"))
        try:
            r = self_test()
        except Exception as exc:
            import traceback
            print(f"[{BAD}] self-test raised {type(exc).__name__}: {exc}")
            # Say *where*. "raised AttributeError" with no location gives
            # neither the user nor a bug report anything to act on.
            frames = traceback.extract_tb(exc.__traceback__)
            ours = [f for f in frames if "hvi_emp" in (f.filename or "")]
            for f in (ours or frames)[-3:]:
                print(f"         {os.path.basename(f.filename)}:{f.lineno} "
                      f"in {f.name}()")
                if f.line:
                    print(f"           {f.line.strip()}")
            print("\n       Versions in this interpreter:")
            for mod in ("numpy", "scipy"):
                try:
                    m = __import__(mod)
                    print(f"         {mod:<8}{getattr(m, '__version__', '?')}"
                          f"   {getattr(m, '__file__', '?')}")
                except ImportError:
                    print(f"         {mod:<8}not importable")
            print("\n       If this is a version problem the line above says "
                  "which package and\n       where it was imported from -- "
                  "worth checking for two installs on sys.path.")
            return 1
        print(f"[{OK if r['scenario_ok'] else BAD}] full scenario"
              f"{'':<20}{r['scenario_s']:.1f} s  "
              f"(T_e = {r['T_eV']:.2f} eV, Q = {r['Q_frozen']:.2e} C)")
        n_open = len(r.get("open_issues", []))
        print(f"[{OK if r['validation_ok'] else BAD}] validation suite"
              f"{'':<17}{r['validation_s']:.1f} s  "
              f"({r['validation']} checks pass"
              + (f", {n_open} known open" if n_open else "") + ")")
        for name in r.get("open_issues", []):
            print(f"         open: {name}")
            print("               run `python -m hvi_emp validate` for why")
        if r.get("regressions"):
            print(f"         {r['regressions']} REGRESSION(S) -- see "
                  "`python -m hvi_emp validate`")
        print(f"[{OK if r['vti_ok'] else BAD}] .vti writer"
              f"{'':<23}{r['vti_s']:.1f} s  "
              f"({r['vti_MB_per_frame']:.2f} MB/frame at 32^3)")

    # ---- guidance ------------------------------------------------------
    print(_hr("What you can run now"))
    cores = hw["cores_physical"]
    print("""  python -m hvi_emp run --projectile Fe --target Al --mass 1e-12 --velocity 50e3
  python examples/01_single_impact.py          # the whole chain + figure
  python examples/14_impact_scenes.py          # impact + plume ParaView scenes
  python examples/14_impact_scenes.py --list-quality   # what your RAM allows
  python -m hvi_emp validate                   # 36 published-data checks
  python scripts/benchmark.py                  # where the time goes here

  config-driven runs (docs/WORKFLOW.md):
  python -m hvi_emp.run --print-config         # resolve + validate, run nothing
  python -m hvi_emp.run sweep=velocity         # a sweep, across every core""")
    from .cluster import scheduler_available
    from .config import USING_HYDRA
    from .schema import USING_PYDANTIC
    if not (USING_HYDRA and USING_PYDANTIC):
        missing = [n for n, ok in (("hydra-core", USING_HYDRA),
                                   ("pydantic", USING_PYDANTIC)) if not ok]
        print(f"    ({', '.join(missing)} absent -- the fallback composer and "
              f"validator\n     are in use, with identical rules. "
              f"`pip install '.[workflow]'` for the full set.)")
    if scheduler_available("slurm"):
        print("""
  this is a Slurm submit host:
  python -m hvi_emp.run sweep=velocity cluster=slurm_array --submit
  snakemake --profile workflow/profiles/slurm  # the DAG, one job per rule""")

    print(_hr("To go further on this machine"))
    todo = []
    if hw["gpu_state"] in ("driver_missing", "driver_error"):
        # Do this before any GPU build: compiling for CUDA against a machine
        # whose driver is absent wastes hours and then fails at run time.
        todo.append(("**the GPU driver** -- fix this before any GPU build",
                     "sudo ubuntu-drivers install && sudo reboot\n"
                     "      (the card is on the PCI bus; nothing can use it "
                     "until the driver loads)"))
    from .pkgmgr import detect
    # probe=True: this is a diagnostic, so it is worth a second of conda
    # startup to distinguish "conda, already on libmamba, nothing to do"
    # from "conda 22.9, classic resolver, this is why your install crawled".
    pm = detect(probe=True)
    from .solvers import have_pywarpx, lammps_diagnostics
    if not lammps_diagnostics()["ok"]:
        todo.append(("LAMMPS MD (Stage 1, nm scale)",
                     pm.install("lammps")
                     if os.environ.get("CONDA_PREFIX") and pm.name
                     else "pip install lammps mpich"))
    from .solvers.m2c_stage1 import find_m2c
    if not find_m2c():
        # The Stage-1 hydrocode route. iSALE is no longer prompted for:
        # access requests went unanswered, and M2C covers the same stage
        # while also doing oblique incidence, which iSALE2D cannot.
        todo.append(("M2C hydrocode (Stage 1) -- 3-D, tabular EOS, "
                     "non-ideal Saha",
                     "git clone https://github.com/kevinwgy/m2c.git\n"
                     "      cd m2c && mkdir build && cd build && cmake .. && "
                     "make -j\n"
                     "      export M2C_HOME=$PWD   # needs PETSc + MPI"))
    if not have_pywarpx():
        cpu_build = ((pm.install("warpx") + "   # CPU build, one command")
                     if pm.name else
                     "<pkgmgr> install -c conda-forge warpx   # CPU build; "
                     "needs micromamba/mamba/conda first, see below")
        if not hw["gpu"]:
            how = cpu_build
        elif shutil.which("spack"):
            # conda-forge's warpx is CPU-only; CUDA needs Spack or CMake.
            how = ("spack install warpx +python compute=cuda   # GPU build\n"
                   "      (conda-forge's warpx package is CPU-only -- "
                   "see docs/INSTALLING_SOLVERS.md §2)")
        else:
            # A GPU is present but spack is not. Printing the spack command
            # alone sends the reader to "command not found"; lead with the
            # build they can actually do now, and make the extra step to
            # the GPU build explicit rather than implied.
            how = (f"{cpu_build}\n"
                   "      The CUDA build needs Spack, which is not installed "
                   "here. To add it:\n"
                   "        git clone -c feature.manyFiles=true "
                   "https://github.com/spack/spack.git ~/spack\n"
                   "        . ~/spack/share/spack/setup-env.sh\n"
                   "        spack install warpx +python compute=cuda\n"
                   "      (conda-forge's warpx is CPU-only -- "
                   "docs/INSTALLING_SOLVERS.md §2. The CPU build above is "
                   "enough to run and validate the Stage-3 decks.)")
        todo.append(("WarpX PIC (Stage 3)", how))
    if not os.environ.get("IDEFIX_DIR"):
        gpu = " -- GPU-capable, spherical geometry, non-ideal MHD" if hw["gpu"] else ""
        todo.append((f"Idefix MHD (Stage 2){gpu}",
                     "git clone --recurse-submodules "
                     "https://github.com/idefix-code/idefix.git\n"
                     "      export IDEFIX_DIR=$PWD/idefix"))
    if not (os.environ.get("PICSRC") or shutil.which("pic-build")):
        if hw["gpu"]:
            todo.append(("PIConGPU (Stage 3) -- GPU-native PIC, probe "
                         "particles for the antenna",
                         "git clone https://github.com/"
                         "ComputationalRadiationPhysics/picongpu.git\n"
                         "      see docs/INSTALLING_SOLVERS.md section 4"))
    if not (shutil.which("paraview") or shutil.which("pvpython")):
        todo.append(("ParaView, to view the .vti output",
                     pm.install("paraview") if pm.name
                     else "see docs/INSTALLING_SOLVERS.md section 7"))
    # Package-manager advice last, and only when it changes what to do.
    # Three cases, three different truths:
    #   nothing installed -> several rows above cannot be actioned at all
    #   conda, classic    -> they will work, slowly; name the one-line fix
    #   anything else     -> say nothing; it is already fast
    if todo:
        if not pm.name:
            todo.append(("a conda-family installer -- several rows above "
                         "need one",
                         pm.speedup_hint().replace("\n", "\n      ")
                         + "\n      or: scripts/install_solvers.sh "
                           "--bootstrap-mamba"))
        elif pm.name == "conda":
            tip = pm.speedup_hint()
            if tip:
                todo.append((f"a faster solver -- {pm.why()}",
                             tip.replace("\n", "\n      ")))
    if todo:
        for what, how in todo:
            print(f"  * {what}\n      {how}")
    else:
        print("  everything is installed -- nothing to add")

    print(_hr("Sizing for this machine"))
    warpx_where = ("GPU, if built with CUDA" if hw["gpu"]
                   else "CPU (no GPU detected)")
    # Fletcher's six-speed velocity series, now run on M2C rather than
    # iSALE: M2C is MPI-parallel per run, so the six cases go sequentially
    # across all cores instead of concurrently as serial jobs.
    n_series = 6
    m2c_note = f"MPI, {min(cores, 64)} cores/case"
    rows = [
        ("reduced chain", "1 core", "seconds -- run anything"),
        ("M2C 2D velocity series", m2c_note,
         f"hours/case, {n_series} sequential"),
        ("LAMMPS MD (nm scale)", f"MPI, up to {min(cores, 64)} cores",
         "minutes"),
        # The trailing space matters: the columns are fixed-width and
        # "32 cores or GPU" ran straight into "hours" -> "or GPUhours".
        ("OpenMHD 2D cavity", f"MPI, up to {min(cores, 32)} cores"
         + (" or GPU " if hw["gpu"] else " "), "hours"),
        ("WarpX 2D PIC", warpx_where, "minutes to hours"),
    ]
    for what, how, when in rows:
        print(f"  {what:<26}{how:<26}{when}")

    if cores < 8:
        print(f"\n  note: {cores} cores is thin for M2C. The 2-D cases will "
              "run; a 3-D oblique case wants a cluster.")
    if hw["memory_GB"] and hw["memory_GB"] > 64:
        print(f"\n  {hw['memory_GB']:.0f} GB of RAM is not the constraint "
              "anywhere in this workflow except a 3-D M2C run --")
        print("  core count is, everywhere else.")
    if hw["gpu"]:
        # Count the list rather than asserting a number in prose. It said
        # "two things" above five bullets.
        gpu_users = 5
        print(f"\n  {gpu_users} things here can use your GPU, and none of "
              f"them does so by default:")
        print("    * Idefix   -- cmake -DKokkos_ENABLE_CUDA=ON. Easiest "
              "GPU win here.")
        print("    * PIConGPU -- pic-build -b cuda:<arch>. GPU-native by "
              "design.")
        print("    * WarpX    -- needs a CUDA build (Spack or CMake; the "
              "conda-forge\n                  package is CPU-only)")
        print("    * OpenMHD  -- its *_gpu directories are CUDA Fortran and "
              "need\n                  nvfortran from the NVIDIA HPC SDK")
        print("    * LAMMPS   -- conda-forge ships CUDA/Kokkos builds: "
              "lammps=*=cuda*")
        print("  the framework itself and the .vti writer are CPU-only by "
              "design.")
    print(_hr())
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(report())


__all__ = ["report", "hardware", "compute_capability", "python_packages",
           "solvers", "self_test",
           "find_openmhd", "gpu_toolchains"]
