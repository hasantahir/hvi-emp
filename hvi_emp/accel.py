"""CUDA acceleration, applied only where the arithmetic is actually shaped for it.

Read this before using it
-------------------------
A GPU is a throughput device with a fixed per-kernel cost of roughly 5-10 us.
That single number decides where CUDA belongs in this framework, and profiling
says it is a *small* part of it:

===========================  ===========  ==========================================
workload                     array size   verdict
===========================  ===========  ==========================================
Stage-2 expansion ODE        24 doubles   **CPU only.** 44,000 sequential steps, each
                                          dependent on the last. Kernel launch costs
                                          more than the whole right-hand side; the
                                          GPU would be ~100x *slower*.
Saha table build             10,240 pts   CPU multiprocessing wins -- `brentq` is
                                          scalar, iterative and branchy.
LAMMPS frame post-process    10^6-10^8    **GPU.** Per-atom binning and reduction
                             atoms        over millions of independent atoms.
`.vti` volume sampling       10^6-10^8    **GPU.** Pure elementwise evaluation of an
                             voxels       analytic field on a grid.
Batched table lookup         10^5+ pts    **GPU.** Gather plus arithmetic, no
                                          branching.
===========================  ===========  ==========================================

So the honest summary is: **CUDA does not make a single impact simulation
faster.** It makes the bulk array stages faster -- the ones that dominate when
post-processing molecular-dynamics output or rendering a volume -- and those
are exactly the stages that hurt today. For throughput across many scenarios,
process-level parallelism (`hvi_emp.parallel`) is the tool, not the GPU.

Design
------
Rather than duplicate every routine, this module exposes the array namespace
in use (`xp`), so a function written once against `xp` runs on either device.
CuPy deliberately mirrors the NumPy API, which makes this practical.

    from .accel import get_namespace, to_device, to_host

    xp, on_gpu = get_namespace(prefer_gpu=True)
    d = to_device(host_array, xp)
    ...                                       # same code either way
    return to_host(result)

Transfer cost is the trap. A PCIe round trip is ~10 GB/s against ~1 TB/s of
on-device bandwidth, so an operation that moves an array to the GPU, does one
pass over it and moves it back is bandwidth-bound on the *transfer* and gains
nothing. `should_use_gpu` encodes the crossover as a minimum element count,
below which the CPU path is chosen even when a GPU is present.

Numerical agreement
-------------------
CuPy uses the same IEEE-754 doubles, but reductions run in a different order,
so sums over large arrays can differ in the last bits. Results are therefore
expected to agree to floating-point tolerance, not bitwise. Anything that
must be reproducible exactly should stay on the CPU path, and this module
never silently switches a *physics* result to the GPU -- callers opt in.

State of testing
----------------
The CPU paths here are exercised by the test suite. **The CUDA paths have not
been executed** -- this was developed on a machine with no GPU, and the target
workstation's driver was not loading at the time of writing (`nvidia-smi`
failing against an RTX 5090 with `nvidia-dkms-590` not built for kernel
6.17). They are written against the documented CuPy API and are structurally
simple, but treat their first run as unverified: `python -m hvi_emp doctor`
reports what is actually available, and `accel_selftest()` below checks
GPU-vs-CPU agreement on real arrays once a driver is present.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cupy_available", "gpu_info", "get_namespace", "to_device",
           "to_host", "should_use_gpu", "GPU_MIN_ELEMENTS", "accel_selftest"]

#: Below this many elements the PCIe round trip costs more than the kernel
#: saves, so the CPU path is used even when a GPU is present. Chosen as the
#: order of magnitude at which a single-pass elementwise operation stops
#: being transfer-bound; tune with `scripts/benchmark.py --gpu`.
GPU_MIN_ELEMENTS = 1_000_000

_CUPY = None
_CUPY_CHECKED = False


def cupy_available() -> bool:
    """True if CuPy imports *and* a device is actually usable.

    Importing CuPy succeeds on a machine with the library but no working
    driver; the device query is what fails. Both are checked here so callers
    get one honest answer instead of an exception later.
    """
    global _CUPY, _CUPY_CHECKED
    if not _CUPY_CHECKED:
        _CUPY_CHECKED = True
        try:
            import cupy                                     # noqa: PLC0415
            cupy.cuda.runtime.getDeviceCount()
            _CUPY = cupy
        except Exception:                                   # noqa: BLE001
            _CUPY = None
    return _CUPY is not None


def gpu_info() -> dict:
    """What device is present, or why there isn't one.

    Never raises: this is called by `doctor`, whose job is to report a
    broken environment rather than fail in it.
    """
    if not cupy_available():
        try:
            import cupy                                     # noqa: PLC0415,F401
            reason = ("CuPy is installed but no CUDA device is usable -- "
                      "this is normally a driver problem, not a CuPy "
                      "problem. Check `nvidia-smi`.")
        except Exception as exc:                            # noqa: BLE001
            reason = f"CuPy not importable: {exc}"
        return {"available": False, "reason": reason}
    cp = _CUPY
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        free, total = dev.mem_info
        cc = f"{props['major']}.{props['minor']}"
        return {
            "available": True,
            "name": props["name"].decode() if isinstance(props["name"], bytes)
            else str(props["name"]),
            "compute_capability": cc,
            "sm_arch": f"sm_{props['major']}{props['minor']}",
            "total_GB": total / 1024 ** 3,
            "free_GB": free / 1024 ** 3,
            "n_devices": cp.cuda.runtime.getDeviceCount(),
            "cupy_version": cp.__version__,
        }
    except Exception as exc:                                # noqa: BLE001
        return {"available": False, "reason": f"device query failed: {exc}"}


def get_namespace(prefer_gpu: bool = True, n_elements: int | None = None):
    """Return ``(xp, on_gpu)`` -- the array module to use, and which it is.

    `n_elements`, when given, applies the transfer-cost threshold, so a
    caller can ask for the GPU and still be handed NumPy for a small array.
    """
    if prefer_gpu and cupy_available():
        if n_elements is None or n_elements >= GPU_MIN_ELEMENTS:
            return _CUPY, True
    return np, False


def should_use_gpu(n_elements: int, prefer_gpu: bool = True) -> bool:
    """Whether a GPU path is worth taking for this many elements."""
    return (prefer_gpu and cupy_available()
            and n_elements >= GPU_MIN_ELEMENTS)


def to_device(a, xp=None):
    """Move `a` onto the device backing `xp` (no copy if already there)."""
    if xp is None or xp is np:
        return to_host(a)
    return xp.asarray(a)


def to_host(a) -> np.ndarray:
    """Bring an array back to the host as a NumPy array."""
    if _CUPY is not None and isinstance(a, _CUPY.ndarray):
        return _CUPY.asnumpy(a)
    return np.asarray(a)


def free_memory() -> None:
    """Release CuPy's cached device blocks.

    CuPy pools allocations and does not return them to the driver, so a
    large intermediate stays resident and a later allocation can fail with
    an out-of-memory error despite the data being dead. Call between large
    frames.
    """
    if cupy_available():
        _CUPY.get_default_memory_pool().free_all_blocks()
        _CUPY.get_default_pinned_memory_pool().free_all_blocks()


# ---------------------------------------------------------------------------
# GPU kernels for the workloads that are actually shaped for a GPU
# ---------------------------------------------------------------------------

def bilinear_lookup(grid, x0: float, xspan: float, n_xcell: int,
                    y0: float, yspan: float, n_ycell: int,
                    x, y, xp=None):
    """Bilinear interpolation on a uniform 2-D grid, device-agnostic.

    This is `IonisationTable.Zbar`'s arithmetic, factored out so the same
    expression serves a 24-element ODE call on the CPU and a
    hundred-million-cell MD frame on the GPU. Both axes are uniform, so
    locating a cell is a division rather than a search -- which also makes
    it branch-free and therefore GPU-friendly.

    `grid` must already live on the device `xp` describes; it is small
    (64x160 doubles = 80 kB) so uploading it once and reusing it across
    frames is the intended pattern.
    """
    if xp is None:
        xp = np
    fx = (x - x0) * xspan
    fy = (y - y0) * yspan
    i = xp.minimum(xp.maximum(fx.astype(xp.intp), 0), n_xcell)
    j = xp.minimum(xp.maximum(fy.astype(xp.intp), 0), n_ycell)
    wx = xp.minimum(xp.maximum(fx - i, 0.0), 1.0)
    wy = xp.minimum(xp.maximum(fy - j, 0.0), 1.0)
    return ((1.0 - wx) * ((1.0 - wy) * grid[i, j] + wy * grid[i, j + 1])
            + wx * ((1.0 - wy) * grid[i + 1, j] + wy * grid[i + 1, j + 1]))


def bin_atoms(pos, vel, lo, cell: float, shape, m_atom: float,
              min_atoms_per_cell: int = 4, prefer_gpu: bool = True):
    """Per-cell count, bulk velocity and thermal energy from particle data.

    The GPU-worthwhile half of `lammps_postprocess.analyse_frame`: every
    atom is independent, the work is scatter-add reductions, and real frames
    carry 10^6-10^8 atoms. Falls back to NumPy below `GPU_MIN_ELEMENTS`.

    Temperature here is the velocity dispersion *about each cell's own bulk
    motion*, never a raw per-atom kinetic energy -- a fast cold lump is not
    hot, and conflating the two fabricates plasma that is not there.

    Returns ``(count, T_K, flat_index)`` as host arrays.
    """
    n_atoms = len(pos)
    xp, on_gpu = get_namespace(prefer_gpu, n_elements=n_atoms * 3)
    from .constants import K_B

    p = to_device(pos, xp)
    v = to_device(vel, xp)
    lo = to_device(np.asarray(lo), xp)
    shp = np.asarray(shape, dtype=np.int64)
    ncell = int(np.prod(shp))

    idx = xp.floor((p - lo) / cell).astype(xp.intp)
    for d in range(3):
        idx[:, d] = xp.minimum(xp.maximum(idx[:, d], 0), int(shp[d]) - 1)
    flat = (idx[:, 0] * int(shp[1]) + idx[:, 1]) * int(shp[2]) + idx[:, 2]

    count = xp.bincount(flat, minlength=ncell).astype(xp.float64)
    safe = xp.maximum(count, 1.0)
    vsum = xp.stack([xp.bincount(flat, weights=v[:, d], minlength=ncell)
                     for d in range(3)], axis=-1)
    vbar = vsum / safe[:, None]

    dv = v - vbar[flat]
    ke = 0.5 * m_atom * xp.sum(dv * dv, axis=-1)
    ke_sum = xp.bincount(flat, weights=ke, minlength=ncell)
    T_K = xp.where(count >= min_atoms_per_cell,
                   (2.0 / 3.0) * ke_sum / safe / K_B, 0.0)

    out = (to_host(count), to_host(T_K), to_host(flat))
    if on_gpu:
        free_memory()
    return out


def accel_selftest(n: int = 4_000_000, seed: int = 0) -> dict:
    """Check that the GPU path agrees with the CPU path on real arrays.

    Run this **first** on any machine where the CUDA paths have not been
    exercised before; this module was written without a working GPU to test
    against. Returns a dict with the measured maximum relative difference
    and the timings, and never raises if no GPU is present.

    Agreement is checked to floating-point tolerance rather than bitwise,
    because CuPy's reduction order differs from NumPy's.
    """
    import time
    info = gpu_info()
    if not info.get("available"):
        return {"ran": False, "reason": info.get("reason", "no GPU")}

    rng = np.random.default_rng(seed)
    grid = rng.random((64, 160))
    x = rng.uniform(0.0, 63.0, n)
    y = rng.uniform(0.0, 159.0, n)

    t0 = time.perf_counter()
    ref = bilinear_lookup(grid, 0.0, 1.0, 62, 0.0, 1.0, 158, x, y, xp=np)
    t_cpu = time.perf_counter() - t0

    cp = _CUPY
    g_d, x_d, y_d = cp.asarray(grid), cp.asarray(x), cp.asarray(y)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    got = bilinear_lookup(g_d, 0.0, 1.0, 62, 0.0, 1.0, 158, x_d, y_d, xp=cp)
    cp.cuda.Stream.null.synchronize()
    t_gpu = time.perf_counter() - t0

    err = float(np.max(np.abs(to_host(got) - ref)))
    free_memory()
    return {"ran": True, "device": info["name"], "n": n,
            "max_abs_error": err, "agrees": err < 1e-12,
            "cpu_s": t_cpu, "gpu_s": t_gpu,
            "speedup": t_cpu / t_gpu if t_gpu > 0 else float("nan")}
