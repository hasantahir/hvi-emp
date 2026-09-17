"""Tests for the parallel and CUDA machinery.

The theme is **equivalence, not speed**. A benchmark belongs in
`scripts/benchmark.py`, where a slow machine makes it slow rather than red.
What must never change is the *answer*: parallel must equal serial, the fast
lookup must equal the reference interpolation, and the GPU must equal the CPU.

Every test here corresponds to a way the optimisation could have been wrong.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp import get_material                                   # noqa: E402
from hvi_emp.accel import (bilinear_lookup, bin_atoms,             # noqa: E402
                           cupy_available, get_namespace, gpu_info,
                           should_use_gpu, to_host)
from hvi_emp.ionization import IonisationTable, get_table          # noqa: E402
from hvi_emp.parallel import (ParallelResult, cpu_count,           # noqa: E402
                              default_workers, parallel_map)

warnings.simplefilter("ignore")


# --- helpers that must be importable by a child process --------------------

def _square(x):
    return x * x


def _fail_on_three(x):
    if x == 3:
        raise ValueError("three is bad")
    return x * 10


# ---------------------------------------------------------------------------
# parallel_map
# ---------------------------------------------------------------------------

def test_parallel_map_preserves_order():
    """Completion order is not submission order; results must not shuffle."""
    items = list(range(64))
    out = parallel_map(_square, items, workers=4)
    assert list(out) == [i * i for i in items]


def test_parallel_map_matches_serial_exactly():
    items = list(range(40))
    assert list(parallel_map(_square, items, workers=1)) == \
        list(parallel_map(_square, items, workers=4))


def test_parallel_map_skip_records_failures_rather_than_hiding_them():
    """A failed item must be visible, not silently a zero.

    A sweep that quietly reports zero charge for a non-converged velocity
    looks exactly like a real physical result, which is the worst possible
    failure mode for a plot.
    """
    out = parallel_map(_fail_on_three, [1, 2, 3, 4, 5], workers=1,
                       on_error="skip")
    assert out[2] is None and out.n_failed == 1
    assert list(out) == [10, 20, None, 40, 50]
    assert isinstance(out.errors[2], ValueError)


def test_parallel_map_raise_propagates():
    with pytest.raises(RuntimeError, match="failed"):
        parallel_map(_fail_on_three, [1, 2, 3], workers=1, on_error="raise")


def test_parallel_map_rejects_unknown_error_policy():
    with pytest.raises(ValueError, match="on_error"):
        parallel_map(_square, [1, 2], on_error="ignore")


def test_parallel_map_handles_empty_input():
    out = parallel_map(_square, [])
    assert list(out) == [] and out.n_failed == 0


def test_default_workers_never_exceeds_items_or_cores():
    assert default_workers(1) == 1
    assert default_workers(3) <= min(3, cpu_count())
    assert default_workers(10_000) <= cpu_count()
    assert default_workers(10_000, cap=2) == 2


def test_parallel_result_is_a_list():
    r = ParallelResult([1, 2, 3], {})
    assert r[0] == 1 and len(r) == 3 and r.n_failed == 0


# ---------------------------------------------------------------------------
# The vectorised table lookup
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def table():
    return get_table(get_material("Al"))


def _interp_reference(t, n_h, T):
    """The pre-optimisation implementation: bilinear via `np.interp`."""
    ln = np.log10(max(float(n_h), 10.0 ** t.logn[0]))
    lt = np.clip(np.log10(max(float(T), 1.0)), t.logT[0], t.logT[-1])
    i = int(np.clip(np.searchsorted(t.logn, ln) - 1, 0, len(t.logn) - 2))
    w = float(np.clip((ln - t.logn[i]) / (t.logn[i + 1] - t.logn[i]), 0, 1))
    z0 = np.interp(lt, t.logT, t.Zbar_grid[i])
    z1 = np.interp(lt, t.logT, t.Zbar_grid[i + 1])
    return float((1 - w) * z0 + w * z1)


def test_vectorised_lookup_matches_the_interp_reference(table):
    """The optimisation must not have moved any number.

    The replacement dropped `np.interp` for direct corner gathering and a
    uniform-grid division instead of a binary search. All three are only
    valid because the grids are geometric; this pins that they agree.
    """
    rng = np.random.default_rng(0)
    ns = 10 ** rng.uniform(14, 31, 3000)
    Ts = 10 ** rng.uniform(2, 6.6, 3000)
    got = table.Zbar(ns, Ts)
    ref = np.array([_interp_reference(table, n, T) for n, T in zip(ns, Ts)])
    assert np.max(np.abs(got - ref)) < 1e-12


def test_lookup_is_per_element_in_temperature(table):
    """Regression: the old vectorised branch collapsed T to `np.min(T)`.

    With the scalar T_e the 2T solver passes that was accidentally correct.
    With an array it evaluated every point at the coldest temperature
    present -- silently, and only in the vectorised branch, so the scalar
    path disagreed with the vector path on the same inputs.
    """
    n = np.full(4, 1e26)
    T = np.array([3e3, 1e4, 3e4, 1e5])
    got = table.Zbar(n, T)
    one = np.array([table.Zbar(1e26, float(t)) for t in T])
    assert np.allclose(got, one)
    assert got[0] != got[-1], "temperature had no effect: collapsed again"


def test_lookup_shapes_and_scalar_type(table):
    assert isinstance(table.Zbar(1e26, 1e4), float)
    assert table.Zbar(np.full(7, 1e26), 1e4).shape == (7,)
    assert table.Zbar(1e26, np.full(5, 1e4)).shape == (5,)
    assert table.Zbar(np.full((3, 4), 1e26), 1e4).shape == (3, 4)


def test_lookup_clamps_outside_the_grid(table):
    """Out-of-range must clamp to the edge, as `np.interp` did."""
    assert table.Zbar(1e5, 1e4) == pytest.approx(table.Zbar(1e14, 1e4))
    assert table.Zbar(1e40, 1e4) == pytest.approx(table.Zbar(1e31, 1e4))
    assert table.Zbar(1e26, 1e9) == pytest.approx(table.Zbar(1e26, 5e6))
    assert table.Zbar(1e26, 1.0) >= 0.0


# ---------------------------------------------------------------------------
# Parallel table construction
# ---------------------------------------------------------------------------

def test_parallel_table_build_is_bit_identical():
    """Rows are independent, so parallelism must change nothing at all."""
    Al = get_material("Al")
    a = IonisationTable(Al, n_pts=12, T_pts=20, workers=1)
    b = IonisationTable(Al, n_pts=12, T_pts=20, workers=3)
    assert np.array_equal(a.Zbar_grid, b.Zbar_grid)
    assert np.array_equal(a.u_grid, b.u_grid)


# ---------------------------------------------------------------------------
# The device-agnostic kernels, on the CPU path
# ---------------------------------------------------------------------------

def test_bilinear_lookup_reproduces_the_table(table):
    """`accel.bilinear_lookup` is the table's own arithmetic, factored out.

    If these ever diverge, a GPU post-processing run would disagree with the
    CPU chain while both looked plausible.
    """
    rng = np.random.default_rng(1)
    ns = 10 ** rng.uniform(14, 31, 2000)
    Ts = 10 ** rng.uniform(2, 6.6, 2000)
    ref = table.Zbar(ns, Ts)
    got = bilinear_lookup(
        table.Zbar_grid, table.logn[0], table._n_span, table._n_cell,
        table.logT[0], table._T_span, table._T_cell,
        np.log10(np.maximum(ns, table._n_floor)),
        np.log10(np.maximum(Ts, 1.0)))
    assert np.array_equal(got, ref)


def test_bin_atoms_conserves_atoms_and_is_frame_invariant():
    """Temperature is dispersion about the cell's bulk motion.

    Boosting every atom by a constant velocity must not change the
    temperature: a fast cold lump is not hot. Getting this wrong is the
    standard way to fabricate plasma that is not there.
    """
    rng = np.random.default_rng(2)
    n = 20_000
    pos = rng.uniform(0, 2e-8, (n, 3))
    vel = rng.normal(0, 3e3, (n, 3))
    lo, cell, shape = pos.min(axis=0), 1e-9, (21, 21, 21)

    c0, T0, _ = bin_atoms(pos, vel, lo, cell, shape, 3.0e-26)
    assert int(c0.sum()) == n
    c1, T1, _ = bin_atoms(pos, vel + 5e4, lo, cell, shape, 3.0e-26)
    assert np.allclose(T0, T1), "temperature changed under a Galilean boost"
    assert np.all(T0 >= 0.0)


def test_get_namespace_respects_the_transfer_threshold():
    """A small array must stay on the CPU even when a GPU exists."""
    xp, on_gpu = get_namespace(prefer_gpu=True, n_elements=10)
    assert xp is np and not on_gpu
    assert not should_use_gpu(10)


def test_accel_reports_honestly_without_a_gpu():
    info = gpu_info()
    assert isinstance(info["available"], bool)
    if not info["available"]:
        assert info["reason"], "unavailable GPU must say why"
    assert cupy_available() == info["available"]


def test_to_host_is_a_noop_for_numpy():
    a = np.arange(5.0)
    assert to_host(a) is a or np.array_equal(to_host(a), a)


@pytest.mark.skipif(not cupy_available(), reason="no CUDA device")
def test_gpu_agrees_with_cpu():
    """Runs only where a GPU exists; the CUDA paths ship unverified."""
    from hvi_emp.accel import accel_selftest
    r = accel_selftest(n=1_000_000)
    assert r["ran"] and r["agrees"], r


# ---------------------------------------------------------------------------
# Idefix multi-GPU planning
# ---------------------------------------------------------------------------

def test_decompose_is_exact_and_balanced():
    from hvi_emp.solvers.idefix_stage2 import decompose
    grid = (512, 256, 1)
    for n in (1, 2, 4, 8, 16, 32, 64):
        dec = decompose(n, grid)
        assert int(np.prod(dec)) == n, f"{n} ranks -> {dec}"
        for ax in range(3):
            assert grid[ax] % dec[ax] == 0, "subdomain does not divide evenly"


def test_decompose_refuses_impossible_rank_counts():
    from hvi_emp.solvers.idefix_stage2 import decompose
    with pytest.raises(ValueError, match="cannot place"):
        decompose(7, (512, 256, 1))


def test_gpu_plan_flags_a_grid_too_big_for_the_card():
    from hvi_emp.solvers.idefix_stage2 import gpu_plan
    from hvi_emp.solvers.idefix_stage2 import fletcher_idefix_config
    cfg = fletcher_idefix_config("fig6_parallel")
    # 134M cells at ~400 B/cell is ~54 GB: well past the 24 GB a 32 GB card
    # can safely use, so a single-device plan must say so rather than let the
    # run die hours in with an out-of-memory abort.
    cfg.n_r, cfg.n_theta = 16384, 8192
    plan = gpu_plan(cfg, "RTX5090", n_gpu=1)
    assert plan["kokkos_arch"] == "BLACKWELL120"
    assert plan["memory_GB_per_rank"] > plan["memory_budget_GB"]
    assert any("exceeds" in w for w in plan["notes"]), plan["notes"]
    # and it must say how many devices would actually fit
    assert any("GPUs" in w for w in plan["notes"])


def test_gpu_plan_single_device_needs_no_mpi():
    from hvi_emp.solvers.idefix_stage2 import (fletcher_idefix_config,
                                               gpu_plan)
    plan = gpu_plan(fletcher_idefix_config("fig6_parallel"), "RTX5090", 1)
    assert plan["command"] == "./idefix"
    assert plan["decomposition"] == (1, 1, 1)


# ---------------------------------------------------------------------------
# Regressions from a real 64-core / RTX A5000 machine
# ---------------------------------------------------------------------------

def test_explicit_worker_count_is_capped_at_the_item_count():
    """Asking for more workers than items must not fork them.

    Measured on a 64-core box: an 8-point sweep got 7.39x with 8 workers and
    6.79x with 64 -- the 56 surplus processes each cost a fork and a fresh
    interpreter and had nothing to do. `default_workers` capped; an explicit
    count did not, so `workers=cpu_count()` was a pessimisation on any sweep
    shorter than the machine.
    """
    import hvi_emp.parallel as P

    seen = {}
    real = P.ProcessPoolExecutor

    class Spy(real):                      # type: ignore[misc,valid-type]
        def __init__(self, max_workers=None, **kw):
            seen["max_workers"] = max_workers
            super().__init__(max_workers=max_workers, **kw)

    P.ProcessPoolExecutor = Spy
    try:
        out = P.parallel_map(_square, list(range(8)), workers=64)
    finally:
        P.ProcessPoolExecutor = real
    assert list(out) == [i * i for i in range(8)]
    assert seen["max_workers"] == 8, \
        f"forked {seen['max_workers']} workers for 8 items"


def test_gpu_tables_know_the_professional_ampere_cards():
    """RTX A5000 (0x2231, sm_86, 24 GB) was absent from every table."""
    from hvi_emp.doctor import GPU_DEVICE_IDS
    from hvi_emp.solvers.idefix_stage2 import GPU_MEMORY_GB, KOKKOS_ARCH

    name, cc, kokkos = GPU_DEVICE_IDS["0x2231"]
    assert "A5000" in name and cc == "86" and kokkos == "AMPERE86"
    for card in ("A5000", "A4000", "A6000"):
        assert KOKKOS_ARCH[card] == "AMPERE86"
        assert GPU_MEMORY_GB[card] > 0
    assert GPU_MEMORY_GB["A5000"] == 24, "A5000 is a 24 GB card"


def test_nvidia_smi_beats_the_pci_table_when_the_driver_works(monkeypatch):
    """Regression: the fallback table won even with a working driver.

    On a machine whose card was not in the table, this printed "card not in
    this table. Once the driver works: ..." while the driver was working
    fine and nvidia-smi could have answered immediately.
    """
    import hvi_emp.doctor as D

    monkeypatch.setattr(D, "_run", lambda cmd, timeout=10: (
        (True, "NVIDIA RTX A5000, 8.6")
        if "nvidia-smi" in cmd[0] else (False, "")))
    flags = D.gpu_build_flags()
    assert flags is not None and not flags["unknown"]
    assert flags["compute_capability"] == "86"
    assert flags["kokkos_arch"] == "AMPERE86"
    assert "AMPERE86" in flags["idefix"]


def test_unknown_card_with_no_driver_still_explains_itself(monkeypatch):
    import hvi_emp.doctor as D
    monkeypatch.setattr(D, "_run", lambda *a, **k: (False, ""))
    out = D.gpu_build_flags("0xdead")
    assert out["unknown"] and "nvidia-smi" in out["hint"]


@pytest.mark.parametrize("release,expected", [
    ("release 11.8, V11.8.89", "cupy-cuda11x"),
    ("release 12.4, V12.4.131", "cupy-cuda12x"),
    ("release 13.1, V13.1.115", "cupy-cuda13x"),
])
def test_cupy_wheel_matches_the_installed_cuda_major(monkeypatch, release,
                                                     expected):
    """A CUDA 13.1 machine was told to install the CUDA 12 wheel.

    The wheels link against a specific runtime and are not interchangeable.
    """
    import hvi_emp.doctor as D
    monkeypatch.setattr(D.shutil, "which",
                        lambda n: "/usr/bin/nvcc" if n == "nvcc" else None)
    monkeypatch.setattr(D, "_run", lambda *a, **k: (True, release))
    assert D.cupy_package() == expected


def test_gpu_plan_refuses_to_assume_a_memory_budget():
    """An unrecognised card must be flagged, not given a silent default.

    Sizing a run for 32 GB on a 24 GB card yields a plan that looks fine and
    then dies of out-of-memory hours in.
    """
    from hvi_emp.solvers.idefix_stage2 import (fletcher_idefix_config,
                                               gpu_plan)
    cfg = fletcher_idefix_config("fig6_parallel")
    plan = gpu_plan(cfg, "SomeNewCard", n_gpu=1)
    assert plan["kokkos_arch"] is None
    assert any("GPU_MEMORY_GB" in n for n in plan["notes"])
    assert any("KOKKOS_ARCH" in n for n in plan["notes"])
    # a known card must be silent about both
    quiet = gpu_plan(cfg, "A5000", n_gpu=1)
    assert quiet["kokkos_arch"] == "AMPERE86"
    assert not any("not in" in n for n in quiet["notes"])


def test_every_kokkos_arch_has_a_minimum_cuda_toolkit():
    """Complements the GPU_DEVICE_IDS check: cover the derived map too.

    `_kokkos_arch_for` maps compute capabilities that no table entry uses
    (72, 87), and adding a card for one of them without a toolkit minimum
    would produce "cuda_min: ?" in the build advice.
    """
    import hvi_emp.doctor as D
    for cc in ("70", "72", "75", "80", "86", "87", "89", "90", "100", "120"):
        assert cc in D.CUDA_MIN_TOOLKIT, f"sm_{cc} has no minimum toolkit"
        assert D._kokkos_arch_for(cc).startswith(
            ("VOLTA", "TURING", "AMPERE", "ADA", "HOPPER", "BLACKWELL"))
