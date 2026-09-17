#!/usr/bin/env python3
"""Measure where the time goes, on this machine.

    python scripts/benchmark.py              # everything
    python scripts/benchmark.py --sweep 24   # 24-point sweep scaling
    python scripts/benchmark.py --gpu        # CUDA paths, if a GPU is present

Why a script rather than a number in the README: the answer depends on the
machine. A 4-core laptop and a 48-core workstation have different bottlenecks,
and the point of the parallel work is that the second should behave very
differently from the first.

What this measures, and what it deliberately does not
-----------------------------------------------------
The single-expansion timing is *not* expected to improve with core count. A
Stage-2 expansion is 44,000 sequential ODE steps on a 24-element state
vector: step n+1 needs step n, so there is nothing to parallelise, and the
arrays are far too small for a GPU kernel launch to pay for itself. If that
number goes down it is because the arithmetic got cheaper, not because more
hardware was used.

What *should* scale is the sweep and the table build.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.simplefilter("ignore")


def _t(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return time.perf_counter() - t0, out


def bench_core() -> None:
    from hvi_emp import get_material
    from hvi_emp.expansion import simulate_expansion
    from hvi_emp.expansion_2t import simulate_expansion_2t
    from hvi_emp.impact import Projectile, simulate_impact
    from hvi_emp.ionization import IonisationTable, get_table

    Fe, Al = get_material("Fe"), get_material("Al")
    print("Ionisation table (64 x 160 Saha solves)")
    ts, _ = _t(IonisationTable, Al, workers=1)
    tp, _ = _t(IonisationTable, Al, workers=None)
    print(f"  serial            {ts * 1e3:8.1f} ms")
    print(f"  all cores         {tp * 1e3:8.1f} ms    {ts / tp:5.2f}x")

    tab = get_table(Al)
    rng = np.random.default_rng(0)
    n24 = 10 ** rng.uniform(20, 28, 24)
    t, _ = _t(lambda: [tab.Zbar(n24, 1.2e4) for _ in range(2000)])
    print(f"\nTable lookup (the 2T inner loop, 24 shells)")
    print(f"  per call          {t / 2000 * 1e6:8.2f} us")

    big = 10 ** rng.uniform(20, 28, 2_000_000)
    t, _ = _t(tab.Zbar, big, 1.2e4)
    print(f"  2e6 points        {t * 1e3:8.1f} ms  "
          f"({2e6 / t / 1e6:.1f} M points/s)")

    print("\nSingle impact (sequential by nature -- should NOT scale)")
    _, imp = _t(simulate_impact, Projectile(Fe, 1e-12, 50e3), Al, warn=False)
    t1, _ = _t(simulate_expansion, imp, t_end=1e-5)
    t2, _ = _t(simulate_expansion_2t, imp, t_end=1e-5)
    print(f"  Stage 2, 1T       {t1 * 1e3:8.1f} ms")
    print(f"  Stage 2, 2T       {t2 * 1e3:8.1f} ms")


def bench_sweep(n_points: int) -> None:
    from hvi_emp.parallel import cpu_count
    from hvi_emp.pipeline import velocity_sweep

    v = np.linspace(40e3, 66e3, n_points)
    print(f"\nVelocity sweep, {n_points} points ({cpu_count()} cores usable)")
    ts, a = _t(velocity_sweep, "Fe", "Al", 1e-12, v, workers=1)
    print(f"  serial            {ts:8.2f} s")
    if cpu_count() > n_points:
        print(f"  note: {n_points} points cannot use more than {n_points} "
              f"workers, so this\n        understates a {cpu_count()}-core "
              f"machine. Use --sweep {cpu_count()} to see it scale.")
    for w in sorted({2, 4, 8, min(cpu_count(), n_points)}):
        if w > cpu_count() or w < 2:
            continue
        tw, b = _t(velocity_sweep, "Fe", "Al", 1e-12, v, workers=w)
        same = np.allclose(a["Q_frozen"], b["Q_frozen"])
        print(f"  {w:2d} workers        {tw:8.2f} s    {ts / tw:5.2f}x    "
              f"{'identical' if same else 'MISMATCH'}")


def bench_volume() -> None:
    from hvi_emp import run_scenario
    from hvi_emp.viz.vti import ImpactVolume

    sc = run_scenario("Fe", "Al", mass=1e-9, velocity=30e3)
    print("\nVolume sampling (.vti frames)")
    for n in (64, 128, 160):
        t0 = time.perf_counter()
        vol = ImpactVolume(sc, shape=(n, n, n))
        t_grid = time.perf_counter() - t0
        t_frame, _ = _t(vol.frame, 1e-7)

        def real(a):
            b = a
            while b.base is not None:
                b = b.base
            return b.nbytes
        coords = sum(real(a) for a in
                     (vol.X, vol.Y, vol.Z, vol.Rxy, vol.Rsph))
        print(f"  {n:3d}^3  grid {t_grid * 1e3:6.1f} ms   frame "
              f"{t_frame * 1e3:7.1f} ms   coords {coords / 1e6:6.1f} MB "
              f"(dense would be {5 * n ** 3 * 8 / 1e6:6.1f} MB)")


def bench_gpu() -> None:
    from hvi_emp.accel import accel_selftest, gpu_info
    print("\nCUDA")
    info = gpu_info()
    if not info.get("available"):
        print(f"  unavailable: {info.get('reason')}")
        return
    print(f"  device            {info['name']} ({info['sm_arch']}, "
          f"{info['total_GB']:.0f} GB)")
    r = accel_selftest()
    print(f"  agreement         max|GPU-CPU| = {r['max_abs_error']:.2e}  "
          f"{'OK' if r['agrees'] else 'DISAGREES'}")
    print(f"  {r['n']:,} lookups  CPU {r['cpu_s'] * 1e3:.1f} ms  "
          f"GPU {r['gpu_s'] * 1e3:.1f} ms   {r['speedup']:.1f}x")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=8,
                    help="points in the velocity-sweep scaling test")
    ap.add_argument("--gpu", action="store_true",
                    help="also exercise the CUDA paths")
    ap.add_argument("--skip-volume", action="store_true")
    args = ap.parse_args()

    print("=" * 68)
    print("hvi_emp benchmark")
    print("=" * 68)
    bench_core()
    bench_sweep(args.sweep)
    if not args.skip_volume:
        bench_volume()
    if args.gpu:
        bench_gpu()
    print("\nA single expansion is a sequential ODE: it does not scale with\n"
          "cores or benefit from a GPU. Sweeps and table builds do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
