# Performance, parallelism and CUDA

```bash
python scripts/benchmark.py            # measure this machine
python scripts/benchmark.py --gpu      # include the CUDA paths
python -m hvi_emp doctor               # what acceleration is available
```

## The short version

**A single impact simulation does not benefit from more cores or from a GPU,
and it never will.** Stage 2 is 44,000 sequential steps of a stiff ODE on a
24-element state vector; step *n+1* needs step *n*. There is no parallelism in
it, and a CUDA kernel launch (~5–10 µs) costs more than the entire right-hand
side. Anyone who tells you otherwise is about to make it slower.

What *did* make a single run faster was arithmetic, not hardware:

| | before | after | |
|---|---|---|---|
| Stage 2, two-temperature | 14.36 s | **2.40 s** | 6.0× |
| table lookup (24 shells) | 141 µs | **17 µs** | 8.3× |

What scales with hardware is everything around the ODE. Both columns are
measured, the second on a 64-core Xeon:

| workload | mechanism | 4 cores | 64 cores |
|---|---|---|---|
| ionisation table build | processes | 3.4× | **12.8×** |
| velocity sweep (8 points) | processes | 4.1× | **7.4×** at 8 workers |
| MD post-processing | CUDA | — | untested |
| `.vti` volume sampling | memory layout | 5× less RAM | 5× less RAM |

The sweep number is the honest one to read carefully: **8 points cannot use
more than 8 workers.** On the 64-core run, 8 workers gave 7.4× and 64 workers
gave 6.79× — *worse*, because the 56 surplus processes each cost a fork and a
fresh interpreter and then had nothing to do. `parallel_map` now caps the
worker count at the item count even when a larger number is passed
explicitly, so `workers=cpu_count()` is no longer a pessimisation on a short
sweep. Scale the sweep, not the pool: 64 points across 64 cores is where a
big machine pays.

The table build scales further (12.8× on 64 cores) because it is 64
independent rows, not 8.

## Where the time went, and why

Profiling a two-temperature expansion put **89% of the runtime inside
`IonisationTable.Zbar`**, in a Python loop issuing **5.68 million scalar
`np.interp` calls**. Two things were wrong:

1. It interpolated **every** density row of the 64-row grid when only the two
   rows bracketing the requested density are ever used — 32× more work than
   needed, all discarded.
2. It did that in a list comprehension, paying NumPy's ~1.3 µs per-call
   dispatch 64 times per invocation to do nanoseconds of arithmetic.

The fix gathers the four bracketing corners by fancy indexing. Both grid axes
are geometric, hence uniform in log, so locating a cell is a division rather
than a binary search — which also makes it branch-free and therefore usable
unchanged on a GPU (`accel.bilinear_lookup`).

`np.clip` then became visible at 445,000 calls: it routes through
`fromnumeric._wrapfunc` and constructs a `numpy.finfo` every time.
`np.minimum(np.maximum(...))` does the same arithmetic with none of that.

**A latent bug fell out of this.** The old vectorised branch collapsed
temperature to `np.min(T)` and applied that single value to every element.
With the scalar `T_e` the 2T solver passes, that was accidentally correct;
with an array `T` it silently evaluated everything at the coldest temperature
present, and disagreed with the scalar path on identical inputs. Temperature
is now interpolated per element. `test_lookup_is_per_element_in_temperature`
pins it.

The same pattern appeared a second time in
`solvers/lammps_postprocess.analyse_frame` — a Python loop over live cells
calling the scalar path once each, which on a real MD frame is tens of
thousands of interpreter round-trips. It is now a single call.

## Parallelism

`hvi_emp.parallel.parallel_map` is an ordered parallel map, used by the table
build and by `velocity_sweep`.

```python
from hvi_emp.pipeline import velocity_sweep
res = velocity_sweep("Fe", "Al", 1e-12, np.linspace(40e3, 66e3, 40))
res = velocity_sweep(..., workers=1)     # force serial
print(res["n_failed"])                   # scenarios that raised
```

Four properties worth knowing:

**Order is preserved.** Completion order is not submission order; results are
placed by index, so a sweep is reproducible run to run.

**Results are bit-identical to serial.** Each item is computed independently
by the same code and there is no reduction whose order could vary. This is
tested, not assumed.

**Failures are recorded, not hidden.** A scenario that raises becomes `None`
(and zero in the sweep arrays) rather than aborting the batch, but the count
comes back in `n_failed`. A quietly-zeroed non-convergence looks exactly like
a real physical result on a log-log plot, which is the worst failure mode
available.

**Pools never nest.** `parallel_map` detects that it is already in a worker
process and runs inline. Without that, a 48-way sweep whose workers each fork
a 48-way table build would ask for 2,304 processes on 48 cores.

Worker functions must be module-level — `multiprocessing` dispatches by
pickling the callable, so a lambda or closure fails at submit time.

## CUDA

`hvi_emp.accel` exposes the array namespace in use, so a routine written once
against `xp` runs on NumPy or CuPy.

```python
from hvi_emp.accel import get_namespace, to_device, to_host
xp, on_gpu = get_namespace(prefer_gpu=True, n_elements=arr.size)
```

Install the wheel matching your CUDA **major** version -- they are not
interchangeable, since each links against that runtime:

| CUDA toolkit | wheel |
|---|---|
| 11.x | `pip install cupy-cuda11x` |
| 12.x | `pip install cupy-cuda12x` |
| 13.x | `pip install cupy-cuda13x` (CuPy ≥ 13.6) |

`python -m hvi_emp doctor` reads `nvcc --version` and names the right one, so
you do not have to check. Everything degrades to NumPy when CuPy or the
driver is absent; nothing requires a GPU.

### Where CUDA actually helps

| workload | size | verdict |
|---|---|---|
| Stage-2 ODE | 24 doubles | **CPU only** — sequential, launch-bound |
| Saha table build | 10,240 pts | **CPU processes** — `brentq` is scalar and branchy |
| MD frame post-processing | 10⁶–10⁸ atoms | **GPU** |
| `.vti` volume sampling | 10⁶–10⁸ voxels | **GPU** |
| batched table lookup | 10⁵+ pts | **GPU** |

`GPU_MIN_ELEMENTS` (10⁶) is the crossover below which the PCIe round trip
costs more than the kernel saves; `get_namespace` returns NumPy below it even
when a GPU is present.

Results agree to floating-point tolerance, **not bitwise** — CuPy's reduction
order differs from NumPy's. Nothing switches a physics result to the GPU
silently; callers opt in.

### The CUDA paths have not been executed

This was developed on a machine with no GPU. The target workstation now has a
working driver (RTX A5000, 24 GB, sm_86, driver 610.57.04, CUDA 13.1) but
CuPy is not installed there yet, so the GPU paths still have not run
anywhere. The CPU paths are tested; the CUDA paths are written against the
documented CuPy API and are structurally simple, but **treat their first run
as unverified**. Before relying on them:

```python
from hvi_emp.accel import accel_selftest
print(accel_selftest())    # GPU vs CPU agreement on 4M real lookups
```

`test_gpu_agrees_with_cpu` runs the same check but skips where there is no
device, so a green suite on a CPU machine says nothing about CUDA.

## Volume sampling memory

`ImpactVolume` built five dense coordinate grids — X, Y, Z, Rxy and Rsph — at
5 × nx × ny × nz doubles. At 256³ that is **671 MB of coordinates** before a
single field is sampled, and every field expression then streams all of it
through cache.

Four of the five do not vary in all three axes. They are now computed sparse
and `broadcast_to` the full shape, which returns a *view*: the arrays still
report shape `(nx, ny, nz)` and index identically, while sharing a small
buffer. Only `Rsph` is genuinely three-dimensional. 671 MB → 134 MB.

The public shape of `vol.X`/`Y`/`Z`/`Rxy` is deliberately unchanged. Making
them sparse would have been faster still, but callers and tests index them as
full 3-D arrays, and quietly changing that is the same class of contract
break as returning a different meaning for the same field.

The views are read-only, which is correct — nothing writes to a coordinate
array — and would fail loudly if anything tried.

## Multi-GPU solver runs

For the external solvers, `idefix_stage2.gpu_plan` sizes a run:

```python
from hvi_emp.solvers.idefix_stage2 import fletcher_idefix_config, gpu_plan
plan = gpu_plan(fletcher_idefix_config("fig6_parallel"), "RTX5090", n_gpu=4)
print(plan["command"])       # mpirun -np 4 ... ./idefix -dec 4 1 1
print(plan["kokkos_arch"])   # BLACKWELL120
```

`decompose` splits ranks across axes to keep subdomains as cubic as possible
— MPI halo traffic scales with subdomain *surface* area, so a slab
decomposition of a cube moves several times the data a cubic one does — and
refuses rank counts that cannot divide the grid rather than emitting a command
Idefix will reject.

Two things it will warn about:

* **Memory.** ~400 B/cell for Idefix MHD, against 75% of the card's memory
  (the CUDA context, MPI buffers and any attached display take the rest). An
  out-of-memory abort happens hours in, not at startup.
* **CUDA-aware MPI.** Without it every halo exchange round-trips through host
  memory and multi-GPU is *slower* than one GPU. Check with
  `ompi_info --parsable --all | grep mpi_built_with_cuda_support`.

`gpu_plan()` detects the card from the driver when you do not name one, and
**refuses to guess silently**: an unrecognised card is flagged rather than
given a default memory budget. Sizing a run for a 32 GB card on a 24 GB one
produces a plan that looks fine and then dies of out-of-memory hours in.

RTX 50-series cards are `Kokkos_ARCH_BLACKWELL120` and need CUDA ≥ 12.8 with
Kokkos ≥ 4.5; an older Kokkos configures happily and then emits code the card
cannot run. The professional Ampere workstation cards (RTX A4000/A4500/A5000/
A5500/A6000) are `AMPERE86`.

## What was not done

**Numba/JIT on the ODE right-hand side.** With `Zbar` fixed, the remaining 2T
cost is ~44,000 × ~10 small NumPy operations, which is dominated by
per-operation dispatch rather than arithmetic. A JIT would plausibly give
another 5–10×. It was not done because it adds a compiled dependency and a
second code path that could silently diverge from the reference — and because
sweeps, which are the common case for real work, already parallelise.

**GPU Saha solves.** `brentq` is scalar, iterative and branch-heavy: the wrong
shape for SIMT. Multiprocessing is the right tool and already gets most of it.
