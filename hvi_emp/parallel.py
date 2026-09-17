"""Process-level parallelism for the parts of the framework that have it.

Where parallelism actually helps here, and where it does not
------------------------------------------------------------
Profiling a two-temperature expansion says the work is 44,000 sequential
evaluations of a 24-element ODE right-hand side. That is inherently serial --
step *n+1* needs step *n* -- and each step touches a few hundred bytes. Threads
would contend, and a GPU kernel launch (~5-10 us) costs more than the entire
right-hand side. **A single expansion cannot be parallelised**, and pretending
otherwise would make it slower.

What *is* parallel is everything above and below that loop:

* **Saha table construction.** 64 x 160 = 10,240 completely independent
  nonlinear solves, ~3 s serially. This is the framework's dominant one-off
  cost and it scales almost perfectly.
* **Parameter sweeps.** Velocity series, sensitivity studies, the Fletcher
  replication grid: each scenario is independent. This is where a 48-core
  workstation actually pays, and it is the common case for real work.

So this module provides one thing -- an ordered parallel map -- and the two
call sites that use it.

Determinism
-----------
`parallel_map` preserves input order regardless of completion order, so
results are reproducible run to run. Floating-point results are identical to
the serial path because each work item is computed independently by the same
code; there is no reduction whose order could vary.

Failure handling
----------------
A worker that raises does not kill the batch. The exception is captured and
re-raised in the parent when that item's result is read, or -- with
``on_error="skip"`` -- recorded as ``None`` so a sweep can survive a single
non-converging point. Silent partial failure is the thing to avoid: the count
of failures is always returned.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Sequence

__all__ = ["cpu_count", "default_workers", "parallel_map", "ParallelResult",
           "in_worker_process"]


def in_worker_process() -> bool:
    """True if this interpreter is already a multiprocessing child.

    Nested pools are the classic way to turn a parallel program into a slow
    one: a 48-way sweep whose workers each fork a 48-way table build asks
    for 2,304 processes on 48 cores, and the machine spends its time
    context-switching and running out of file descriptors.

    Every `parallel_map` checks this and runs inline when it is true, so
    parallelism is applied at the outermost level that has work and nowhere
    below it. Callers do not have to reason about it.
    """
    try:
        return mp.parent_process() is not None
    except AttributeError:                      # Python < 3.8
        return mp.current_process().name != "MainProcess"


def cpu_count() -> int:
    """Cores this process may actually use.

    Three sources, in decreasing authority:

    1. ``$HVI_EMP_WORKERS`` -- set by the generated job script from
       ``cpus_per_task``. Explicit and wins.
    2. ``$SLURM_CPUS_PER_TASK`` -- what the scheduler allocated. A job that
       requested 4 CPUs and then forks 48 workers gets throttled or killed,
       and steals from whoever shares the node. `os.cpu_count()` reports the
       *node's* cores, which on a shared cluster is the wrong number by a
       factor of ten.
    3. ``sched_getaffinity`` -- the cgroup/taskset mask, correct inside a
       container.
    """
    for var in ("HVI_EMP_WORKERS", "SLURM_CPUS_PER_TASK"):
        raw = os.environ.get(var)
        if raw:
            try:
                n = int(raw)
                if n >= 1:
                    return n
            except ValueError:
                pass                            # malformed: fall through
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                      # not Linux
        return max(1, os.cpu_count() or 1)


def default_workers(n_items: int, cap: int | None = None) -> int:
    """How many processes to use for `n_items` pieces of work.

    Never more workers than items (idle processes still cost a fork and an
    interpreter), and never more than the available cores.
    """
    w = min(cpu_count(), max(1, n_items))
    if cap is not None:
        w = min(w, max(1, cap))
    return w


class ParallelResult(list):
    """Results in input order, plus a record of what failed.

    Subclasses `list` so it can be used directly as the results sequence,
    while carrying `errors` for callers that want to know.
    """

    def __init__(self, values: Iterable, errors: dict[int, BaseException]):
        super().__init__(values)
        self.errors = errors

    @property
    def n_failed(self) -> int:
        return len(self.errors)

    def raise_if_failed(self) -> "ParallelResult":
        """Re-raise the first failure, with its item index attached."""
        if self.errors:
            i = min(self.errors)
            raise RuntimeError(
                f"{len(self.errors)} of {len(self)} work items failed; "
                f"first was item {i}") from self.errors[i]
        return self


def parallel_map(fn: Callable[[Any], Any],
                 items: Sequence,
                 workers: int | None = None,
                 *,
                 on_error: str = "raise",
                 min_items: int = 4,
                 progress: Callable[[int, int], None] | None = None
                 ) -> ParallelResult:
    """Ordered parallel map over `items`.

    Parameters
    ----------
    fn
        Must be importable by name in a child process -- a module-level
        function, not a lambda or a closure. This is a hard constraint of
        `multiprocessing`'s pickle-based dispatch, not a style preference;
        a closure fails at submit time with a confusing pickling error.
    workers
        Process count. ``None`` picks `default_workers`. ``1`` runs inline
        with no pool at all, which keeps tracebacks intact and is what the
        test suite and any nested call should use.
    on_error : {"raise", "skip"}
        ``"raise"`` propagates the first failure once the batch completes.
        ``"skip"`` records ``None`` for failed items and carries on.
    min_items
        Below this many items, run serially. Spawning a pool costs a fork
        and a fresh interpreter per worker (~0.3-0.5 s total); for three
        scenarios that is pure loss.

    Returns
    -------
    ParallelResult
        A list in input order, with `.errors` mapping item index to the
        exception raised.
    """
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got "
                         f"{on_error!r}")
    items = list(items)
    n = len(items)
    errors: dict[int, BaseException] = {}
    if n == 0:
        return ParallelResult([], errors)

    # Cap at the item count even when the caller names a number.
    #
    # Measured on a 64-core machine with an 8-point sweep: 8 workers gave
    # 7.39x, 64 workers gave 6.79x. The extra 56 processes each cost a fork
    # and a fresh interpreter, do nothing, and the speedup goes *down*.
    # `default_workers` already caps; an explicit count used not to, so
    # `workers=cpu_count()` -- the obvious thing to write -- was a
    # pessimisation on any sweep shorter than the machine.
    w = default_workers(n) if workers is None else min(max(1, int(workers)), n)
    if in_worker_process():
        w = 1                                   # never nest pools
    if w == 1 or n < min_items:
        out = []
        for k, it in enumerate(items):
            try:
                out.append(fn(it))
            except Exception as exc:             # noqa: BLE001
                errors[k] = exc
                out.append(None)
            if progress is not None:
                progress(k + 1, n)
        res = ParallelResult(out, errors)
        return res if on_error == "skip" else res.raise_if_failed()

    out = [None] * n
    with ProcessPoolExecutor(max_workers=w) as pool:
        futures = {pool.submit(fn, it): k for k, it in enumerate(items)}
        done = 0
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                out[k] = fut.result()
            except Exception as exc:             # noqa: BLE001
                errors[k] = exc
            done += 1
            if progress is not None:
                progress(done, n)

    res = ParallelResult(out, errors)
    return res if on_error == "skip" else res.raise_if_failed()
