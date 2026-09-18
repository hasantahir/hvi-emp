# Changelog

## Note on history before this file

Everything up to and including **2026.09.17-b14** predates version control.
The project was developed and shipped as dated zip archives, so there are no
commits for that work — the first commit in this repository is the b14 tree
as a single snapshot.

The b-series entries below are reconstructed from `BUILD.txt` and the fixes
they record. They are a record of *what changed*, not a substitute for the
diffs, which no longer exist. From b14 onwards, use `git log`.

---

## Unreleased

### Composite render: black frames

Reported from a real `pvbatch` run — text overlays drew, geometry did not.
Two independent causes, found in that order:

- **The `.pvd` index was missing.** 111 frames on disk, no collection. The
  composite reads `.pvd`, so every chapter was skipped before the camera
  mattered. `scripts/repair_pvd.py` rebuilds the index from frames already
  present; times are recomputed through the writer's own call and verified
  bit-exact against an original. The frames are never rewritten.
- **The scene itself.** `ResetCamera()` now runs before any camera move, so
  a stale clipping range cannot cut a chapter away across nine decades of
  scale; representation is chosen from `GetDataClassName()` rather than
  forcing `Volume` on `.vtp` point data; the handover card no longer sits on
  the banner.

Caught while building the repair, and worth recording: the first version
ignored the impact scene's `t_end` clamp and produced 1e-5 s where the
writer used 9.22e-9 s. Frames would have rendered with a silently wrong
time axis — the exact failure the tool exists to prevent.

### Tooling

- `scripts/pipeline.sh` — run and render end to end, from Hasan's workflow
  script. `--render-only` reuses solver data (verified: no stage re-runs,
  zero `.vti` rewritten). Offers the `.pvd` repair automatically rather than
  telling you to re-run a solver.
- `scripts/check_render_data.py` — distinguishes "the scene is wrong" from
  "the data is empty" in under a second, without ParaView.
- `scripts/test_m2c.py` — staged M2C verification.
- `.github/workflows/tests.yml` — CI on 3.10 and 3.12, no optional solvers.

### Fixed

- `requirements.txt` omitted PyYAML, needed at import time.
- Six tests failed rather than skipped without LAMMPS.


## 2026.09.17-b14 — first commit

Snapshot of the tree as shipped in `hvi_emp_2026-09-17_b14.zip`
(sha256 `66e563b0c5f4a145c84b2d83ee9f87d139f8af51b8e7bee7b1733e125899caa2`).

### Environment and installation

- Package manager detection prefers micromamba → mamba → conda, with
  **conda solver bands**: ≥ 23.10 already defaults to libmamba and needs
  nothing; 22.11–23.9 can opt in; below 22.11 the `solver` config key does
  not exist and the advice everyone repeats fails outright.
- `--bootstrap-mamba` fetches a standalone micromamba for machines with no
  usable conda.
- Shipped scripts carry the execute bit. Builds b4 and earlier did not, so
  the documented invocation gave "Permission denied".
- `doctor` no longer suggests `spack` on machines without it.
- `--submit` warns up front when there is no `sbatch`, instead of reporting
  four stages as "blocked" for a reason buried in per-stage text.

### MPI

- `_preload_mpi` reads the required soname from the installed LAMMPS rather
  than assuming MPICH's `libmpi.so.12`, and **refuses to pre-load when a
  different MPI family is already mapped**. Forcing MPICH symbols into a
  process holding Open MPI communicators produced
  `Invalid communicator` — which reads as a LAMMPS fault and is not one.
- LAMMPS is probed in a subprocess, so an `MPI_Abort` cannot terminate
  `doctor` and cost the whole report.
- `mpi_conflict()` names the mismatched families and the fix.

### LAMMPS

- Work around an upstream crash: `lammps/__init__.py` parses its own
  packaging metadata as a date (`strptime(v, "%Y.%m.%d")`), but ships a
  four-component version, so `import lammps` raises `ValueError` on
  conda-forge builds. The PyPI wheel hardcodes `__version__` and is immune.

### M2C

- **`AtomicData` is generated** next to `input.st`. M2C ships none, so decks
  referencing it aborted at startup. Files are numeric-only on purpose:
  M2C's reader breaks only on EOF, so a `#` sets failbit and it appends
  **1000 zeros** instead of erroring.
- `PartitionFunctionEvaluation = OnTheFly`. `CubicSplineInterpolation`
  asserts (`expmin<expmax`) on ground-state-only data, where the partition
  function is constant in temperature and there is nothing to interpolate.
  This is a workaround for impoverished level data, not a preference — the
  paper's intended fast path is tabulation plus cubic B-spline.
- `results/` is created with the deck; M2C does not mkdir it.
- `check_grammar` scans all `.h`/`.cpp` (the ionisation assigner is not in
  `IoData.cpp`) and verifies enum **values**, not just keys. It previously
  reported "133 of 133 keywords found" for a deck that then aborted.
- `find_m2c` prefers a built tree over a source-only one, and explains
  `M2C_HOME` pointed at the binary rather than reporting "not found".
- Generated commands use `${M2C_HOME:?...}`, so an unset variable aborts
  with a readable message instead of launching the literal `/m2c`.
- **Cost model corrected against a real run.** It had been wrong in both
  directions, and the errors partly cancelled:
  `cell_count` assumed a uniform fine mesh (16× high: 5.76M vs 354k cells)
  while the rate assumed 2 µs/cell-step (593× low). 3-D at 2-D resolution
  is ~461M cells / 86 GB, not 13.8 billion / 2.6 TB.
  Cost is dominated by the non-ideal Saha solver, not the hydro.
- `scripts/test_m2c.py`: staged verification — grammar, standalone Saha
  solver, M2C's own shipped test, smoke, physics vs the reduced chain.

### Known open items

- Cost model still needs a third correction: the measured rate was taken
  from the startup transient, and the CFL estimate is ~19× optimistic.
- Ionisation ladder truncates at 3 stages; a real run reported the Ebeling
  depression exceeding the ionisation energy (pressure ionisation).
- EOS validity ceiling above 20 km/s.
- CUDA paths have never executed anywhere.
