# Installing the production solvers

M2C, LAMMPS, WarpX, Idefix, PIConGPU, OpenMHD, iSALE and ParaView — what each
one actually needs on a Linux workstation, how to prove it worked, and what
goes wrong.

**None of this is required.** The framework runs its whole four-stage chain on
NumPy and SciPy alone, and every bridge *generates complete, runnable input
decks whether or not its solver is installed*. You install these to execute
the decks, not to write them. Do not let a failing WarpX build stop you doing
science.

---

## 0. Before anything else

```bash
python -m hvi_emp doctor
```

`doctor` is the source of truth for this document: it probes the hardware,
tests each solver by importing or locating it, and prints the specific next
command for whatever is missing. Every install below ends by re-running it.
If a claim here disagrees with `doctor` on your machine, believe `doctor`.

### A correction, stated plainly

Earlier versions of these docs told you to run `pip install pywarpx`. **That
package does not exist** — PyPI returns 404 for `pywarpx`, `warpx`,
`warpx-pybind` and `amrex` alike. The `pywarpx` *module* is real, but it comes
from a conda-forge, Spack or source build of WarpX, never from pip's index.
§2 has the routes that work. WarpX's own docs are explicit that pre-compiled
PyPI wheels are a future plan, not a present one.

### The scripted route

Most of this document is tedium a script can do. There is one:

```bash
scripts/install_solvers.sh --dry-run all     # see what it would do first
scripts/install_solvers.sh idefix openmhd    # or just what you want
scripts/install_solvers.sh all
source ~/.hvi_emp_solvers.env                # then re-check
python -m hvi_emp doctor
```

It is idempotent (re-running after a failure costs only the failed step),
verifies each install, and writes `IDEFIX_DIR`, `OPENMHD`, `PICSRC` to a file
you source rather than editing your shell rc behind your back.

**It never uses sudo and never touches apt.** Two things it therefore reports
rather than attempts: the NVIDIA driver (needs root and a reboot) and iSALE
(needs a licence a human approves, and in practice has not). The rest of this
document is the manual version, and the reference for when the script's
summary says something failed.

### The recommended order

Earlier versions of this list opened with "apply for an iSALE licence and wait
days". That is gone: the requests went unanswered, and **M2C covers the same
stage with no gatekeeper.** Nothing in this list now blocks on anyone's
approval.

| # | thing | effort | blocking? |
|---|---|---|---|
| 1 | **M2C** (Stage 1: 3-D, tabular EOS, non-ideal Saha) | 30 min if PETSc is packaged | no |
| 2 | LAMMPS | 5 min | no |
| 3 | ParaView | 5 min | no |
| 4 | WarpX (CPU) | 20 min | no |
| 5 | Idefix (Stage 2 MHD, GPU) | 20 min | no |
| 6 | OpenMHD | an hour | no |
| 7 | PIConGPU (Stage 3 PIC, GPU) | half a day | no |
| 8 | GPU builds (WarpX CUDA, OpenMHD CUDA Fortran, LAMMPS Kokkos) | half a day | no — and often not worth it, see §9 |
| — | iSALE | licence application, indefinite | optional, and not recommended as a starting point |

If you only install one thing, make it M2C. It covers the Stage-1 ground iSALE
would have, it is the only bridge that can do an oblique impact, and it
carries the tabular EOS and non-ideal Saha solver this framework is still
missing.

### Which installer: micromamba, mamba, or conda

Everything below assumes the **conda-forge channel**. Which *command* you use
to talk to it is a separate question, and it is the difference between a
twenty-second install and a ten-minute one. These dependency graphs — PETSc,
MPI, VTK, CUDA — are precisely the ones the classic conda resolver struggles
with.

Preference order, fastest first:

| command | what it is | when |
|---|---|---|
| `micromamba` | a single static binary, no base environment at all | nothing installed yet, or you want to skip conda entirely |
| `mamba` | the libmamba solver in a conda base env | you already have Miniforge |
| `conda` | **fine, if it is ≥ 23.10** — see below | already installed and current |

**The part people get wrong:** conda **23.10 and later defaults to the
libmamba solver** — the same solver mamba uses. If you are on a current
conda, it is already fast and installing mamba buys you almost nothing.

So check your version first, because there are three bands and they need
different things:

```bash
conda --version
```

| your conda | situation | what to do |
|---|---|---|
| **≥ 23.10** | libmamba is already the default | nothing — you are done |
| **22.11 – 23.9** | classic by default, but switchable | `conda config --set solver libmamba` |
| **< 22.11** | **cannot reach libmamba at all** | get micromamba (below) |

That bottom row is the trap. The `solver` config key did not exist before
conda 22.11 (it arrived with the CEP-4 plugin hook and
conda-libmamba-solver 22.12.0), and current plugin releases require conda
≥ 23.3. On, say, conda 4.10.3 the advice you will find everywhere fails
immediately:

```
$ conda config --set solver libmamba
CondaValueError: Key 'solver' is not a known primitive parameter.
```

Upgrading conda in place from that far back is itself a large classic-solver
solve, and often ends in a broken base environment. The cheap, low-risk route
is a standalone micromamba, which ignores your conda entirely while still
installing into the environment you already have:

```bash
scripts/install_solvers.sh --bootstrap-mamba
```

It drops a single static binary in `$HVI_SRC_DIR/bin` and installs with
`-p $CONDA_PREFIX`, so your active env keeps working and nothing in conda's
base is touched.

For the middle band, the cheap fix comes first — one config line, no
download:

```bash
conda config --set solver libmamba
# if that errors, install the plugin first:
conda install -n base -c conda-forge conda-libmamba-solver
```

Either way, set the channel once:

```bash
conda config --add channels conda-forge     # mamba and micromamba read the same ~/.condarc
conda config --set channel_priority strict
```

`python -m hvi_emp doctor` reports which of the three it found and, only when
it is actually the slow one, how to fix it.

### The `--no-deps` rule

Whichever of the three you use, when you install *this* package into that
environment:

```bash
pip install -e . --no-deps
```

`--no-deps` stops pip reinstalling NumPy and SciPy on top of the conda-forge
builds, which is the single most common way to break a working scientific
stack.

---

## 1. LAMMPS — Stage 1, molecular dynamics

**Used by:** `hvi_emp.solvers.lammps_stage1`, `examples/05_lammps_impact.py`.
Runs **in-process** through LAMMPS's Python bindings, so the framework drives
it directly rather than shelling out.

### Route A — conda-forge (recommended)

```bash
mamba install -c conda-forge lammps
```

The conda-forge package ships the Python module (its own test suite asserts
`import lammps`), so this is one command and done.

Variants exist and the plain name gives you whichever the solver picks. To be
explicit:

```bash
mamba install -c conda-forge "lammps=*=*mpi_mpich*"    # MPI-parallel
mamba install -c conda-forge "lammps=*=*nompi*"        # serial
mamba install -c conda-forge "lammps=*=cuda*"          # GPU via Kokkos
```

### Route B — the PyPI wheel

```bash
pip install lammps mpich
```

Works, but read §8 first: the wheel links against MPICH's `libmpi.so.12`,
which the loader usually cannot find on its own. The framework preloads it for
you (`hvi_emp.solvers._preload_mpi`), so this route is less fragile here than
it is generally — but conda-forge is still cleaner inside a conda-family environment.

### One MPI per environment — read before installing PETSc next to LAMMPS

conda-forge ships **both** MPICH and Open MPI variants of LAMMPS, PETSc and
everything else that speaks MPI. They are not interchangeable inside a single
process: the two libraries use incompatible representations of
`MPI_COMM_WORLD`, so a handle made by one is garbage to the other.

The failure looks like this, and it is *not* a LAMMPS, PETSc or cluster fault:

```
UCX  WARN  UCP API version is incompatible: required >= 1.20, actual 1.19.1
Abort(201947909) on node 0: Fatal error in internal_Comm_size:
  Invalid communicator
```

`internal_Comm_size` is MPICH's internal name, so in that example MPICH's
`MPI_Comm_size` was handed an Open MPI communicator.

**How environments get into this state:** installing PETSc (for M2C) into an
environment that already has a LAMMPS built against the other MPI. The solver
is free to satisfy `petsc` with Open MPI while `lammps` stays on MPICH, and
nothing warns you until something calls `MPI_Init`.

Two ways out, both fine:

```bash
# 1. Pin one family for the whole environment and let the solver rebuild it
micromamba install -p $CONDA_PREFIX -c conda-forge "mpi=*=openmpi"
#    (or "mpi=*=mpich" -- pick one and never mix)

# 2. Or keep them apart. The chain stages are separate processes that
#    exchange files, so they do not need to share an environment:
micromamba create -n hvi-md   -c conda-forge python=3.11 lammps
micromamba create -n hvi-m2c  -c conda-forge petsc openmpi cmake
```

Option 2 is the more robust choice on a workstation you care about: M2C is a
compiled binary driven by `mpirun`, and it never has to be importable from the
same interpreter as LAMMPS.

Check what you have:

```bash
micromamba list -p $CONDA_PREFIX | grep -Ei "mpi|ucx|lammps|petsc"
python -c "from hvi_emp.solvers import mpi_conflict; print(mpi_conflict())"
```

`doctor` reports this specific mismatch by name rather than blaming LAMMPS,
and probes LAMMPS in a subprocess so an `MPI_Abort` cannot take the whole
report down with it.

### The `unconverted data remains: .4.0` import error

If `import lammps` dies with:

```
ValueError: unconverted data remains: .4.0
```

nothing is wrong with your installation. LAMMPS's own `lammps/__init__.py`
computes its `__version__` at import time by parsing its packaging metadata
as a date:

```python
vstring = importlib.metadata.version('lammps')   # e.g. "2025.7.22.4.0"
t = time.strptime(vstring, "%Y.%m.%d")           # chokes on the ".4.0"
```

The version it ships has four or five components; the format string expects
three. The PyPI wheel sidesteps this by hardcoding `__version__` instead of
calling the function, so **conda-forge installs are the ones that break**.
The compiled library underneath is fine — only the Python package's import
fails.

`hvi_emp` works around it automatically: `hvi_emp.solvers.import_lammps()`
patches `importlib.metadata.version` for the duration of the import and
restores it immediately, so `doctor` will report LAMMPS as OK and note that
it worked around the bug. Every LAMMPS import in the package goes through
that function.

If you want your *own* `import lammps` to work outside `hvi_emp`, either use
the wheel:

```bash
pip install lammps mpich
```

or edit the last line of `$CONDA_PREFIX/lib/python*/site-packages/lammps/__init__.py`
to read `__version__ = 0`.

### Verify

```bash
python -m hvi_emp doctor          # look for "LAMMPS (MD, Stage 1)  OK"
python -c "import lammps; print(lammps.lammps().version())"
python examples/05_lammps_impact.py --scale small --velocity 9e3
```

If `doctor` says *installed but will not load*, that is the MPI runtime, not
LAMMPS. `hvi_emp.solvers.lammps_diagnostics()` distinguishes the two cases and
prints the fix for yours.

### Potentials

The bridge defaults to `W_zhou.eam.alloy` for the Fraile et al. (2022)
tungsten protocol. LAMMPS potential files ship with the source, not always
with the binary packages. If a run dies with *cannot open potential file*,
fetch it from the LAMMPS repository's `potentials/` directory and point
`MD_LIBRARY` at it.

---

## 2. WarpX — Stage 3, electromagnetic PIC

**Used by:** `hvi_emp.solvers.warpx_stage3`, `examples/06_warpx_openmhd_decks.py`.

Deck generation — `write_picmi_script`, `write_warpx_inputs` — needs **nothing
installed**. You need WarpX only to run what it writes.

### Route A — conda-forge (CPU only)

```bash
mamba create -n warpx -c conda-forge warpx
mamba activate warpx
```

**This build has no GPU support.** That is an upstream limitation of the
feedstock, not a configuration you can flip:
<https://github.com/conda-forge/warpx-feedstock/issues/89>. It is still the
right first step — get a correct CPU answer, then decide whether you need it
faster.

Note the separate environment. WarpX's dependency graph collides with other
scientific stacks often enough that upstream recommends isolating it; you can
run the deck in the `warpx` environment and post-process in yours, since the
handoff is openPMD files on disk.

### Route B — Spack, with CUDA

This is the shortest path to a **GPU** WarpX.

```bash
# optional but strongly recommended — pulls prebuilt binaries
spack mirror add rolling https://binaries.spack.io/develop
spack buildcache keys --install --trust

spack install warpx +python compute=cuda
spack load warpx +python
```

`+python` is what gives you the `pywarpx` module the PICMI scripts import;
without it you get executables only, which work with `write_warpx_inputs` but
not `write_picmi_script`. Expect hours on a first build if the cache misses.

### Route C — CMake from source

Most control, most work. Install WarpX's dependencies first
(<https://warpx.readthedocs.io/en/latest/install/dependencies.html>), then:

```bash
git clone https://github.com/BLAST-WarpX/warpx.git ~/src/warpx
cd ~/src/warpx
cmake -S . -B build -DWarpX_COMPUTE=CUDA -DWarpX_PYTHON=ON -DWarpX_DIMS="2;3"
cmake --build build -j 32
# executables land in build/bin/
```

Note the repository is `BLAST-WarpX/warpx` — it moved from `ECP-WarpX`, and
old URLs in tutorials still point at the former.

### Route D — pip, from source

WarpX's docs do offer a pip route, but it **compiles from git**; it is not a
wheel download, and it needs the full dependency set already present:

```bash
python3 -m pip install -U pip build packaging "setuptools[core]" wheel cmake
python3 -m pip wheel -v git+https://github.com/BLAST-WarpX/warpx.git
python3 -m pip install *whl
```

### Post-processing needs nothing special

```bash
pip install openpmd-api picmistandard
```

Both are on PyPI and install cleanly. `load_warpx_probe` reads WarpX's openPMD
output back into arrays comparable with `emp.EMPResult`, so you can compare a
PIC run against the reduced Stage 3 without WarpX itself being importable.

### Verify

```bash
python -c "import pywarpx; print(pywarpx.__version__)"
python -m hvi_emp doctor
python examples/06_warpx_openmhd_decks.py     # writes the decks either way
```

### Blackwell (RTX 50-series, sm_120) needs CUDA >= 12.8

Compute capability is not optional detail: targeting the wrong one either
fails at compile time (`nvcc fatal: Unsupported gpu architecture`) or builds
cleanly and then refuses to run. The flags per card:

| card | compute cap | Idefix | PIConGPU | WarpX |
|---|---|---|---|---|
| V100 | 70 | `Kokkos_ARCH_VOLTA70` | `cuda:70` | `AMReX_CUDA_ARCH=7.0` |
| A100 | 80 | `Kokkos_ARCH_AMPERE80` | `cuda:80` | `8.0` |
| L40 / RTX 4090 | 89 | `Kokkos_ARCH_ADA89` | `cuda:89` | `8.9` |
| H100 | 90 | `Kokkos_ARCH_HOPPER90` | `cuda:90` | `9.0` |
| B200 | 100 | `Kokkos_ARCH_BLACKWELL100` | `cuda:100` | `10.0` |
| **RTX 5080/5090, RTX PRO 6000** | **120** | `Kokkos_ARCH_BLACKWELL120` | `cuda:120` | `12.0` |

Three traps on Blackwell:

* **CUDA >= 12.8.** Earlier toolkits cannot target sm_120 at all.
* **sm_120 is not sm_100.** Consumer Blackwell (GB202/GB203) and datacentre
  Blackwell (GB100) are separate compilation targets despite sharing the name.
* **Kokkos must be new enough to know `BLACKWELL120`.** Idefix bundles Kokkos
  as a submodule; if cmake rejects the arch, update it:
  `cd $IDEFIX_DIR && git submodule update --remote src/kokkos`.

`python -m hvi_emp doctor` prints these flags for the card in *your* machine,
identified from its PCI ID — so it works even when the driver is broken and
`nvidia-smi` cannot be asked.

### When nvidia-smi cannot reach the driver

The userspace half being installed while the kernel module is missing is
common, and `doctor` diagnoses which of four causes you have. The one that
catches people on a fresh or recently-upgraded machine:

```bash
sudo apt install linux-headers-$(uname -r)   # DKMS cannot build without these
sudo dkms autoinstall
sudo reboot
```

`modprobe: FATAL: Module nvidia not found in directory /lib/modules/<kernel>`
with `nvidia-dkms-*` installed means the DKMS build never ran or failed —
almost always missing headers for the running kernel. Check
`/var/lib/dkms/nvidia/<version>/build/make.log` if it still fails, and if the
build itself errors, the driver branch is older than the kernel: install a
newer one, or boot the previous kernel from GRUB's advanced menu.


---

## 3. Idefix — Stage 2, GPU MHD in spherical geometry

**Used by:** `hvi_emp.solvers.idefix_stage2`. This is the GPU-capable
alternative to the OpenMHD bridge, and the one to reach for first: it has
spherical geometry, non-ideal MHD, and CUDA via a single CMake flag.

Kokkos ships as a git submodule, so **nothing else needs installing** for a
serial CPU build — no external Kokkos, no MPI, no HDF5.

```bash
git clone --recurse-submodules \
    https://github.com/idefix-code/idefix.git $HOME/src/idefix
export IDEFIX_DIR=$HOME/src/idefix        # add to ~/.bashrc
```

`--recurse-submodules` is not optional; without it the Kokkos directory is
empty and CMake fails in a confusing way.

Idefix is compiled *per problem*: you configure inside the problem directory
and it builds an `idefix` executable there. The bridge writes that directory
for you.

```python
from hvi_emp.solvers.idefix_stage2 import IdefixConfig, write_problem_directory
cfg = IdefixConfig.from_expansion(sc.expansion)
out = write_problem_directory(cfg, "cavity_run", gpu_arch="AMPERE80", mpi=True)
print(out["cmake"])
```

Then:

```bash
cd cavity_run
cmake $IDEFIX_DIR -DIdefix_MHD=ON -DIdefix_RECONSTRUCTION=Parabolic \
      -DIdefix_EVOLVE_VECTOR_POTENTIAL=ON
make -j 8
./idefix
```

### GPU

```bash
cmake $IDEFIX_DIR -DIdefix_MHD=ON \
      -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
```

Pick the `Kokkos_ARCH_*` matching your card — `VOLTA70` (V100), `AMPERE80`
(A100/A30), `ADA89` (L40, RTX 6000 Ada), `HOPPER90` (H100). `cmake_command()`
in the bridge assembles this for you. With `-DIdefix_MPI=ON` on GPUs, Idefix
assumes the MPI library is GPU-aware.

### Verify

```bash
cd $IDEFIX_DIR/test/MHD/OrszagTang
cmake $IDEFIX_DIR -DIdefix_MHD=ON && make -j 8 && ./idefix
python -m hvi_emp doctor          # should report the tree
```

Output is VTK, so ParaView opens it directly. To read it in Python the bridge
calls Idefix's own `$IDEFIX_DIR/pytools/vtk_io.py` rather than reimplementing
the format — which is why `IDEFIX_DIR` must be set for `read_idefix_vtk`.

---

## 4. PIConGPU — Stage 3, GPU-native PIC

**Used by:** `hvi_emp.solvers.picongpu_stage3`. GPLv3, alpaka-based, native
openPMD output, and probe particles that record E and B at fixed points —
the simulated equivalent of the patch antenna in Close et al. (2013).

This is the heaviest install in this document. It needs boost, CMake, an MPI,
and openPMD-api/ADIOS2. PIConGPU distributes a `picongpu.profile` template
per system; write yours from
<https://picongpu.readthedocs.io/en/latest/install/profile.html> and source
it in every shell.

```bash
git clone https://github.com/ComputationalRadiationPhysics/picongpu.git \
    $HOME/src/picongpu
export PICSRC=$HOME/src/picongpu
export PATH=$PICSRC/bin:$PATH
export PIC_EXAMPLES=$PICSRC/share/picongpu/examples
export PIC_BACKEND="cuda"
```

PIConGPU compiles its `.param` files into the binary, so the workflow is
create → overlay → build → run:

```bash
pic-create $PIC_EXAMPLES/KelvinHelmholtz $HOME/picInputs/hviEMP
# overlay the generated .param and .cfg files, then
cd $HOME/picInputs/hviEMP
pic-build -b "cuda:80"                       # 80 = A100; see the table below
tbg -s bash -c etc/picongpu/1.cfg \
    -t etc/picongpu/bash/mpiexec.tpl $SCRATCH/runs/hviEMP_001
```

| card | `-b cuda:` |
|---|---|
| V100 | 70 |
| A100, A30 | 80 |
| A40, RTX 3090 | 86 |
| L40, RTX 4090, RTX 6000 Ada | 89 |
| H100, H200 | 90 |

### The precision trap

**PIConGPU builds in single precision by default.** An ion drifting at
20 km/s has a Lorentz factor of 1 + 2×10⁻⁹, which is below float32 epsilon
(1.2×10⁻⁷) — assign it and the drift is silently lost. The bridge computes
this, warns, and switches the build command to:

```bash
pic-build -b "cuda:80" -c "-DPRECISION_PIC=precision64Bit"
```

`build_command()` decides automatically from your plume state.

### Verify

```bash
which pic-build tbg
python -m hvi_emp doctor
python -c "import openpmd_api; print(openpmd_api.__version__)"
```

Note the param file rename: `simulation.param` is called `grid.param` in
releases before 0.8. If `pic-build` cannot find it, rename it.


---

## 5. OpenMHD — Stage 2, magnetised plume

**Used by:** `hvi_emp.solvers.openmhd_stage2`, which writes a `model.f90`
initial condition for the diamagnetic-cavity problem.

OpenMHD is **not installed to a prefix and produces no executable called
`OpenMHD`**. Each problem directory is compiled in place by its own Makefile
into `a.out` (serial) and `ap.out` (MPI). Searching `$PATH` for it will always
fail — `doctor` looks for the source tree instead.

### Build

```bash
git clone https://github.com/zenitani/OpenMHD ~/OpenMHD
export OPENMHD=~/OpenMHD          # add to ~/.bashrc so `doctor` finds it

cd $OPENMHD/2D_basic
# the Makefile's F90 line is the only thing you normally edit;
# the default is `mpif90 -Wall -O2`, which is right for gfortran + MPI
make
```

You should get `a.out` and `ap.out`. Requirements are a Fortran 90 compiler
and an MPI — gfortran and OpenMPI or MPICH are fine; the Makefile also carries
commented lines for Intel `ifx`/`mpiifx` and AMD `flang`.

### Running the generated problem

```python
from hvi_emp.solvers.openmhd_stage2 import (OpenMHDConfig, write_model_f90,
                                            write_run_notes)
cfg = OpenMHDConfig.from_expansion(sc.expansion)
write_model_f90(cfg, f"{OPENMHD}/2D_basic/model.f90")
write_run_notes(cfg, "openmhd_notes.txt")
```

Then rebuild in that directory and `mpirun -np 32 ./ap.out`. The run notes
carry the SI conversion factors — OpenMHD works in normalised Alfvén units, so
raw output is dimensionless and meaningless until you multiply through.

### GPU

OpenMHD ships `2D_basic_gpu`, `2D_reconnection_gpu`, `3D_basic_gpu` and
`3D_reconnection_gpu`. These are **CUDA Fortran** (`.cuf` sources), not
OpenACC-annotated Fortran, and their Makefiles want the NVIDIA HPC SDK:

```makefile
F90 = mpif90 -cuda -O2 -mcmodel=medium -cudalib=nvshmem
```

So you need `nvfortran` and `nvshmem` from the NVIDIA HPC SDK, plus a
`-gpu=ccXX` flag matching your card's compute capability (the Makefile carries
`cc80` and `cc120` examples). This is the most self-contained way to get your
GPU doing real work in this workflow, but it is a compiler-toolchain
installation, not a package install.

### Verify

```bash
ls $OPENMHD/2D_basic/*.out
python -m hvi_emp doctor          # should now report the tree and what is built
```

---

## 6. iSALE — Stage 1, optional and currently unobtainable

**Status: not the recommended route.** iSALE is free for academic
non-commercial use but a human approves each request, and requests from this
project went unanswered. The bridge (`hvi_emp.solvers.isale_stage1`) still
works and still writes complete decks, so nothing is lost if access does
arrive — but **do not plan around it.** §6b is the route that works.

If you already have access: iSALE is a Fortran code built with its own
configure script, wanting gfortran plus an MPI (iSALE3D only — **iSALE2D is
serial**, which was the single most important fact for planning, see §7). Set
`ISALE` to the install prefix or put `iSALE2D` on `$PATH` so `doctor` sees it.

```bash
which iSALE2D || echo $ISALE
python -m hvi_emp doctor
```

What it still has over M2C: Lagrangian tracers, and a crater-scaling heritage
benchmarked against Pierazzo et al. (2008). If a benchmarked *crater volume*
is what you need, it remains the better instrument. For everything else in
this framework — fields, plasma state, oblique geometry — M2C is ahead.

---

## 6b. M2C — Stage 1, the route that works

**Used by:** `hvi_emp.solvers.m2c_stage1`. See
[`SOLVERS.md`](SOLVERS.md) §5 for what it buys you; in short, it is the only
bridge that can represent an oblique impact, and the only one with a tabular
EOS and a non-ideal Saha solver.

### No application, unlike iSALE

GPLv3 on GitHub. Nobody has to approve you.

```bash
git clone https://github.com/kevinwgy/m2c.git ~/src/m2c
```

### Dependencies

PETSc and an MPI. On Ubuntu:

```bash
sudo apt install petsc-dev libopenmpi-dev cmake
```

Or, staying inside conda and avoiding sudo:

```bash
mamba install -c conda-forge petsc openmpi cmake
export PETSC_DIR=$CONDA_PREFIX
```

PETSc is the dependency that goes wrong. If CMake cannot find it, `PETSC_DIR`
(and `PETSC_ARCH`, on a source-built PETSc) is almost always the reason.

### Build

```bash
cd ~/src/m2c && mkdir -p build && cd build
cmake .. && make -j
export M2C_HOME=$PWD          # add to ~/.hvi_emp_solvers.env
```

### The standalone Saha solver — what it is, and is not

Zhao et al. (2026) §5.7 mentions a second repository:

> "For ease of reproducibility, the ionization module has been packaged as a
> standalone solver, available at www.github.com/kevinwgy/saha."

**It is not a dependency and not a replacement.** It is the *same* module
M2C already contains, packaged separately so the He/Ne/Ar verification case
(paper Fig. 11, reference Zaghloul) can be reproduced without running the
flow solver. Ionisation in M2C is in-loop — the paper's time-step algorithm
lists "Solve the Saha equation (if ionization effects are considered)"
between the laser and FSI steps. You do not need `saha` to run an impact.

It is, however, **the fastest way to check the atomic data**, which is worth
real time here: a coupled run takes days to reach a Z-bar you could have
sanity-checked in seconds.

```bash
git clone https://github.com/kevinwgy/saha ~/src/saha
cd ~/src/saha && mkdir -p build && cd build && cmake .. && make -j
python scripts/test_m2c.py --check-only --saha ~/src/saha
```

One thing the paper makes explicit that matters for the generated deck: the
partition function is **tabulated against exp(-1/T) at startup and looked up
by cubic B-spline** — that is the intended fast path. This bridge emits
`OnTheFly` instead, because the ground-state-only atomic data it generates
has a single level at E = 0 and the spline setup asserts on it (see below).
`OnTheFly` is therefore a *workaround for impoverished level data*, not a
preference. Supply real NIST levels and switch back.

### Atomic data: M2C ships none

The `Ionization` blocks reference `AtomicData/I_<Z>.txt` and friends. **The
M2C repository contains no `AtomicData` directory**, so a deck that names one
dies at startup:

```
*** Error: Cannot open ionization energy file AtomicData/I_13.txt.
```

`write_problem_directory()` now generates these files next to `input.st`,
from the same NIST-sourced ladder in the hvi_emp material YAML that the
reduced chain uses — so both codes solve the same problem and their Z-bar is
directly comparable. Nothing to copy or symlink.

Three sets are written per element, and all three matter:

| file | contents | if missing |
|---|---|---|
| `I_<Z>.txt` | ionisation energies | **fatal** |
| `E_<Z>_<r>.txt` | excitation energies | warning, but `rmax` collapses |
| `g_<Z>_<r>.txt` | degeneracies | warning, but `rmax` collapses |

M2C sets `rmax = min(len(I), len(E), len(g))`, so supplying `I` alone gives
`rmax = 0` — ionisation silently switched off, with only a warning.

**The files are numeric-only, deliberately.** M2C reads them with
`file >> double`, breaking only on eof. A `#` sets failbit rather than
eofbit, so the reader does not stop — it appends the failed value (`0`)
up to its 1000-entry limit. A commented `I_13.txt` loads as a thousand zero
ionisation energies and the run *completes*, with nonsense. Provenance
therefore lives in `AtomicData/README.txt`, which M2C never opens.

**The deck must use `PartitionFunctionEvaluation = OnTheFly` with this
data**, and does. `CubicSplineInterpolation` aborts on every rank:

```
AtomicIonizationData.cpp:218: Assertion `expmin<expmax' failed.
```

`InitializeInterpolationForCharge` takes the first *non-zero* excitation
energy as `factor = -E[r][i]/kb`. With one level at E = 0 there is no such
entry, so `factor` stays 0 and `expmin = expmax = exp(0) = 1`, making the
assertion `1 < 1`. On-the-fly is also right on physics grounds: with a single
level `U_r = g_r` is constant in temperature, so there is nothing to
interpolate. Switch to the spline only after supplying real level data, where
it is genuinely faster.

What is written is the **ground-state approximation**: one level per charge
state at zero excitation energy, so the partition function is exactly the
ground-state statistical weight. Excited states raise it, and so raise Z-bar
at high temperature. To improve on it, drop full NIST ASD level lists into
`E_*` and `g_*` — M2C reads up to 10,000 levels per charge state in the same
one-number-per-line format.

**`M2C_HOME` is a directory, not the executable.** After the `cd build`
above, `$PWD` is `~/src/m2c/build` and the binary is `$M2C_HOME/m2c` — which
is how every command here invokes it (`mpirun -np 64 $M2C_HOME/m2c ...`).
Setting it to `~/src/m2c/build/m2c` makes the path fail its `isdir` check;
`find_m2c()` recognises that particular slip and tells you the directory to
use instead of reporting "not found".

Two different paths are in play and it is worth keeping them straight:

| what | path | why |
|---|---|---|
| `M2C_HOME` | `~/src/m2c/build` | holds the compiled `m2c` binary |
| grammar check | `~/src/m2c` | reads the C++ **source** to verify keywords |

You can also leave `M2C_HOME` unset: the search looks in `./m2c`, `~/m2c`,
`~/src/m2c` and `~/src/m2c/build`, and prefers a built tree over a
source-only one wherever it finds them.

`doctor` distinguishes a cloned-but-unbuilt checkout from a built one, so a
half-finished install reports as such rather than as success.

### Verify — including the input grammar

```bash
python -m hvi_emp doctor
python -m hvi_emp.solvers.m2c_stage1 --find
python -m hvi_emp.solvers.m2c_stage1 --check ~/src/m2c
python examples/13_m2c_oblique_impact.py m2c_runs
```

The `--check` step greps M2C's own parser (`IoData.cpp`) for every keyword
the bridge emits. It should print *all present*. If it lists anything
missing, M2C's input format has moved since the bridge was written — fix the
spelling before running, because an unrecognised keyword is silently ignored
rather than rejected.

### Atomic data

The generated `Ionization` blocks reference ionisation-energy, excitation and
degeneracy files by path (default `AtomicData/`). Copy or symlink M2C's
shipped atomic data directory next to `input.st` before running, or the Saha
solver will not find its level data.

### Run

```bash
cd m2c_runs/oblique_45
mpirun -np 64 "${M2C_HOME:?set me}"/m2c input.st   # :? catches an unset M2C_HOME
```

Each generated directory carries a `README.txt` with its own command, an
order-of-magnitude cost estimate, and any warnings about the case.

---

## 7. ParaView — viewing the `.vti` output

```bash
mamba install -c conda-forge paraview
paraview vti_out/impact.pvd
```

The `.vti` writer in `hvi_emp.viz` has **no VTK dependency** — it emits the
XML ImageData format directly, with zlib-compressed appended binary. You need
ParaView only to look at the result. The rendering recipe (volume rendering,
transfer functions, the opacity curve that makes the plume visible) is in
[`VISUALISATION.md`](VISUALISATION.md) §4.

Any VTK-capable viewer works — VisIt, or `pyvista` if you would rather stay in
Python:

```bash
pip install pyvista
python -c "import pyvista; print(pyvista.read('vti_out/impact_0000.vti'))"
```

---

## 8. Failure modes, and what they actually mean

| symptom | cause | fix |
|---|---|---|
| `ERROR: Could not find a version that satisfies the requirement pywarpx` | there is no such package | §2 — conda-forge, Spack or source |
| `OSError: libmpi.so.12: cannot open shared object file` | LAMMPS wheel's MPI runtime missing | `pip install mpich`, or use conda-forge LAMMPS |
| `import lammps` works, `lammps.lammps()` raises | architecture mismatch (common on macOS: arm64 wheel, x86_64 Python) | `python -c "import platform; print(platform.machine())"`, reinstall matching |
| LAMMPS installed by conda/mamba but `import lammps` fails | you are in a different environment than you installed into | `conda activate`, then `which python` |
| `import lammps` raises `ValueError: unconverted data remains: .4.0` | **a bug in LAMMPS's own package** — see below | nothing; hvi_emp works around it |
| `doctor` says OpenMHD not found although you cloned it | it looks for the *tree*, not `$PATH` | `export OPENMHD=/path/to/OpenMHD` |
| OpenMHD `make` fails on `mpif90: command not found` | no MPI | `mamba install -c conda-forge openmpi` or edit `F90 = gfortran -O2` for a serial build |
| WarpX build succeeds but `import pywarpx` fails | built without Python bindings | Spack: `+python`. CMake: `-DWarpX_PYTHON=ON` |
| `ImportError: Cannot import packaging.licenses` | setuptools ≥ 77 with `packaging` < 24.2 | `pip install -U packaging` (this package deliberately avoids the trigger, but others will not) |
| NumPy/SciPy suddenly broken after installing a solver | pip overwrote conda's builds | `pip install -e . --no-deps`, and prefer conda for compiled dependencies |
| `pip install .` seems to install `UNKNOWN 0.0.0` | setuptools < 61 ignoring `pyproject.toml` | `pip install -U setuptools`; `setup.py` guards against this and will say so |

---

## 9. Is the GPU worth it?

Probably not first, and here is the honest accounting for a Xeon + NVIDIA box.

| component | GPU-capable? | how | worth it? |
|---|---|---|---|
| the framework itself | no, by design | — | it finishes in seconds |
| `.vti` writer | no | — | seconds to minutes |
| LAMMPS | **yes**, Kokkos | `mamba install "lammps=*=cuda*"` | at nm scale, MPI on 64 Xeon cores is already fast; modest gain |
| WarpX | **yes** | Spack `compute=cuda` or CMake `-DWarpX_COMPUTE=CUDA` | the clearest win — PIC is what GPUs are for |
| OpenMHD | **yes**, CUDA Fortran | NVIDIA HPC SDK + `*_gpu` directories | good win, but a toolchain install |
| **Idefix** | **yes**, Kokkos | `-DKokkos_ENABLE_CUDA=ON` | **the easiest GPU win here** — one CMake flag |
| **PIConGPU** | **yes**, GPU-native | `pic-build -b cuda:<arch>` | designed for it; heaviest install |
| iSALE | no | — | and iSALE2D is *serial*, which dominates your wall clock |

The binding constraint on this workflow is **iSALE2D's serial execution**, not
FLOPs. Six velocity cases on 128 cores run concurrently, so the series costs
as long as its slowest case (~12 h) rather than their sum — and no GPU changes
that. Spend the first week on cores and concurrency; come back to CUDA when a
specific run is the thing you are waiting on.

One caveat on the conda-forge CUDA LAMMPS: the recipe pins a fixed Kokkos
architecture (Maxwell 5.0 for the CUDA 12 builds, Ampere 8.0 for CUDA 13).
CUDA's PTX JIT means it will still *run* on a newer card, but not optimally.
If LAMMPS GPU performance matters, build it yourself with the right
`Kokkos_ARCH_*`.

---

## 10. What I verified, and what I did not

Being specific, because I already shipped one piece of wrong install advice in
this document's predecessor.

**Verified directly:**

* `pywarpx`, `warpx`, `warpx-pybind`, `amrex` all return HTTP 404 on PyPI.
* `lammps`, `mpich`, `openpmd-api` have manylinux wheels on PyPI;
  `picmistandard` is sdist-only.
* The conda-forge LAMMPS recipe tests `import lammps`, and builds both CUDA
  and CPU variants with `mpi_mpich` / `mpi_openmpi` / `nompi` strings.
* WarpX's documentation states the conda-forge package has no GPU support, and
  that PyPI wheels do not yet exist.
* OpenMHD's `2D_basic/Makefile` uses `mpif90 -Wall -O2`;
  `2D_basic_gpu/Makefile` uses `mpif90 -cuda -cudalib=nvshmem` on `.cuf`
  sources — i.e. CUDA Fortran via the NVIDIA HPC SDK.
* The whole framework, its 181 tests, all nine examples and `doctor` run
  from a clean extract with only NumPy and SciPy.

**Not verified on your hardware:** I have not built WarpX with CUDA, compiled
OpenMHD, or run iSALE. Wall-clock figures in §7 and in
[`REPLICATION_FLETCHER2021.md`](REPLICATION_FLETCHER2021.md) are cost-model
estimates, not measurements. Treat them as sizing, not promises — and if
yours differ by more than a factor of two, the cost model is what is wrong.
