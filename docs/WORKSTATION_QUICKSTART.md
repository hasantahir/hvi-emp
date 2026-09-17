# Workstation quickstart (Linux, Xeon, NVIDIA GPU)

Short answer: **yes — start now.** The framework itself is pure
Python/NumPy/SciPy, and Linux is the platform everything here was developed
and tested on. Nothing needs the GPU, a licence, or a compiler to get going.

---

## 1. Fifteen minutes to first results

```bash
# 1. environment. mamba does the solving (fast); conda does the activating,
#    because `mamba activate` needs a shell hook that is often not set up.
#    Plain `conda create` works too and, on conda >= 23.10, is the same
#    libmamba solver underneath -- see docs/INSTALLING_SOLVERS.md.
mamba create -n hvi python=3.11 -y      # or: conda create ...
conda activate hvi
mamba install -c conda-forge numpy scipy matplotlib pytest -y

#    Neither mamba nor conda installed? Get a standalone micromamba:
#    scripts/install_solvers.sh --bootstrap-mamba

# 2. the package
cd hvi_emp_framework
pip install -e . --no-deps        # --no-deps so pip does not shadow conda

# 3. ask the machine what it can do
python -m hvi_emp doctor
```

`doctor` probes the hardware, checks every optional solver, runs the real
chain plus the 31 published-data comparisons, and prints a prioritised list
of what to install next. **If it ends cleanly you are done with setup.**

```bash
# 4. prove it end to end
python -m pytest tests/ -q                       # 194 tests
python examples/01_single_impact.py              # full chain + figure
python examples/08_vti_visualisation.py          # ParaView .vti animation
```

---

## 2. What runs on what

| task | uses | wall clock | needs |
|---|---|---|---|
| reduced chain, sweeps, sensitivity | 1 core | seconds | nothing |
| `.vti` visualisation | 1 core | seconds–minutes | nothing |
| LAMMPS MD (nm scale) | MPI, 16–64 cores | minutes | `mamba install -c conda-forge lammps` |
| WarpX PIC (Stage 3) | CPU, or GPU if built for it | minutes–hours | WarpX build (see below) |
| **Idefix** (diamagnetic cavity) | **GPU**, or MPI cores | ~0.5 h | `git clone --recurse-submodules` + cmake |
| **PIConGPU** (Stage 3 EMP) | **GPU** | ~1 h | source build (boost, MPI, openPMD) |
| OpenMHD (diamagnetic cavity) | MPI, 16–32 cores, or GPU | hours | `git clone` + gfortran |
| M2C 2D velocity series | MPI, all cores, cases sequential | ~1 h/case | git clone + PETSc |
| M2C 3D oblique | MPI, ~64 cores, ~12 GB | hours–days | same |

**RAM is not the constraint anywhere in this workflow**, at any size worth
calling a workstation. Core count is — the
Fletcher velocity series is six independent runs, so **any** box with six or
more cores finishes it in the time of its slowest case rather than their sum.
Beyond six cores, extra cores help LAMMPS and OpenMHD but not that series.

`python -m hvi_emp doctor` prints the sizing table for *your* core count
rather than an assumed one; trust it over the numbers here.

**Using your GPU is a deliberate choice, but no longer a hard one.** Ranked by
effort for the payoff:

* **Idefix** (Stage 2 MHD) — `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_<yours>=ON`
  and that is all. Kokkos ships as a submodule, so there is nothing else to
  install. **Start here.**
* **PIConGPU** (Stage 3 PIC) — GPU-native by design, `pic-build -b cuda:80`.
  The build itself is the work: boost, MPI, openPMD-api.
* **LAMMPS** — conda-forge ships CUDA/Kokkos builds: `lammps=*=cuda*`.
* **WarpX** — the conda-forge package is CPU-only
  ([upstream issue](https://github.com/conda-forge/warpx-feedstock/issues/89)).
  CUDA means Spack or a CMake build.
* **OpenMHD** — the `*_gpu` directories are CUDA Fortran and want `nvfortran`
  from the NVIDIA HPC SDK.

The framework itself, M2C and the `.vti` writer are CPU-only by design. See
[`INSTALLING_SOLVERS.md`](INSTALLING_SOLVERS.md) §9 for whether it is worth it
— the honest answer for the Fletcher campaign is that M2C's *CPU-only*
execution, not FLOPs, sets your wall clock.

---

## 3. Installing the optional pieces, in the order I would do it

Full per-solver instructions — every route, every verification step, every
failure mode I know of — are in
**[`INSTALLING_SOLVERS.md`](INSTALLING_SOLVERS.md)**. The short version and
the order that wastes least time:

| when | what | why then |
|---|---|---|
| **day 1, minute 0** | `git clone` and build M2C | no approval needed; PETSc is the only fiddly part |
| day 1, +5 min | LAMMPS | one command, immediate payoff |
| day 1, +10 min | ParaView | you will want to see the `.vti` output |
| day 1, +30 min | WarpX (CPU, conda-forge) | unblocks running the decks |
| day 1, +45 min | **Idefix** | one clone + one cmake; GPU for free |
| week 1 | OpenMHD | a second, independent MHD answer |
| week 1 | **PIConGPU** | the GPU-native PIC; budget an afternoon |

**Or just run the installer**, which does everything below except the two
things needing root or a human:

```bash
scripts/install_solvers.sh --dry-run all    # look first
scripts/install_solvers.sh all
source ~/.hvi_emp_solvers.env
python -m hvi_emp doctor
```

The manual equivalent, if you would rather see each step:

```bash
# 1. M2C — do this first, it is the Stage-1 hydrocode
#    https://github.com/kevinwgy/m2c   (GPLv3, no approval needed)

# 2. LAMMPS
mamba install -c conda-forge lammps
python -m hvi_emp doctor            # confirm it flipped to OK
python examples/05_lammps_impact.py

# 3. ParaView
mamba install -c conda-forge paraview
paraview vti_out/impact.pvd         # recipe in VISUALISATION.md §4

# 4. WarpX — note: NOT `pip install pywarpx`, no such package exists
mamba install -c conda-forge warpx  # CPU-only build
python -c "import pywarpx; print(pywarpx.__version__)"

# 5. Idefix — the easiest GPU win here
git clone --recurse-submodules \
    https://github.com/idefix-code/idefix.git ~/src/idefix
export IDEFIX_DIR=~/src/idefix      # --recurse-submodules is not optional
cd $IDEFIX_DIR/test/MHD/OrszagTang && cmake $IDEFIX_DIR -DIdefix_MHD=ON \
    && make -j 8 && ./idefix        # smoke test

# 6. OpenMHD
git clone https://github.com/zenitani/OpenMHD ~/OpenMHD
cd ~/OpenMHD/2D_basic && make       # needs gfortran + an MPI
export OPENMHD=~/OpenMHD            # so `doctor` can find it

# 7. PIConGPU — budget an afternoon; see INSTALLING_SOLVERS.md §4
git clone https://github.com/ComputationalRadiationPhysics/picongpu.git \
    ~/src/picongpu
export PICSRC=~/src/picongpu; export PATH=$PICSRC/bin:$PATH
```

Input generation for Idefix, PIConGPU, WarpX and OpenMHD works without any of
them installed — you only need them to *run* what the bridges write:

```bash
python examples/09_gpu_solver_decks.py --gpu A100
```

---

## 4. A first day that produces something

```bash
# morning: baseline and sanity
python -m hvi_emp doctor
python -m hvi_emp validate
python examples/03_sensitivity.py        # how uncertain the answers are

# the science you can do immediately
python -m hvi_emp sweep --vmin 20e3 --vmax 72e3 -n 30 --json sweep.json
python examples/07_fletcher2021_replication.py --cores 128

# afternoon: install LAMMPS, then a real MD impact
mamba install -c conda-forge lammps -y
python examples/05_lammps_impact.py --scale small --velocity 9e3

# and a visualisation to show people
python examples/08_vti_visualisation.py --mass 1e-9 --velocity 30e3 \
       --frames 80 --n 192
```

That last one is ~200 MB and a couple of minutes; 192³ is a good quality
setting for a workstation.

---

## 5. Things that will bite you

* **`pip` inside conda.** Use `pip install -e . --no-deps` so pip does not
  reinstall NumPy/SciPy over conda's builds.
* **Old `packaging`.** setuptools ≥ 77 with `packaging` < 24.2 breaks many
  builds (not this one — the licence field is deliberately omitted — but it
  will bite you elsewhere). `pip install -U packaging`.
* **`python setup.py`** is not the installer; it prints usage and exits.
* **Running package modules directly** (`python hvi_emp/solvers/lammps_stage1.py`)
  cannot work — they are package modules. Use `python -m hvi_emp.solvers.lammps_stage1`.
* **matplotlib is optional.** Examples run without it and print every number,
  skipping only the PNG.
* **Read [`THEORY.md`](THEORY.md) §7 before quoting absolute EMP amplitudes.**
  They are uncertain by ~2 orders of magnitude; the scalings and the
  vaporisation thresholds are the reliable outputs.

---

## 6. If `doctor` reports a problem

It tells you the specific fix. The three most likely:

| symptom | fix |
|---|---|
| `numpy`/`scipy` missing | `mamba install -c conda-forge numpy scipy` |
| LAMMPS "installed but will not load" | `pip install mpich` (the wheel links `libmpi.so.12`) |
| `GPU PRESENT BUT NO DRIVER` | the card is on the PCI bus, the driver is not: `sudo ubuntu-drivers install && sudo reboot` |
| `GPU PRESENT BUT BROKEN` | `doctor` diagnoses this one: DKMS built for the wrong kernel, module built but unloaded, userspace/kernel mismatch, or Secure Boot. Each has a different fix and only one of them is "reboot". |
| `GPU none` | genuinely no NVIDIA card visible; everything still runs on CPU |
| self-test raises | `doctor` now prints the failing line plus the numpy/scipy versions *and paths* — check for two installs on `sys.path` before anything else |
| `AttributeError: module 'numpy' has no attribute 'trapezoid'` | fixed: `np.trapz` was renamed in NumPy 2.0 and the code now resolves whichever exists. If you still see it, you are running a stale extract |

`doctor` reads `/sys/bus/pci/devices` directly for the GPU check, so it can
tell "no card" from "card with no driver" without needing `lspci` or a working
driver. When `nvidia-smi` is present but cannot reach the driver it goes
further and reports the running kernel, whether the module is loaded, which
kernels DKMS has it built for, and the Secure Boot state — then names the
likely cause. **The commonest one on a recently-upgraded machine is a DKMS
module still built for the previous kernel, where rebooting does nothing.** Those two look identical if you only run `nvidia-smi`, and they need
completely different fixes — building for CUDA before the driver exists wastes
hours and then fails at run time.

Full troubleshooting list in [`USER_GUIDE.md`](USER_GUIDE.md) §9.
