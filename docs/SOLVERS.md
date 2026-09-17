# Production-solver bridges: M2C, LAMMPS, iSALE, Idefix, OpenMHD, PIConGPU, WarpX

The reduced-order chain answers in seconds; these bridges connect each stage
to an established open-source code for when you need the physics resolved
rather than closed.

| stage | solver | bridge module | can run here? |
|---|---|---|---|
| 1 — impact + ionisation | **M2C** (multi-material FV) | `hvi_emp.solvers.m2c_stage1` | input generation + readback (GPLv3, `git clone`, no application) |
| 1 — impact/vaporisation | **LAMMPS** (MD) | `hvi_emp.solvers.lammps_stage1` | yes, in-process via the official wheel |
| 1 — impact/cratering | iSALE (continuum) | `hvi_emp.solvers.isale_stage1` | input generation — *optional; access requests unanswered, see §5* |
| 2 — magnetised plume | **Idefix** (MHD, GPU) | `hvi_emp.solvers.idefix_stage2` | input generation (compile per problem; Kokkos bundled) |
| 2 — magnetised plume | **OpenMHD** (MHD) | `hvi_emp.solvers.openmhd_stage2` | input generation (Fortran source, compile per problem) |
| 3 — EMP radiation | **PIConGPU** (EM-PIC, GPU) | `hvi_emp.solvers.picongpu_stage3` | input generation (.param compile into the binary); openPMD readback |
| 3 — EMP radiation | **WarpX** (EM-PIC) | `hvi_emp.solvers.warpx_stage3` | input generation; in-process where `pywarpx` is installed |

`from hvi_emp.solvers import availability; availability()` reports what your
environment supports.

### The one file that runs everything

```bash
python scripts/run_full_chain.py --list-scales     # what your box can do
python scripts/run_full_chain.py --submit          # run + queue everything
python scripts/run_full_chain.py --render          # build the composite
```

It runs the whole waterfall — reduced chain, LAMMPS, M2C, OpenMHD, PIConGPU —
submitting the expensive stages to Slurm and skipping whatever is already
done. Re-run it as jobs land; state lives in `manifest.json`. A missing solver
blocks only its own stage and still gets its deck written.

**The composite it renders is not one simulation.** It joins four independent
runs at different scales with different projectile sizes, so the handovers
change *model*, not just magnification. That sentence is burnt into every
frame of the output, and `hvi_emp/viz/composite.py` explains why.

### MD scale — the default is deliberately small

`--md-scale smoke` is 25k atoms and looks like a toy, because it exists to
prove the machinery works. The real sizes:

| scale | cells | atoms | RAM | 64-core hours | for |
|---|---|---|---|---|---|
| `smoke` | 26 | 0.02 M | 0.0 GB | <0.01 | does it work at all |
| `talk` | 80 | 0.72 M | 0.2 GB | 0.23 | looks like a real impact in a slide |
| `detailed` | 140 | 3.85 M | 1.0 GB | 2.0 | resolved ejecta and fragments |
| `paper` | 200 | 11.2 M | 2.8 GB | 9.4 | publishable size distribution |
| `fraile` | 306 | 40.7 M | 10.2 GB | 85 | the largest run in this literature |

`talk` is 30× the smoke run for a quarter of an hour, and is the one to use
for anything you will show.

Those hours come from a **measured** throughput — 4.9×10⁵ atom-steps/s on one
core, flat to 4% across 3.9k, 11.7k and 25.2k atoms — times an **assumed** MPI
efficiency of 0.85, which cannot be measured from one rank. The measurement
was on an aarch64 core, so it is a floor for a modern x86 one rather than a
target. `estimate_cost()` returns `rate_is_measured` and
`efficiency_is_assumed` separately so the two never blur.

Anything estimated over `--md-local-minutes` (default 20) is submitted rather
than run in-process, with wall time and memory sized from the estimate.

**For the GPU**, the default wheel is CPU-only (MPI + OpenMP, no KOKKOS). The
CUDA build is `mamba install -c conda-forge "lammps=*=cuda*"` — see
[`INSTALLING_SOLVERS.md`](INSTALLING_SOLVERS.md) §1.

### How to invoke them

The bridge modules are **package modules, not standalone scripts** — they
import from their siblings, so running the file by path
(`python lammps_stage1.py`), or after copying it elsewhere, cannot work. They
detect that and tell you so rather than emitting a bare relative-import
traceback. Use any of:

```bash
# command line (easiest)
python -m hvi_emp lammps --material W --velocity 9e3 --out in.impact
python -m hvi_emp lammps --radius 5e-10 --run          # run in-process

# each module's own demo
python -m hvi_emp.solvers.lammps_stage1 --help
python -m hvi_emp.solvers.warpx_stage3 --n-e 1e16 --out warpx_emp.py
python -m hvi_emp.solvers.openmhd_stage2 --energy 1e-3
python -m hvi_emp.solvers.isale_stage1 --out isale_runs --cores 128
python -m hvi_emp.solvers.idefix_stage2
python -m hvi_emp.solvers.m2c_stage1 --check $M2C_HOME   # verify the grammar

# the worked examples
python examples/05_lammps_impact.py
python examples/06_warpx_openmhd_decks.py
python examples/09_gpu_solver_decks.py
python examples/13_m2c_oblique_impact.py
```

```python
# or import them
from hvi_emp.solvers.lammps_stage1 import LammpsImpactConfig, run_impact
```

All of these must be run from the project root — the directory containing the
`hvi_emp/` folder — unless you have `pip install`ed the package.

**Design rule.** Every bridge can always *generate* complete, runnable input
decks and can always *post-process* solver output back into framework objects,
whether or not the solver is installed. In-process execution is a convenience
layered on top where a Python interface exists.

## 0. Which code for which stage

Stage 1 now has three bridges and Stage 2 and 3 have two each. They are
alternatives, not a pipeline.

| stage | bridge | GPU | geometry / method | pick it when |
|---|---|---|---|---|
| 1 | **M2C** | no | 2-D axisym or **3-D** multi-material FV, level sets, tabular EOS, non-ideal Saha | **default choice** — the only bridge that can do an oblique impact, the only one with a coupled ionisation solver, and the only Stage-1 code you can simply clone |
| 1 | LAMMPS | yes (Kokkos) | MD, nm scale | you want the shock-and-release closures tested from first principles |
| 1 | iSALE | no | continuum shock physics, axisymmetric | you specifically need its benchmarked crater scaling *and* you have access — requests for it went unanswered, so M2C is the route |
| 2 | **Idefix** | **yes** (Kokkos/CUDA) | spherical or Cartesian, ideal + Ohmic/ambipolar/Hall | **default choice** — geometry matches the cavity, resistivity matters, one CMake flag for the GPU |
| 2 | OpenMHD | yes (CUDA Fortran) | Cartesian uniform, ideal | you want a second, independent MHD answer, or you already have it built |
| 3 | **PIConGPU** | **yes**, GPU-native | 2D3V/3D3V EM PIC, probe particles | **default choice** — fastest on your card, probes map onto a physical antenna, native openPMD |
| 3 | WarpX | yes, with a CUDA build | 2D/3D EM PIC | you want the Fletcher & Close configuration as originally bridged, or you prefer PICMI |

Idefix and PIConGPU normalise from the same Stage-2 state as their
counterparts, so running both and comparing is a genuine numerical check
rather than a duplicate. The test suite asserts that the Idefix and OpenMHD
normalisations agree to machine precision.

**Neither GPU code changes the physics limits.** MHD still cannot produce
charge-separation EMP; PIC still hits the same scale-separation wall between
the plasma period and the microsecond pulse. What they change is how much of
the affordable window you can actually resolve.

---

## 0b. Seeing the crater turn into plasma (MD resolution)

`hvi_emp.solvers.lammps_postprocess` converts a LAMMPS trajectory into the
per-atom view of cratering *with the plasma state attached*:

```bash
python examples/15_md_ejecta_fragments.py --run --sensitivity
pvpython md_scenes/scene_thermal.py      # dust cloud, plasma glowing behind
pvpython md_scenes/scene_fragments.py    # discrete debris, one colour each
```

It writes per-atom `.vtp` (every atom, lattice and ejecta, as in an OVITO
render) and binned `.vti` (the same state as continuum fields).

### Ejecta and fragments

`hvi_emp.solvers.fragments` turns the atom cloud into **discrete objects**:
ejecta are identified geometrically (above the original surface, moving away),
then grouped into fragments as connected components under a distance cutoff.
Each atom then carries `fragment_id`, `fragment_mass` and `is_ejecta` in the
`.vtp`, so ParaView can colour by fragment — the debris-cloud look — and
`size_distribution()` gives the cumulative N(≥m) that a fragmentation result
is normally reported as.

**The cutoff is the one free parameter and it changes the answer.** Too small
and a solid fragment shatters into singletons on thermal displacement; too
large and the cloud merges into one object. The default is `CUTOFF_FACTOR`
(1.4) times the median nearest-neighbour distance **of the whole frame**, not
of the ejecta — ejecta are dispersed, so their own spacing is larger and keeps
growing as they fly apart, and a cutoff derived from them would progressively
merge fragments over a time series. On a real frame that was 3.07 Å against
the bulk's 2.67 Å at 0.2 ps and rising.

`cutoff_sensitivity()` re-clusters across a range and reports the spread. Show
it alongside any size distribution:

```
  factor   cutoff [A]   fragments   largest   singletons
     1.1         2.94          53         6           47
     1.2         3.21          30         9           20
     1.4         3.74          21        20           15
     1.6         4.27          17        45           15
     2.0         5.34          17        45           15
```

It also does not know about bonding, only distance: a hot bound droplet and a
loose cluster of vapour at the same mean spacing are indistinguishable to it.
Above the melt point, read "fragment" as "connected region".

**Classical MD cannot ionise.** An EAM potential has no electrons; nothing in
a LAMMPS trajectory is an ionisation event, and Fraile et al. (2022) are
simulating mechanical cratering and sputtering, not plasma. So the ionisation
here is **inferred**: MD supplies density and temperature, and the framework's
own Saha solver answers what charge state would obtain in LTE at that density
and temperature. The caveat is written into every `.pvd` so it reaches
whoever opens the file.

Two details that decide whether the picture is honest:

* **Temperature is the velocity dispersion within a cell**, not per-atom
  kinetic energy. A cold lump moving at 10 km/s has enormous KE and zero
  temperature; using KE as temperature makes the entire ejecta plume look
  ionised when it is merely fast. There is a test for exactly this.
* **MD velocities ionise a small fraction, not the bulk — and this was
  measured, not assumed.** An earlier version of this document claimed a
  9 km/s MD run "will show a crater and *no* plasma". That is wrong, and a
  live run disproved it: W→W at 9 km/s reaches **5.75 eV and Z̄ = 1.62** in
  the hottest cells. Those are not noise — 29 cells, 46–77 atoms each, far
  above the 4-atom floor — and 5.8 eV is 11% of the 51 eV that full
  thermalisation of 9 km/s would give, so it is energetically unremarkable.

  What it is *not* is bulk ionisation. Those cells hold **6.8% of the
  atoms**. The reduced chain's 15–20 km/s threshold is for *mass-averaged
  complete vaporisation of the projectile*, which is a different quantity
  from a local peak at the impact point, and both can be right at once.

  This is the same structure as Jean & Rollins' 1970 result and the jetting
  model in `hvi_emp.jetting`: a small mass fraction reaches ionising
  conditions far below the bulk threshold, and the reason it does not
  overturn the RF threshold is that there is too little of it. Three
  independent routes — a 1970 ballistic range, shaped-charge theory, and MD
  — land in the same place.

  So read `Zbar_max` alongside the ionised *mass fraction*, never alone.
  `analyse_frame` now reports both.

---

## 1. Why bridges, not a rewrite

The scale separation makes "just do it all in MD/PIC" impossible, and it is
worth being precise about why:

* **MD (LAMMPS).** Fraile et al. (Nucl. Fusion **62**, 026034, 2022) — the
  paper this Stage-1 bridge is modelled on — ran the largest impacts in this
  literature: 4×10⁷ atoms, projectile radius ~13 nm, 100 ps. A 1 pg
  micrometeoroid is ~10¹³ atoms; the plume evolves for microseconds. Full-scale
  MD is ~10⁶ beyond reach.
* **PIC (WarpX).** Resolving the plume at the collisional→collisionless
  transition (n_e ~ 10²⁶ m⁻³) needs 10⁻¹⁶ s steps and sub-nm cells, while the
  sensor sits 0.3 m and ~ns away. Fletcher & Close (2017) hit exactly this
  wall and normalised their DG-PIC to the *late* plume; the WarpX bridge does
  the same, explicitly.
* **MHD (OpenMHD).** Ideal MHD is valid only in the window where the plume is
  collisional *and* magnetised — roughly cm-to-m plume radii in LEO. Earlier
  is hydro, later is kinetic.

So the division of labour is: the reduced chain provides the connective tissue
and the scaling; **LAMMPS resolves what the shock-and-release closures
approximate** (at nm scale, where MD is exact); **WarpX resolves the
charge-separation radiation mechanism** that reduced Stage 3 closes with one
parameter; **OpenMHD resolves the diamagnetic-cavity physics** that the
reduced chain omits entirely (THEORY §7.2, gap #4).

Every bridge initialises its solver **from the framework's state objects**, so
the decks are physically consistent with the impact being studied instead of
hand-picked.

---

## 2. LAMMPS — Stage 1 (molecular dynamics)

### Install

```bash
pip install lammps mpich              # official wheel + the MPI it links to
mamba install -c conda-forge lammps   # preferred inside a conda-family environment
```

Wheels exist for Linux (x86-64, aarch64), macOS (arm64 and x86-64) and
Windows, so `pip install lammps` works on Apple Silicon.

Two things go wrong often enough to be worth naming:

* **Installed but will not import.** The wheels link against MPICH's
  `libmpi.so.12`; without the `mpich` wheel or a system MPI you get
  `OSError: libmpi.so.12: cannot open shared object file` even though
  `pip show lammps` looks fine. The bridge preloads the MPI library from the
  `mpich` wheel automatically when it can find it, so usually
  `pip install mpich` is the whole fix.
* **Architecture mismatch on macOS.** An x86-64 wheel under a Rosetta Python,
  or vice versa. Check with
  `python -c "import platform; print(platform.machine())"`.

`hvi_emp.solvers.lammps_diagnostics()` tells you which of these you have:

```python
from hvi_emp.solvers import lammps_diagnostics
d = lammps_diagnostics()
print(d["ok"], d["stage"], d["reason"])
print(d["hint"])        # platform- and conda-aware fix
```

**None of this blocks you.** Without the Python bindings the bridge still
writes a complete, runnable deck; you just execute it with the standalone
`lmp` binary instead of in-process. The physics setup is identical.

### The protocol

`generate_input_deck` / `run_impact` implement the methodology of **Fraile,
Dwivedi, Bonny & Polcar, Nucl. Fusion 62, 026034 (2022)** (W-on-W, up to
9 km/s, up to 4×10⁷ atoms):

* single-crystal target, (001) surface; periodic x/y, free z;
* minimise → 300 K equilibration → **NVE** production with the projectile
  given a uniform downward velocity;
* 0.1 fs timestep; optional `fix viscous` damping walls at the lateral edges
  (their checked-and-optional shock absorber);
* target auto-sized to the projectile so boundaries don't matter;
* the paper's unit check is enforced in the test suite:
  KE/atom = 0.952 eV × (v [km/s])² for tungsten.

One deliberate deviation: the z boundary is shrink-wrapped (`p p m`) rather
than fixed with a hand-chosen vacuum gap, so fast ejecta can never be lost
through the box top — the automatic version of the paper's "sufficiently
large vacuum" requirement.

### Potentials

The wheel ships `W_zhou.eam.alloy` (Zhou 2004), which is the default so W
runs out of the box. Fraile et al. used the **Marinica EAM4** potential; to
reproduce their setup exactly, download it from the NIST Interatomic
Potentials Repository (search "2013 Marinica W") and pass:

```python
cfg = LammpsImpactConfig(material="W",
                         potential_file="/path/W_MNB_2013.eam.fs",
                         pair_style="eam/fs")
```

Al, Cu, Fe, Ni defaults also ship with the wheel (`MD_LIBRARY` lists them).

### Run it

```python
from hvi_emp.solvers.lammps_stage1 import (LammpsImpactConfig, run_impact,
                                           compare_with_reduced_model)

cfg = LammpsImpactConfig(material="W", velocity=9e3,
                         projectile_radius=1.3e-9,      # the paper's size
                         t_equilibrate=2e-12, t_production=1e-11)
res = run_impact(cfg)              # minutes at this scale, one core
print(res.summary())
print(compare_with_reduced_model(res))

# hand the MD ejecta straight to Stages 2-4:
from hvi_emp import simulate_expansion, simulate_emp
exp = simulate_expansion(res.to_impact_handoff(), t_end=1e-6)
```

For paper-scale runs (10⁷ atoms, 100 ps) don't run in-process — write the deck
and submit it:

```python
open("in.impact", "w").write(generate_input_deck(cfg))
# then:  mpirun -np 256 lmp -in in.impact
```

### What the handoff means — and does not

Classical EAM MD has **no electrons**: it cannot ionise. `to_impact_handoff`
takes the *thermodynamic* state of the MD ejecta (mass, temperature, bulk
velocity, cloud size) and assigns the charge state with the same Saha
machinery the reduced model uses. This mirrors the hydrocode→PIC waterfall of
Fletcher (2021), with MD in place of the hydrocode. And MD's nm-scale ejecta
are systematically less thermalised than continuum-scale plumes — treat the
comparison with reduced Stage 1 (`compare_with_reduced_model`) as a
factor-of-a-few consistency check, which is all nm-scale MD supports.

---

## 3. WarpX — Stage 3 (electromagnetic PIC)

### Install

**There is no `pywarpx` package on PyPI** — `pip install pywarpx` fails with a
404. The `pywarpx` *module* that the generated PICMI scripts import comes from
a conda-forge, Spack, or source build of WarpX. Shortest routes:

```bash
mamba install -c conda-forge warpx        # CPU only, minutes
spack install warpx +python compute=cuda  # GPU, hours
```

Full instructions, including the CMake/CUDA build, are in
[`INSTALLING_SOLVERS.md`](INSTALLING_SOLVERS.md) §2.

The bridge needs none of these to *generate* inputs — only to run them.
`openpmd-api` (which genuinely is pip-installable) is enough to post-process
the output.

### The setup

`WarpXEMPConfig.from_expansion(exp)` builds the Fletcher & Close (2017)
configuration from your Stage-2 state instead of hand-picked numbers:

* 2D TE domain, PML absorbing boundaries, Yee solver, cubic particle shape;
* Gaussian plume in a wedge of half-angle π/8 about the surface normal;
* electrons given a bulk drift **√(m_i/m_e) × the ion drift** — the paper's
  central *assumption*, which the PIC then tests;
* their two canonical cases: `case="cold"` (v_t = 0.1 v_d) and
  `case="warm"` (v_t = 10 v_d);
* grid chosen to resolve the Debye length and ω_pe, CFL-limited timestep —
  and `grid()["debye_resolved"]` tells you honestly when the capped grid
  cannot (Fletcher & Close ran in exactly that regime, leaning on the
  high-order method; plain Yee will numerically heat, so treat under-resolved
  runs as qualitative).

**Choose the epoch deliberately.** `from_expansion(exp)` at the transition
epoch gives the physically-correct initial condition but a hopeless cost
(f_pe ~ 10¹⁴ Hz). `from_expansion(exp, at_frequency=916e6)` initialises at
the epoch when the peak plasma frequency has fallen to the antenna band —
affordable, and what the experiment actually sees. This is the same
normalisation choice Fletcher & Close made, exposed as a parameter.

### Run and compare

```python
from hvi_emp import run_scenario
from hvi_emp.solvers.warpx_stage3 import (WarpXEMPConfig, write_picmi_script,
                                          write_warpx_inputs,
                                          load_warpx_probe)

sc = run_scenario("Fe", "W", mass=1e-16, velocity=50e3)
cfg = WarpXEMPConfig.from_expansion(sc.expansion, at_frequency=916e6)

write_picmi_script(cfg, "warpx_emp.py")   # python warpx_emp.py  (pywarpx)
write_warpx_inputs(cfg, "inputs")         # warpx.2d inputs      (executable)

# after the run:
probe = load_warpx_probe("diags", x=0.30, z=0.0)
from hvi_emp.solvers.warpx_stage3 import compare_with_reduced_model
print(compare_with_reduced_model(probe, sc.emp))
```

The three-way comparison — WarpX probe vs reduced Stage 3 closure bracket vs
the Close et al. (2013) 1.9 mV/m measurement — is the point of this bridge:
it turns the framework's biggest uncertainty (the charge-separation closure,
±2 decades) into a computable question.

---

## 4. OpenMHD — Stage 2 (magnetised plume)

### Install

```bash
git clone https://github.com/zenitani/OpenMHD    # Fortran 90 + MPI, gfortran
```

OpenMHD configures problems by editing a `model.f90` in a problem directory —
there is no input-file interface, which is why this bridge generates code
rather than decks.

### The problem it solves

The reduced Stage 2 is unmagnetised. On orbit, the expanding conducting plume
expels the geomagnetic field and forms a **diamagnetic cavity** that grows to
pressure balance and collapses — a low-frequency emission mechanism the
reduced chain lists as its gap #4, shown in ALEGRA-MHD by Fletcher (2021)
and observed at scale in the AMPTE barium releases.

```python
from hvi_emp import run_scenario
from hvi_emp.solvers.openmhd_stage2 import (OpenMHDConfig, cavity_estimates,
                                            write_model_f90, write_run_notes)

sc = run_scenario("Fe", "Al", mass=1e-9, velocity=59e3)
cfg = OpenMHDConfig.from_expansion(sc.expansion,      # late-plume epoch
                                   B0=3e-5,           # LEO field
                                   B_angle_deg=0.0)
write_model_f90(cfg, "model.f90")     # drop over a 2D problem's template
write_run_notes(cfg, "RUN.md")        # build/run/SI-conversion instructions
```

The generated `model.f90` sets a Gaussian plume with self-similar radial
velocity on a uniform magnetised LEO-like ambient, in OpenMHD's normalised
units (lengths in plume radii, velocities in ambient Alfvén speeds); the full
normalisation is written into the file header and the run notes so output
converts back to SI unambiguously.

`cavity_estimates` gives the analytic pressure-balance prediction the run
should be tested against — for a 1 µg Perseid in the LEO field the cavity is
metre-scale with a ~15 kHz characteristic frequency, which is a *very*
different observable from the GHz-band Stage-3 EMP and one reason on-orbit
signatures span such a wide band.

**Validity checks** (written into the run notes): ambient β ≪ 1, plume still
collisional over the run, domain larger than the predicted cavity.

---

## 5. M2C — Stage 1 (multi-material flow with coupled ionisation)

M2C (Zhao, Ma, Islam, Narkhede & Wang, *Comput. Phys. Commun.* 2026;
[github.com/kevinwgy/m2c](https://github.com/kevinwgy/m2c), GPLv3) is the
closest published counterpart to what this framework does at Stage 1, and the
bridge exists because it closes three gaps at once.

### Why it earns its place

**1. It is the hydrocode you can actually get.** iSALE needs an application
and a wait. M2C is `git clone`, GPLv3, on GitHub. It needs PETSc and MPI.

**2. Oblique incidence.** This framework's reduced chain turns `angle_deg`
into `v cos θ` and changes nothing else, so its plume is axisymmetric about
the surface normal at 0° and at 60° alike. Real oblique impacts throw ejecta
and vapour downrange. iSALE2D is axisymmetric too, so it cannot help. M2C is
3-D Cartesian, so the asymmetry is a geometry change rather than a modelling
impossibility — see `docs/THEORY.md` §7 for why this is a limitation of the
reduced model and not a physical result.

**3. The two open physics tasks.** M2C ships Tillotson and
ANEOS-Birch-Murnaghan-Debye equations of state — the tabular EOS the linear
Us–Up fit needs above ~20 km/s — and a multi-species non-ideal Saha solver
with Ebeling/Griem continuum lowering and partition functions built from NIST
level data. Those are exactly the two items still open in
`docs/PHYSICS_AUDIT.md`.

**What it does not do: the EMP.** M2C stops at the plasma state. Charge
separation and the radiated field remain Stage 3 (`emp.py`, `warpx_stage3`,
`picongpu_stage3`). The two codes compose — M2C upstream, `hvi_emp`
downstream — rather than compete.

### The mapping is exact, which is why the bridge is thin

Both codes use the same linear Us–Up Mie-Grüneisen EOS, so every material in
the YAML library maps onto `ExtendedMieGruneisenModel` with no fitting:

| `hvi_emp.Material` | M2C |
|---|---|
| `rho0` | `ReferenceDensity` |
| `c0` | `BulkSpeedOfSound` |
| `s` | `HugoniotSlope` |
| `gamma0` | `ReferenceGamma` |
| `cv_solid` | `SpecificHeatAtConstantVolume` |

`MaxChargeNumber` is set to the number of stages the material's YAML defines,
so M2C solves the *same* ionisation ladder `hvi_emp.ionization` does and the
two answers are directly comparable. If a run reports Z̄ approaching
`MaxChargeNumber`, that is the ladder truncating, not physics — extend the
material file and regenerate.

### Units are the sharp edge

**M2C's input is in mm, g, s, K, A — not SI.** A wrong factor produces a run
that *completes* and is wrong by a power of a thousand, with nothing in the
output announcing it. `TO_M2C` holds every conversion and `tests/test_m2c.py`
pins each one against numbers M2C itself prints in its shipped
hypervelocity-impact deck: tantalum at 16.65e-3 g/mm³, Planck at
6.62607004e-25, electron mass at 9.10938356e-28, Boltzmann at 1.38064852e-14,
c_v = 139 J/(kg K) as 139e6. Note that pressure and charge convert by
**exactly 1.0** — kg/(m s²) and g/(mm s²) are the same number, as are C and
A s. That looks like a bug and is not; a "tidy-up" that gives every constant
a factor is the failure mode those tests exist to catch.

### Verify the grammar once

```bash
git clone https://github.com/kevinwgy/m2c.git ~/src/m2c
python -m hvi_emp.solvers.m2c_stage1 --check ~/src/m2c
```

Every keyword the bridge emits was traced to M2C's shipped deck or to
`IoData.cpp`, its parser. The check re-greps the parser and reports anything
it no longer contains, which catches an upstream rename before it becomes a
run that silently ignores your projectile.

### Use it

```bash
python examples/13_m2c_oblique_impact.py m2c_runs
```

```python
from hvi_emp.materials import get_material
from hvi_emp.solvers.m2c_stage1 import M2CConfig, write_problem_directory

cfg = M2CConfig(projectile=get_material("al"), target=get_material("al"),
                diameter=1e-3, velocity=1e4, angle_deg=45.0)   # -> 3-D
res = write_problem_directory(cfg, "m2c_oblique", cores=64)
print(res["command"], res["estimate"])
```

Mesh defaults depend on dimensionality. Cubing the axisymmetric numbers
(50 cells per radius over 24 radii) gives ~461M cells and ~86 GB on the
graded mesh the deck actually writes. 3-D defaults drop to 20 cells per
radius over 10 radii — ~12M cells, ~2 GB. Raise them deliberately; the
config states the cost rather than silently generating something
unrunnable.

**Corrected in build b13.** These figures were previously quoted as 13.8
*billion* cells and 2.6 TB, from a cell count that assumed a uniform fine
mesh across the whole domain. Against a real run M2C reported 353,864 cells
where that model predicted 5.76M — 16x high. 3-D at full 2-D resolution is
therefore a large-workstation job, not a cluster allocation. Runtime is a
separate question, and the binding one: see §Cost below.

### Reading it back

```python
from hvi_emp.solvers.m2c_stage1 import read_ionization_probes, to_plume_state
pr = read_ionization_probes("results/ionization_probes.txt")
print(to_plume_state(pr, target))     # n_e peak in m^-3, and when
```

`MeanCharge = On` and `ElectronDensity = On` in the generated Output block
give exactly the Z̄ and nₑ fields Stage 3 consumes. Probe files are parsed by
header rather than by assumed column order, because a bridge that assumes an
order fails silently the moment the deck changes.

### The ambient gas is a real difference, not a nuisance

Setting `ambient="ar"` reproduces a chamber experiment. Islam et al. (2023)
find that the ambient gas ionises in its own right, from compression ahead of
the projectile. `hvi_emp`'s Stage 1 has no such term at all, so the two will
not agree — and that disagreement is the point of running with a chamber gas.
The generated run notes say so.

---

## 6. Choosing your tool

| question | tool |
|---|---|
| ranking scenarios, parameter sweeps, scalings | reduced chain (seconds) |
| what does an *oblique* impact actually look like? | M2C bridge (nothing else here can) |
| is the EOS still valid above 20 km/s? | M2C with Tillotson or ANEOS |
| does continuum lowering change the ionisation fraction? | M2C's non-ideal Saha |
| is the shock/vaporisation closure right at small scale? | LAMMPS bridge |
| what does the charge-separation mechanism *actually* radiate? | WarpX or PIConGPU bridge |
| what does the geomagnetic field do to the plume? | Idefix or OpenMHD bridge |
| absolute EMP amplitude for a design decision | PIC, bracketed by the reduced closures (`docs/EMP_UNCERTAINTY.md`) |

All the bridges hand their results back through the same objects
(`ImpactResult`-compatible handoff, `EMPResult`-comparable probes), so a
mixed pipeline — e.g. M2C Stage 1 → reduced Stage 2 → PIConGPU Stage 3 — is
a few lines (`examples/05_lammps_impact.py` does exactly this with LAMMPS).
