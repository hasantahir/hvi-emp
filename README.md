# HVI-EMP

**Hypervelocity impact plasma and electromagnetic pulse simulation framework**

A four-stage, physically-traceable model of what happens when orbital debris or
a meteoroid strikes a spacecraft: shock → vaporisation → ionisation → plume
expansion into vacuum → charge separation → radiated EMP → coupling into the
spacecraft.

Runs the full chain in a few seconds on a laptop. Every closure is derived and
its validity limits stated in [`docs/THEORY.md`](docs/THEORY.md); 36 comparisons
against published measurements are in `hvi_emp/validation.py`.

**→ [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) is the how-to**: install, run,
interpret the output, full parameter and result reference, recipes,
troubleshooting, and how to add your own materials or swap out a stage.

**→ [`docs/VISUALISATION.md`](docs/VISUALISATION.md) writes ParaView-ready
`.vti` time series** of the impact, plume and condensate — no VTK dependency —
plus a ParaView script that builds the view and saves a `.pvsm`. Note that
**one domain cannot show both the crater and the plume**: on a plume-sized box
the crater is 0.013 cells across, which is why a default render looks like a
featureless ball. `write_scene(..., scene="impact")` fixes that.

**→ [`docs/REPLICATION_FLETCHER2021.md`](docs/REPLICATION_FLETCHER2021.md)
is a worked replication plan** for Fletcher's NRL report, including the
M2C/OpenMHD/WarpX substitutions for the unobtainable ALEGRA, workstation
sizing, and a threshold disagreement worth chasing — now testable rather than
merely arguable, because M2C lets you switch continuum lowering on and off.

**→ [`docs/SOLVERS.md`](docs/SOLVERS.md) connects the stages to production
codes**: **M2C** or LAMMPS for Stage 1 (iSALE is bridged but optional — access
requests went unanswered), **Idefix** or OpenMHD for the
magnetised Stage 2, **PIConGPU** or WarpX for the Stage-3 PIC. M2C is the one
to reach for first — GPLv3 with no application to wait on, 3-D so it can do
an **oblique impact** (which the reduced chain structurally cannot), and it
carries the tabular EOS and non-ideal Saha solver this framework still lacks.
Idefix and PIConGPU are the GPU-capable pair — Idefix gives spherical
geometry and non-ideal MHD behind a single CMake flag, PIConGPU is GPU-native
with probe particles that stand in for a physical antenna. Every bridge
writes complete, runnable inputs whether or not its solver is installed.

**→ [`docs/PHYSICS_AUDIT.md`](docs/PHYSICS_AUDIT.md) — 516 invariant checks**
(`python scripts/physics_audit.py`), the **seven** real physics bugs they
caught — three in Stage 1, two in the two-temperature Stage 2, two in the
plume geometry — and the open discrepancy that is deliberately *not* hidden by widening a tolerance.

**→ [`docs/EMP_UNCERTAINTY.md`](docs/EMP_UNCERTAINTY.md) — the amplitude
accuracy problem**, why the 184× bracket is a single parameter, what
calibrating it buys and costs, and the PIC run that would settle it.

**→ [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — config, validation, HPC.**
Materials are YAML (`hvi_emp/conf/materials/`), validated by a pydantic
schema with physics bounds; runs are composed with Hydra; sweeps submit as
Slurm arrays; the DAG is a Snakefile with resume and provenance. All of it
is an optional extra — the core still runs on NumPy + SciPy alone.

**→ [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — parallelism and CUDA.**
`velocity_sweep` and ionisation-table builds run across every core
automatically; `hvi_emp.accel` adds CuPy paths for the bulk array stages.
A *single* expansion is a sequential ODE and benefits from neither — what
made it 6× faster was fixing a hot loop, not adding hardware.
`python scripts/benchmark.py` measures your machine.

**→ [`docs/WORKSTATION_QUICKSTART.md`](docs/WORKSTATION_QUICKSTART.md) —
setting up on a Linux workstation**, including what uses the GPU and what
does not. Start with `python -m hvi_emp doctor`. Per-solver installation —
every route, verification and failure mode — is in
[`docs/INSTALLING_SOLVERS.md`](docs/INSTALLING_SOLVERS.md).

---

## Install

```bash
pip install -e .            # editable install, puts `hvi-emp` on your PATH
# or
pip install .               # regular install
# or nothing at all — it is a plain directory; run from the project root
```

Only **NumPy, SciPy and PyYAML** are required (PyYAML because the material library is YAML, read at import). Optional extras:

| extra | brings | needed for |
|---|---|---|
| `".[plots]"` | matplotlib | the example *figures* only — never the physics |
| `".[dev]"` | pytest | running the test suite |
| `".[solvers]"` | lammps, mpich, picmistandard, openpmd-api | the production-solver bridges |
| `".[workflow]"` | pydantic, hydra-core, omegaconf, snakemake | config validation, HPC submission, orchestration |
| `".[gpu]"` | cupy-cuda12x | the CUDA paths in `hvi_emp.accel` |

**In a conda environment**, prefer conda for the compiled scientific stack and
pip for the rest:

```bash
mamba install -c conda-forge numpy scipy matplotlib pytest
pip install -e . --no-deps        # --no-deps so pip does not shadow conda's builds
```

Examples degrade gracefully: without matplotlib they print every number and
skip only the PNG.

`python setup.py` is **not** the installer — that file is a compatibility
shim and does nothing on its own.

Build requirements are deliberately modest: **setuptools ≥ 61, any
`packaging`**. The licence field is intentionally omitted from `[project]`
because both PEP 621 spellings are hazardous — the table form is deprecated,
and the PEP 639 SPDX form *hard-fails* on the common combination of
setuptools ≥ 77 with `packaging < 24.2`. The MIT licence is in `LICENSE` and
is attached to built distributions via `[tool.setuptools] license-files`.

## Run everything

```bash
python scripts/run_full_chain.py --list-scales   # what your machine can do
python scripts/run_full_chain.py --submit        # run + queue every solver
python scripts/run_full_chain.py --render        # composite animation
```

One entry point for the whole waterfall — reduced chain, LAMMPS, M2C,
OpenMHD, PIConGPU — with Slurm submission, resume, and a composite animation.
Safe to re-run as jobs land. See [`docs/SOLVERS.md`](docs/SOLVERS.md).

## Quick start

```python
from hvi_emp import run_scenario

sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3)
print(sc.summary())
```

```
Impact: Fe (1.000e-12 kg, 3.12 um radius) -> Al at 50.0 km/s, 0 deg from normal
  peak shock pressure         4312.8 GPa
  vapour mass (total)      3.140e-12 kg (3.14 x m_proj)
  plasma mass (superheat)  1.552e-12 kg (1.552 x m_proj)
  plasma temperature            1.03 eV
  mean charge state Zbar       0.245
  free charge Q            1.033e-06 C
...
  asymptotic v_z               28.07 km/s
  frozen charge Q         2.434e-08 C (Zbar_eff = 0.003)
...
Narrowband response (resonant-shell model):
     315.0 MHz  E = 5.655e-05 V/m
     916.0 MHz  E = 1.289e-04 V/m
```

Add `expansion_model="2T"` for the two-temperature Stage 2 — slower, but the
better physics and the one that reproduces the measured charge-yield exponent
(see below).

The plume's initial aspect ratio is `scenario.aspect0`, default 0.5. It is
**calibrated** against the measured cosine angular law rather than derived,
and it moves peak $n_e$ by ~10× across its plausible range — so sweep it if
your result depends on plume density. `aspect0=1.0` is a fixed point of the
expansion equations and gives an exactly isotropic plume, which the
measurement excludes. See [`docs/PHYSICS_AUDIT.md`](docs/PHYSICS_AUDIT.md).

## The four stages

| stage | module | in → out |
|---|---|---|
| 1 | `impact.py` | projectile + target → shock state, vapour/plasma mass, $T_e$, $\bar Z$, $Q$ |
| 2 | `expansion.py` | plume state → density/temperature history, freeze-out, collisionless transition |
| 2′ | `expansion_2t.py` | same, but $T_e \neq T_i$ with non-equilibrium ionisation (`expansion_model="2T"`) |
| 3 | `emp.py` | transition state → dipole source, spectrum, waveform at a sensor |
| 4 | `coupling.py` | plume + EMP → surface charging, ESD risk, induced harness voltages |

Supporting: `eos.py` (Hugoniot, Vinet cold curve, release), `ionization.py`
(multi-stage Saha, continuum lowering, rate coefficients, tabulated solver),
`propagation.py` (plasma cutoff, dust scattering), `materials.py`,
`validation.py`, `pipeline.py`.

Each stage's interface is a small physical state vector, so any stage can be
swapped for a higher-fidelity code (a real hydrocode for Stage 1, a PIC for
Stage 3) without touching the others.

## Examples

```bash
python examples/01_single_impact.py      # full chain + 6-panel diagnostic figure
python examples/02_velocity_sweep.py     # Q(v), T_e(v), E(v) vs empirical laws
python examples/03_sensitivity.py        # uncertainty from the free parameters
python examples/04_olympus_anomaly.py    # Perseid impact / Olympus-1 scenario
python examples/05_lammps_impact.py      # Stage 1 on LAMMPS (Fraile 2022 protocol)
python examples/06_warpx_openmhd_decks.py  # WarpX + OpenMHD decks from a scenario
python examples/07_fletcher2021_replication.py  # replicate Fletcher (2021)
python examples/08_vti_visualisation.py   # ParaView .vti time series
python examples/09_gpu_solver_decks.py     # Idefix + PIConGPU input sets
python examples/10_md_crater_to_plasma.py --dump impact.dump  # MD crater -> plasma
python examples/11_fletcher_threshold.py    # why our threshold differs from Fletcher
python examples/12_two_temperature.py --sweep  # T_e != T_i, and the beta result
```

Installing the optional solvers is scripted:

```bash
scripts/install_solvers.sh --dry-run all    # then drop --dry-run
```

## Tests and validation

```bash
python -m pytest tests/ -q          # 280 tests (+1 CUDA test, skipped without a device)
python scripts/physics_audit.py     # 516 invariant checks across parameter space
python scripts/benchmark.py         # where the time goes on this machine
python -m hvi_emp validate          # 36 comparisons with published data
python -m hvi_emp doctor            # what this machine can run + self-test
python -m hvi_emp.run --print-config  # resolve and validate a config, run nothing
snakemake -n                        # inspect the workflow DAG
```

The test suite checks things that *must* hold — Rankine-Hugoniot identities,
$P_c = -\mathrm{d}E_c/\mathrm{d}V$, Saha charge conservation, energy
conservation in the expansion ODE, $B = E/c$, the far-field $1/r$ limit,
NRL-formulary values for $\omega_{pe}$ and $\lambda_D$ — so a physics
regression shows up immediately.

## What it gets right

| | model | published |
|---|---|---|
| Al incipient vaporisation | 374 GPa | 380 GPa (Ahrens & O'Keefe) |
| Al complete vaporisation | 1504 GPa | 1500 GPa |
| W→Al complete-vaporisation threshold | 26 km/s | 10–20 km/s (Fletcher 2021) |
| charge-yield exponent $\beta$, $Q\sim v^\beta$ above 40 km/s | 6.70 (1T) / **3.65 (2T)** | 3.48 (Close 2013) |
| plume $T_e$ at 40 km/s | 1.1 eV | 2–2.5 eV (Fletcher & Close 2017) |
| E at 916 MHz, 0.30 m | 1.3×10⁻⁴ – 2.4×10⁻² V/m (closure bracket)<br>1.93×10⁻³ V/m (`closure="calibrated"`) | 1.9×10⁻³ V/m (Close 2013) |

The vaporisation thresholds are a stringent EOS test with **no free
parameters** — they follow entirely from $\rho_0$, $c_0$, $s$ and $\Gamma_0$.

**The charge-yield exponent was the framework's largest open disagreement with
experiment**, and closing it is the clearest result the framework has
produced. The default one-temperature Stage 2 gives β = 6.70 against 3.48
measured. Two candidate mechanisms were named in
[`docs/PHYSICS_AUDIT.md`](docs/PHYSICS_AUDIT.md) *before* either was
implemented; both were then built and measured:

* **radiative cooling** (`radiation.py`) — **ruled out**. The radiative
  cooling time beats the expansion time by ~54× at formation and by six more
  orders of magnitude afterwards. Effect on β: −0.26.
* **a two-temperature plume** (`expansion_2t.py`, $T_e \neq T_i$ with
  non-equilibrium ionisation per mass shell) — **this is the mechanism**. It
  gives **β = 3.65 against 3.48 measured**, and it flips the sign of the
  freeze-out contribution: 1T freeze-out *adds* 2.0 to β, 2T *subtracts* 1.0.

```python
sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3, expansion_model="2T")
```

The 1T path remains the default (it is ~4× faster and is what every other
number in the table above was measured against) and the validation check stays
marked `OPEN` — the agreement is good enough to report, not good enough to
stop looking. Plume temperature still runs ~2× below hydrocode values in both
models, which the 2T work did not touch.

## What it does not

**The absolute EMP amplitude is uncertain by a factor of 184, and it is one
number, not many** — the electron–ion charge separation δ. The two
first-principles closures are the *same* physics with different choices of
the velocity that sets it, and they coincide exactly if the ion sound speed
is used. Everything else in the radiated field is derived or cancels.

Because E is exactly linear in δ, one measurement determines it with no
fitting freedom: **ξ = 15 reproduces Close 2013 to 2%**, available as
`closure="calibrated"`. That makes absolute amplitudes usable while leaving
every *scaling* (velocity, mass, material, frequency, distance) predictive,
since none was used to fit ξ. It is also falsifiable — it claims the
separation is ~15 Debye lengths, which a PIC run measures directly.

The default stays `debye`, so nothing changed underneath existing results.
Full decomposition, including an energy-bound approach that **did not work**,
in [`docs/EMP_UNCERTAINTY.md`](docs/EMP_UNCERTAINTY.md).

The predicted spectral peak sits at 10¹³–10¹⁴ Hz, far above the 315/916 MHz at
which emission is actually detected. **Fletcher & Close report exactly the same
discrepancy** from their DG-PIC simulations; it is the central open problem in
the field, and this framework reproduces it rather than hiding it.

Other gaps: **oblique incidence only reduces the normal velocity component** — the plume stays axisymmetric about the surface normal at every angle and never becomes downrange-directed, so any slice perpendicular to the normal is a perfect circle by construction (see `docs/THEORY.md`); no magnetic field (so no diamagnetic-cavity emission), no
dust–plasma coupling, no target bias, no line radiation (so `radiation.py`
gives a *lower bound* on radiative losses), ionisation state under-predicted
relative to LMD-style pressure ionisation, and the linear $U_s$–$u_p$ EOS
should be replaced with a tabular one above ~30 km/s — Fletcher's 32–62 km/s
series sits far outside its fitted range. Single-temperature electrons and
ions is no longer on this list: pass `expansion_model="2T"`. Full discussion
in [`docs/THEORY.md`](docs/THEORY.md) §7.

## Materials

Built in: `Al`, `Fe`, `W`, `Cu`, `SiO2` (coverglass), `Kapton`, `Olivine`
(stony meteoroid), `Dolomite` (the AVGR target). Add your own by constructing a
`Material` — the required data is $\rho_0$, $c_0$, $s$, $\Gamma_0$, heat
capacity, melt/boil points, latent heats, cohesive energy, atomic mass, and the
first few ionisation potentials with ground-state degeneracies.

Mixed projectile/target plumes are handled by `materials.mix_materials`, which
builds a number-weighted synthetic species — necessary to avoid a spurious
discontinuity in $T_e$ where the mixture crosses 50/50.

## Production-solver bridges

`hvi_emp.solvers` connects each stage to an established open-source code —
M2C (multi-material flow with coupled non-ideal Saha ionisation) and LAMMPS
(MD) for the impact, Idefix or OpenMHD for the
magnetised plume, PIConGPU or WarpX for the EMP PIC. Every bridge is
initialised from the framework's computed state, generates complete runnable
inputs whether or not the solver is installed, and post-processes results
back into framework objects. See [`docs/SOLVERS.md`](docs/SOLVERS.md).

M2C is worth singling out: it is the closest published counterpart to this
framework's Stage 1, it is the only bridge that can represent an oblique
impact, and its Tillotson/ANEOS equations of state and continuum-lowering
Saha solver are exactly the two pieces of physics still open here.

## Why this matters operationally

ESA's [Zero Debris Technical Booklet](https://www.esa.int/Space_Safety/Clean_Space)
names two technical needs this framework speaks to directly:

* **§3.4 C — small particles that cannot be detected or avoided.** Below a
  centimetre there is no catalogue and no collision-avoidance manoeuvre; the
  only defence is design. That is precisely the regime here — micrometeoroids
  and untrackable debris at 1 pg to 1 mg.
* **§3.5 A.II–III — "tools to predict how hypervelocity impacts affect
  *internal* items".** The usual reading is mechanical: spall, ejecta, a rear
  wall. This framework adds the electrical path — surface charging, ESD and
  induced harness currents (`coupling.py`) — which is the mechanism proposed
  for the Olympus-1 loss and which no shielding model captures.

`jetting.py` connects the two. A strike on a bumper delivers a jet to whatever
is behind it at ~6× the impact speed, so the *internal* item sees a far more
energetic event than the external one; `downrange_threshold()` quantifies how
much earlier that starts than bulk vaporisation would suggest.

None of this is a debris-environment model — it takes the impact as given.
For flux, use MASTER or ORDEM.

## Citation of underlying work

This framework implements and connects models from, principally: Close *et al.*
(2010, 2013), Fletcher, Close & Mathias (2015), Fletcher & Close (2017),
Fletcher (NRL 2021), Crawford (2015), Crawford & Schultz (1993, 1999), Melosh
(1989), Ahrens & O'Keefe (1972), Anisimov *et al.* (1993), Vinet *et al.*
(1987), Kodis (1965), Tishkovets *et al.* (2011), Caswell, McBride & Taylor
(1995), and — for the MD bridge — Fraile, Dwivedi, Bonny & Polcar, Nucl.
Fusion **62**, 026034 (2022). The M2C bridge targets Zhao, Ma, Islam,
Narkhede & Wang, Comput. Phys. Commun. (2026), and the chamber-gas ionisation
it exposes is from Islam *et al.* (2023). Full reference list in
[`docs/THEORY.md`](docs/THEORY.md) §8.
