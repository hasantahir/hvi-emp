# Package contents

| path | lines | what it is |
|---|---|---|
| **Documentation** | | |
| `README.md` | — | overview, headline validation table, what it can and cannot do |
| `docs/USER_GUIDE.md` | — | **how to use it**: install, run, interpret, parameters, recipes, troubleshooting |
| `docs/THEORY.md` | — | derivations, closures, validity limits, validation, known gaps |
| `docs/SOLVERS.md` | — | M2C / LAMMPS / iSALE / Idefix / OpenMHD / WarpX / PIConGPU bridges: which, install, run, interpret |
| `docs/REPLICATION_FLETCHER2021.md` | — | worked plan to replicate the NRL report on a workstation |
| `docs/VISUALISATION.md` | — | `.vti` output: what is solved vs drawn, ParaView recipe |
| `docs/WORKSTATION_QUICKSTART.md` | — | Linux workstation setup, what uses the GPU |
| `docs/PHYSICS_AUDIT.md` | — | 509 invariant checks, the 5 physics bugs they caught, the open discrepancy and what closed it |
| `docs/EMP_UNCERTAINTY.md` | — | the 184× amplitude bracket: one parameter, calibration, and the PIC check |
| `docs/WORKFLOW.md` | — | materials YAML, pydantic schema, Hydra, Slurm, Snakemake |
| `docs/DEVELOPMENT.md` | — | the git repo, and moving code to the workstation with `git bundle` |
| `CHANGELOG.md` | — | the b-series history, reconstructed; `git log` from b14 on |
| `docs/PERFORMANCE.md` | — | parallelism, CUDA, what scales and what cannot |
| `docs/INSTALLING_SOLVERS.md` | — | per-solver install: M2C, LAMMPS, WarpX, Idefix, PIConGPU, OpenMHD, ParaView |
| **Core package** | | |
| `hvi_emp/conf/materials/*.yaml` | — | **the material library** — 8 files, one per material |
| `hvi_emp/conf/{config,scenario,sweep,cluster}` | — | Hydra config groups |
| `hvi_emp/schema.py` | 560 | pydantic schema + physics bounds; identical no-pydantic fallback |
| `hvi_emp/config.py` | 280 | Hydra composition, with a plain-YAML fallback composer |
| `hvi_emp/cluster.py` | 340 | Slurm scripts, array jobs, walltime/memory sizing |
| `hvi_emp/run.py` | 280 | `python -m hvi_emp.run` — what sbatch and Snakemake both call |
| `workflow/Snakefile` | 210 | the DAG: validate -> points -> aggregate -> provenance |
| `workflow/profiles/slurm/` | — | Snakemake Slurm executor profile, per-rule resources |
| `hvi_emp/parallel.py` | 175 | ordered parallel map; no nested pools; failures recorded not hidden |
| `hvi_emp/accel.py` | 290 | NumPy/CuPy namespace dispatch + GPU kernels for the bulk stages |
| `hvi_emp/constants.py` | 60 | CODATA constants, unit conversions |
| `hvi_emp/pkgmgr.py` | 230 | picks micromamba/mamba/conda and tells the truth about which is slow; every install hint in the codebase routes through it |
| `hvi_emp/_compat.py` | 35 | NumPy 1.x/2.x shim (`trapz` was renamed `trapezoid` in 2.0) |
| `hvi_emp/materials.py` | 310 | material database + mixture construction |
| `hvi_emp/eos.py` | 400 | Hugoniot, Vinet cold curve, release, waste heat, phase thresholds |
| `hvi_emp/ionization.py` | 400 | multi-stage Saha, continuum lowering, rate coefficients, tabulated solver |
| `hvi_emp/impact.py` | 400 | **Stage 1** — shock, decay, vaporisation, plasma inventory |
| `hvi_emp/jetting.py` | 250 | **Stage 1b** — shaped-charge jetting; why Jean & Rollins (1970) saw plasma far below the bulk threshold |
| `hvi_emp/expansion.py` | 390 | **Stage 2** — plume expansion, shell freeze-out, collisionless transition |
| `hvi_emp/expansion_2t.py` | 320 | **Stage 2 (2T)** — T_e ≠ T_i, non-equilibrium ionisation per shell; brings β from 6.70 to 3.65 |
| `hvi_emp/radiation.py` | 170 | bremsstrahlung, radiative recombination, Kramers escape — measured negligible, kept anyway |
| `hvi_emp/emp.py` | 460 | **Stage 3** — charge separation, spectrum, dipole radiation |
| `hvi_emp/coupling.py` | 340 | **Stage 4** — surface charging, ESD, induced currents |
| `hvi_emp/propagation.py` | 190 | plasma cutoff/transmission, dust scattering |
| `hvi_emp/pipeline.py` | 240 | end-to-end driver, velocity sweep |
| `hvi_emp/validation.py` | 410 | 36 comparisons against published measurements, oldest Jean & Rollins (1970) |
| `hvi_emp/cli.py` | 280 | command-line interface |
| `hvi_emp/doctor.py` | 480 | environment check: hardware, GPU/driver diagnosis, solvers, live self-test |
| `hvi_emp/viz/vti.py` | 560 | VTK ImageData writer + impact/plume/smoke volume sampler; `SCENES` (impact vs plume: one domain cannot show both) |
| `hvi_emp/chain.py` | 250 | the full-waterfall DAG: stage definitions, manifest, resume, Slurm submission |
| `hvi_emp/viz/composite.py` | 330 | the continuous-zoom composite across all solvers, with the "not one simulation" caveat burnt into every frame |
| `hvi_emp/viz/paraview_scene.py` | 400 | generates a `paraview.simple` script that builds the view and saves a `.pvsm`; pins the GPU volume mapper; NVIDIA IndeX; headless `pvbatch` animation; M2C array aliases |
| **Production-solver bridges** (`hvi_emp/solvers/`) | | |
| `solvers/lammps_stage1.py` | 480 | LAMMPS MD impact (Fraile 2022 protocol); runs in-process |
| `solvers/warpx_stage3.py` | 430 | WarpX EM-PIC EMP (Fletcher & Close setup); PICMI + native decks |
| `solvers/openmhd_stage2.py` | 380 | OpenMHD diamagnetic-cavity problem; model.f90 generation; Fletcher Figs. 6-7 |
| `solvers/idefix_stage2.py` | 620 | Idefix: spherical/non-ideal MHD cavity, GPU via Kokkos; setup.cpp + idefix.ini |
| `solvers/picongpu_stage3.py` | 780 | PIConGPU: GPU-native EM PIC with probe particles; .param + .cfg; openPMD readback |
| `solvers/lammps_postprocess.py` | 400 | LAMMPS dump -> local T + Saha ionisation -> per-atom `.vtp` and binned `.vti`; per-atom n_e, ejecta and fragment arrays; reports the ionised **mass fraction**, not just the peak |
| `solvers/fragments.py` | 300 | ejecta identification and connected-component fragment clustering; size distribution; cutoff sensitivity |
| `solvers/isale_stage1.py` | 490 | iSALE continuum shock physics (the ALEGRA substitute); auto-sized meshes |
| `solvers/m2c_stage1.py` | 900 | M2C multi-material flow + non-ideal Saha; **the only bridge that does oblique incidence**; mm-g-s-K-A unit conversion; grammar self-check |
| **Tests and examples** | | |
| `tests/test_hvi_emp.py` | 690 | 63 tests: conservation laws, limiting cases, regressions |
| `tests/test_viz.py` | 300 | 16 tests: byte-exact `.vti` round-trip via an independent parser |
| `tests/test_solvers.py` | 590 | bridge tests incl. a live LAMMPS smoke run, fallbacks, direct-execution guards, and install-advice regression guards |
| `tests/test_gpu_solvers.py` | 480 | 42 tests: Idefix/OpenMHD normalisation agreement, ini and setup.cpp syntax, PIConGPU precision trap, openPMD probe round-trip |
| `tests/test_chain.py` | 400 | 53 tests: resume, atomic manifest, sbatch content, MD size guard, and that the composite cannot lose its caveat |
| `tests/test_fragments.py` | 300 | 39 tests: clustering against configurations with known answers, and the bulk-vs-subset cutoff regression a real MD frame exposed |
| `tests/test_scenes.py` | 470 | 65 tests: crater-per-cell scale, the discarded-domain regression, and that every array the ParaView script references exists in the data |
| `tests/test_pkgmgr.py` | 210 | 26 tests: preference order, the conda 23.10 libmamba boundary, and that no advice names a command the machine lacks |
| `tests/test_packaging.py` | 230 | 7 tests: execute bits on shipped scripts, and that no advice names a tool the machine lacks |
| `tests/test_jetting.py` | 250 | 36 tests: Jean & Rollins (1970) Tables 1-3, the low-velocity divergence, the tungsten-impedance surprise |
| `tests/test_m2c.py` | 620 | 83 tests: unit conversions pinned to M2C's own deck values, deck structure, the emitted-vocabulary invariant, probe readback |
| `examples/01_single_impact.py` | 80 | full chain + 6-panel diagnostic figure |
| `examples/02_velocity_sweep.py` | 75 | Q(v), T_e(v), E(v) against empirical laws |
| `examples/03_sensitivity.py` | 95 | uncertainty from the four free parameters |
| `examples/04_olympus_anomaly.py` | 110 | Perseid impact / Olympus-1 scenario |
| `examples/_plotting.py` | 90 | optional-matplotlib helper (examples run without it) |
| `examples/05_lammps_impact.py` | 95 | MD impact, smoke to paper scale |
| `examples/06_warpx_openmhd_decks.py` | 75 | solver decks from a scenario |
| `examples/07_fletcher2021_replication.py` | 175 | full Fletcher (2021) replication setup |
| `examples/08_vti_visualisation.py` | 90 | ParaView `.vti` time series |
| `examples/10_md_crater_to_plasma.py` | 95 | MD trajectory -> crater, ionisation and plasma in ParaView |
| `examples/09_gpu_solver_decks.py` | 165 | Idefix + PIConGPU input sets from one scenario |
| `examples/13_m2c_oblique_impact.py` | 80 | M2C decks: normal, 45-degree oblique (3-D), and an argon-chamber variant |
| `examples/14_impact_scenes.py` | 150 | impact-scale and plume-scale `.vti` series + their ParaView scripts; `--list-quality`, `--renderer`, `--movie` |
| `examples/15_md_ejecta_fragments.py` | 175 | MD ejecta as particles and fragments, with the plasma behind them; `--run`, `--sensitivity` |
| `examples/12_two_temperature.py` | 125 | 2T energy conservation, e-i decoupling, the beta result |
| `tests/test_workflow.py` | 470 | 53 tests: schema rejections, config composition, sbatch content, CLI exit codes |
| `tests/test_performance.py` | 300 | 23 tests: parallel==serial, fast lookup==reference, GPU==CPU |
| `scripts/run_full_chain.py` | 330 | **the one file to run**: whole waterfall, Slurm submit, resume, composite render |
| `scripts/pipeline.sh` | 150 | run + render end to end; `--render-only` reuses solver data |
| `scripts/check_render_data.py` | 190 | is the black frame a scene bug or empty data? answers in a second, no ParaView |
| `scripts/test_m2c.py` | 300 | staged M2C test ladder: grammar, shipped test, smoke, physics-vs-reduced |
| `scripts/benchmark.py` | 165 | measures table build, lookup, sweep scaling, volume sampling, CUDA |
| `scripts/physics_audit.py` | 330 | conservation/thermodynamic invariants across parameter space; exits non-zero on failure |
| `scripts/install_solvers.sh` | 470 | idempotent installer for Idefix / OpenMHD / PIConGPU / WarpX / LAMMPS / ParaView; mamba-first with `--bootstrap-mamba`; `--dry-run` |
| **Packaging** | | |
| `pyproject.toml`, `setup.py`, `setup.cfg`, `requirements.txt` | — | `pip install .` or `pip install -e .` |

Roughly 8,700 lines of code and 2,000 lines of documentation.

## Where to start

1. `docs/USER_GUIDE.md` §2 — three ways to run it.
2. `python examples/01_single_impact.py` — see the whole chain and its figure.
3. `docs/USER_GUIDE.md` §3 — read an annotated output line by line.
4. `docs/THEORY.md` §7 — the limitations, before quoting any number.
