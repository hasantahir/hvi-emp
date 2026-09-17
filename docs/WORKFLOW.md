# Configuration, validation, job submission and orchestration

```bash
pip install '.[workflow]'                       # hydra, pydantic, snakemake

python -m hvi_emp.run --print-config            # resolve and check, run nothing
python -m hvi_emp.run scenario=fletcher_w_al    # one scenario
python -m hvi_emp.run sweep=velocity            # a sweep, on this machine
python -m hvi_emp.run sweep=velocity --submit   # write an sbatch array
snakemake --profile workflow/profiles/slurm     # the whole DAG on the cluster
```

Four layers, each doing one job:

| layer | file | what it does |
|---|---|---|
| **materials** | `conf/materials/*.yaml` | the material library, one file each |
| **schema** | `hvi_emp/schema.py` | pydantic validation with physics bounds |
| **config** | `conf/`, `hvi_emp/config.py` | Hydra composition and overrides |
| **submission** | `hvi_emp/cluster.py` | Slurm scripts and resource sizing |
| **orchestration** | `workflow/Snakefile` | the DAG, resume, provenance |

Everything above the core is an **optional extra**. Without hydra, pydantic
or snakemake the framework still runs on NumPy and SciPy alone, using a
fallback YAML composer and a dataclass validator with identical rules. That
matters because cluster login nodes are often locked down, and a framework
that cannot be inspected without installing five packages does not get used.

---

## 1. Materials are YAML now

All eight materials live in `conf/materials/`, one file each, validated on
load. They used to be `Material(...)` literals in `materials.py`, which meant
adding a material required editing library code and a typo'd number was
indistinguishable from a measured one.

```yaml
# conf/materials/Al.yaml
name: Al
rho0: 2700.0            # kg/m^3
c0: 5386.0              # m/s
s: 1.339
gamma0: 2.14
E_cohesive_eV: 3.39     # energies in eV; converted on load
E_ion_eV: [5.9858, 18.8286, 28.4477]
g_ion: [6.0, 1.0, 2.0, 1.0]
```

Add your own by dropping a file in, or point `$HVI_EMP_MATERIALS` at your own
directory. `get_material()`, `ALUMINIUM` and the rest are unchanged, and
`tests/test_workflow.py` pins every value against the previous hardcoded
numbers — moving data into config is only safe if the data did not move.

## 2. What the schema actually catches

Type checking is the least of it. These are the rules that matter, each
because breaking it is *silent*:

| rule | what goes wrong without it |
|---|---|
| `len(g_ion) == len(E_ion) + 1` | the partition function silently drops the last ion stage |
| ionisation potentials increasing | a transcription error that still produces a number |
| `T_melt < T_vap` | the melt/vapour lever rule inverts |
| `rho0 >= 100 kg/m^3` | g/cm³ left unconverted; every such value lands below 23 |
| `E_cohesive ≈ L_vap · m_atom` | the two disagree, so one has the wrong unit |
| `velocity >= 1 km/s` | `velocity: 50` meaning 50 km/s runs happily and produces no plasma |
| numbers are not strings | see below |

The cohesive-energy cross-check is sharp: on the shipped library it gives Al
1.15, Cu 1.12, Fe 1.21, W 1.04, SiO₂ 1.87, Dolomite 2.51, Olivine 3.00. It
does **not** apply to molecular solids — a polymer depolymerises into
fragments rather than vaporising to atoms, so Kapton's 10.7 is real physics.
That is why materials declare `bonding: molecular`, explicitly, rather than
having it guessed: a metal mislabelled would silently lose the check that
catches its unit errors.

### The YAML float trap

This bit is worth knowing, because it bit this repository:

```yaml
velocity: 50.0e3     #  the STRING '50.0e3'
L_vap: 5e+06         #  the STRING '5e+06'
velocity: 50000.0    #  the float 50000.0
L_vap: 5.0e+06       #  the float 5000000.0
```

PyYAML implements YAML 1.1, whose float grammar requires a decimal point
*and* a signed exponent. Three of the eight material files shipped with the
string form and nothing complained — because pydantic's lax mode coerces
numeric strings silently, so the pydantic and no-pydantic paths disagreed
about whether the same file was valid. `require_number` now rejects the
string form on both paths, with a message naming the fix.

## 3. Composing a run

```bash
python -m hvi_emp.run scenario=fletcher_w_al sweep=velocity cluster=slurm_array \
                      scenario.velocity=62e3 tag=fletcher
```

Groups live in `conf/scenario/`, `conf/sweep/`, `conf/cluster/`; add a file to
add an option. Overrides are checked against the composed config, so a typo
is an error rather than a new key:

```
override 'scenario.veloctiy=62e3': no key 'veloctiy' in scenario.
Did you mean ['velocity']? ...
(Unknown keys are rejected rather than added, because a typo would
otherwise leave the real value at its default.)
```

That last sentence is the whole point. `--print-config` resolves, validates
and prints without running anything.

## 4. Submitting to Slurm

```bash
python -m hvi_emp.run sweep=velocity cluster=slurm_array --submit
#   script     results/run/submit.sh
#   array size 14
#   walltime   00:30:00
#   dry run: script written, nothing submitted.
```

`--submit` is **dry by default**; add `--submit-now` to queue. A function
that submits by accident is worse than one that needs an argument.

A sweep becomes an **array job**, one task per point. Short independent
tasks backfill into a busy queue far better than one long wide reservation,
one non-converging point costs one task rather than the sweep, and a partial
sweep resumes by re-submitting the missing indices.

The generated script guards the two ways these jobs go wrong:

```bash
export OMP_NUM_THREADS=1        # and MKL, OPENBLAS, NUMEXPR
export HVI_EMP_WORKERS=16       # = cpus_per_task
```

NumPy's BLAS starts a thread per core by default. Inside a job already
running one process per allocated CPU that is `cpus_per_task²` threads on
`cpus_per_task` cores — slower than serial, while looking busy.
`hvi_emp`'s parallelism is at the process level, so those threads have
nothing useful to do. And `parallel.cpu_count()` reads `HVI_EMP_WORKERS` and
`SLURM_CPUS_PER_TASK` before falling back to the affinity mask, so the pool
matches the *allocation* rather than the node — requesting 4 CPUs and forking
48 workers is the standard way to be thrown off a shared cluster.

`estimate_walltime` derives the request from the cost model (point count, 1T
vs 2T, workers) with a 3× margin, rather than leaving you to guess. A job
killed at the wall clock loses everything; one that asks for 24 h to do 20
minutes sits in the queue.

Other schedulers: subclass `cluster.Scheduler` and implement `directives`,
`array_directive` and `submit_command`. PBS, SGE and LSF differ only in the
spelling of those three.

## 5. Snakemake

`velocity_sweep` already spreads N scenarios over one machine's cores, so why
an orchestrator? Four things it cannot do:

* **resume** after 3 of 40 points failed, without recomputing 37;
* **per-stage resources** — the reduced chain wants 4 CPUs, a PIConGPU deck
  wants a GPU and 12 hours;
* **a DAG you can inspect** (`-n`) before it charges an allocation;
* **provenance** — which inputs produced which output.

```bash
snakemake -n                                     # DAG only
snakemake --cores 16                             # locally
snakemake --profile workflow/profiles/slurm      # one Slurm job per rule
snakemake --config tag=fletcher scenario=fletcher_w_al sweep=fletcher_series
```

The unit of work is `python -m hvi_emp.run --sweep-index i` — the same
command an sbatch array runs, so there is one code path and not two. The
rule graph is `validate_config → scenario_point (×N) → aggregate →
provenance`, with `solver_decks` available on request.

`validate_config` runs first and costs a fraction of a second. Its job is to
move failure from hour three to second zero; in this project it has already
caught a YAML float that parsed as a string and an ionisation ladder one
statistical weight short.

`aggregate` lists every point explicitly as an input, so Snakemake will not
build a summary from a partial sweep. On a log-log charge-yield plot, six
missing points are indistinguishable from real physics.

The Slurm profile sets resources per rule. Requires
`snakemake-executor-plugin-slurm` for Snakemake ≥ 8; for Snakemake 7 use
`--cluster "sbatch ..."` with the same resource names.

## 6. What each run leaves behind

```
results/<tag>/point_0003/
    config.yaml      the fully resolved configuration
    result.json      scalars, units in the key names
    manifest.json    versions, hostname, Slurm job id, timings, argv
results/<tag>/
    summary.json     all points, sorted by the swept parameter
    summary.csv      the same, for a plotting tool
    provenance.json  git commit, dirty flag, library versions
```

A results directory without the configuration that produced it is an orphan;
six months later nobody can say what was run. `manifest.json` records
`SLURM_JOB_ID` and `SLURM_ARRAY_TASK_ID` where present, so a number in a plot
traces back to a specific task in a specific job.

Nothing in `run.py` catches exceptions to keep going. A failed scenario exits
non-zero so both the scheduler and the orchestrator see it — a sweep that
quietly records zero for a failure is indistinguishable from a real result.

## 7. Testing without a cluster

`tests/test_workflow.py` (53 tests) covers all of it without needing Slurm:
schema rejections, config composition, the YAML/legacy material equivalence,
generated sbatch content, and the `run` CLI's exit codes. The Snakefile test
checks that every rule the profile assigns resources to actually exists.

To verify the no-dependency path behaves identically, block the imports:

```python
import builtins
_real = builtins.__import__
builtins.__import__ = lambda n, *a, **k: (
    (_ for _ in ()).throw(ImportError(n))
    if n.split(".")[0] in ("pydantic", "hydra", "omegaconf")
    else _real(n, *a, **k))
```

Materials, composition and every error message come out the same.
