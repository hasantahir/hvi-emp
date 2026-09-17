# HVI-EMP User Guide

How to install, run, interpret and extend the framework.

For the *physics* — derivations, closures, validity limits — see
[`THEORY.md`](THEORY.md). This document is about operating the code.

**Contents**

1. [Install](#1-install)
2. [Three ways to run it](#2-three-ways-to-run-it)
3. [Understanding the output](#3-understanding-the-output)
4. [Running the stages individually](#4-running-the-stages-individually)
5. [Complete parameter reference](#5-complete-parameter-reference)
6. [Result-object reference](#6-result-object-reference)
7. [Recipes](#7-recipes)
8. [Adding your own materials](#8-adding-your-own-materials)
9. [Troubleshooting](#9-troubleshooting)
10. [Extending the framework](#10-extending-the-framework)
11. [Reading the results responsibly](#11-reading-the-results-responsibly)

---

## 1. Install

**Required:** Python ≥ 3.9, NumPy, SciPy. **Optional:** matplotlib (example
figures only — the examples run without it and print every number, skipping
just the PNG), pytest (test suite), and the solver bridges' dependencies.

```bash
pip install -r requirements.txt
```

**In a conda environment**, install the compiled scientific stack with conda
and the package itself with pip, so pip does not shadow conda's builds:

```bash
mamba install -c conda-forge numpy scipy matplotlib pytest
pip install -e . --no-deps
```

That is enough to use it — the package is a plain directory, so running from
the project root just works. To install it properly (and get the `hvi-emp`
command on your `PATH`):

```bash
pip install -e .              # editable
pip install .                # regular
pip install -e ".[solvers]"  # plus the LAMMPS/WarpX bridge dependencies
```

**`python setup.py` is not the installer.** That file is a compatibility shim
for build environments predating PEP 660; run on its own it prints usage and
exits.

Build requirements are **setuptools ≥ 61 and any `packaging` version**. If
your setuptools is older, `setup.py` now fails with an explanatory message
rather than silently installing the package as `UNKNOWN 0.0.0`.

Check it — `doctor` is the one command that answers "does this work here?":

```bash
python -m hvi_emp doctor            # hardware + solvers + live self-test
python -c "import hvi_emp; print(hvi_emp.__version__)"
python -m pytest tests/ -q          # 61 core + 37 bridge + 16 viz, ~42 s
python -m hvi_emp validate          # 31 published-data comparisons, ~15 s
```

**Units.** Everything is SI internally — metres, kilograms, seconds, kelvin,
pascals, coulombs. The two exceptions you will meet in *outputs* are
temperatures reported in eV (`T_eV` alongside `T` in kelvin) and pressures
printed in GPa. Impact speeds are passed in **m/s**, so 50 km/s is `50e3`.

---

## 2. Three ways to run it

### 2.1 One-liner in Python

```python
from hvi_emp import run_scenario

sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3)
print(sc.summary())
```

`run_scenario` is the whole chain: impact → expansion → EMP → coupling. It
returns a `Scenario` holding every intermediate result.

### 2.2 Command line

```bash
python -m hvi_emp run --projectile Fe --target Al --mass 1e-12 --velocity 50e3
python -m hvi_emp run --mass 1e-12 --velocity 50e3 --json result.json
python -m hvi_emp sweep --vmin 20e3 --vmax 72e3 -n 15
python -m hvi_emp materials
python -m hvi_emp validate
python -m hvi_emp lammps --material W --velocity 9e3 --out in.impact
python -m hvi_emp doctor            # hardware, solvers, live self-test
```

Add `--quiet` to suppress the physics-validity `RuntimeWarning`s (they are
still reported in the scenario summary, so nothing is lost). `--json` writes a
compact, JSON-safe digest for downstream analysis.

### 2.3 The worked examples

```bash
python examples/01_single_impact.py      # full chain + 6-panel diagnostic figure
python examples/02_velocity_sweep.py     # Q(v), T_e(v), E(v) against empirical laws
python examples/03_sensitivity.py        # uncertainty from the free parameters
python examples/04_olympus_anomaly.py    # Perseid impact / Olympus-1 scenario
python examples/05_lammps_impact.py      # Stage 1 on LAMMPS (Fraile 2022 protocol)
python examples/06_warpx_openmhd_decks.py  # WarpX + OpenMHD decks from a scenario
python examples/07_fletcher2021_replication.py  # replicate Fletcher (2021)
python examples/08_vti_visualisation.py   # ParaView .vti time series
```

Each of 01-04 writes a PNG into `figures/`; 05 runs a live MD impact if the
LAMMPS bindings are installed (`pip install lammps mpich`, or `mamba install -c conda-forge lammps`); otherwise it writes a runnable deck and says where, 06 writes solver
decks into `solver_decks/`, and 07 sets up the full Fletcher (2021)
replication (see [`REPLICATION_FLETCHER2021.md`](REPLICATION_FLETCHER2021.md)), and 08
writes a ParaView `.vti` animation (see
[`VISUALISATION.md`](VISUALISATION.md)). Read these first — they are the fastest way
to see what the framework does and how to drive it.

---

## 3. Understanding the output

`sc.summary()` prints six blocks. Below is the verbatim output of
`run_scenario("Fe", "Al", mass=1e-12, velocity=50e3)`, block by block, with
the parts worth understanding called out afterwards.

### Stage 1 — impact

```
Impact: Fe (1.000e-12 kg, 3.12 um radius) -> Al at 50.0 km/s, 0 deg from normal
  peak shock pressure         4312.8 GPa
  kinetic energy           1.250e-03 J
  vapour mass (total)      3.201e-12 kg (3.20 x m_proj)
  plasma mass (superheat)  1.832e-12 kg (1.832 x m_proj)
    from projectile        1.000e-12 kg (f_vap = 1.00)
    from target            2.201e-12 kg
  target melt mass         5.294e-11 kg
  plasma temperature            1.47 eV
  mean charge state Zbar       0.467
  free charge Q            2.267e-06 C
  initial n_e              7.812e+26 m^-3
  expansion speed              25.00 km/s
```

**Vapour mass ≠ plasma mass.** Vapour mass is everything that crosses the
boiling point, including two-phase material that arrives at `T_boil` with no
superheat and recondenses into dust. Plasma mass is only the superheated part
with energy left over for ionisation. Averaging the two together was the
single biggest error found during development — it made the plume come out ten
times too cold (THEORY §2.6).

Note also that the target contributes more than twice the plasma mass of the
projectile here, and that `free charge Q` is the value **at formation**.

### Stage 2 — expansion

```
Plume expansion (Fe0.5+Al0.5), N_heavy = 5.298e+13
  integration window      2.48e-12 - 2.48e-05 s
  final semi-axes         R_r 83.478 mm, R_z 83.478 mm
  asymptotic v_z          28.38 km/s
  frozen charge Q         5.892e-07 C (Zbar_eff = 0.069)
  mass-weighted freeze-out:
    t                     4.553e-08 s
    n_e (peak)            2.921e+22 m^-3
    T_e                   0.441 eV
  collisional -> collisionless transition (peak density):
    t                     2.479e-12 s
    plume scale           0.0205 mm
    n_e                   1.965e+26 m^-3
    T_e                   1.332 eV
    f_pe                  1.259e+14 Hz
    lambda_De             0.0006 um
```

`Fe0.5+Al0.5` is the synthetic mixed species — the plume is half projectile,
half target by mass (§8, THEORY §2.7).

**Frozen charge is what an RPA measures.** It is 3.8× smaller than the charge
at formation, because the dense core recombines before it can freeze out.
Quote `Q_frozen`, not `impact.Q_free`, when comparing with experiment.

`f_pe = 1.26e14 Hz` at the transition is the **known frequency problem**: it is
five orders of magnitude above the bands where emission is actually detected.
Fletcher & Close report the same discrepancy from their PIC runs (THEORY §5.6).

### Stage 3 — EMP

```
EMP at r = 0.300 m, theta = 90 deg
  peak |E|            3.591e-03 V/m
  peak |B|            1.198e-11 T
  radiated energy     4.264e-16 J
  spectral peak       5.739e+07 Hz
  band 100-500 MHz    1.509e-04 V/m
  band 0.5-2 GHz      1.061e-04 V/m

Narrowband response (resonant-shell model):
     315.0 MHz  E = 4.050e-04 V/m  (n_e = 1.227e+15 m^-3, t = 7.999e-06 s, kr = 1.98)
     916.0 MHz  E = 9.207e-04 V/m  (n_e = 1.095e+16 m^-3, t = 3.866e-06 s, kr = 5.91)
```

Two different amplitudes, deliberately. The broadband `peak |E|` is the whole
pulse. The narrowband values are what an antenna tuned to that frequency sees,
because only the shell with `omega_pe = 2*pi*f` radiates into that band *and*
can escape — anything below the local plasma frequency is evanescent.
**The narrowband number is the one to compare with published antenna
measurements.**

`kr` tells you whether the sensor is in the far field. At 315 MHz, `kr = 1.98`
— the antenna sits in the transition zone, so the near-field terms matter and
are included (THEORY §5.5).

### Stage 4 — coupling, and the verdict

```
Surface charging:
  peak floating potential        -5.69 V
  peak differential V             5.69 V
  ESD threshold (500 V)      not exceeded

EMP coupling:
  loop-induced voltage       6.914e-04 V
  after shielding            6.914e-06 V
  received power                 -44.3 dBm
  CMOS upset (0.5 V)        below threshold

Anomaly verdict: no threshold exceeded for the assumed configuration
```

A picogram at 50 km/s is far from a threat to this assumed configuration. See
`examples/04_olympus_anomaly.py` for the mass at which it becomes one.

### Validity warnings — always read these

```
Validity warnings:
  ! impact speed 50 km/s is far beyond the validated linear Us-Up range; ...
  ! initial coupling parameter Gamma = 1.01 > 1: the plasma starts strongly coupled, ...
```

They are not decoration. They tell you which physics in the chain is being
extrapolated for your particular case, and they are collected in
`sc.warnings` so you can assert on them in scripts.

---

## 4. Running the stages individually

Use this when you want to vary one stage while holding the others fixed, or
substitute your own model for a stage.

```python
import numpy as np
from hvi_emp import (ALUMINIUM, IRON, Projectile, simulate_impact,
                     simulate_expansion, simulate_emp, simulate_charging,
                     couple_to_structure, emission_at_frequency)

# --- Stage 1: impact, shock, vaporisation, ionisation -------------------
proj = Projectile(IRON, mass=1e-12, velocity=50e3, angle_deg=0.0)
imp = simulate_impact(proj, ALUMINIUM, n_decay=2.0, core_scale=1.0,
                      plume_expansion_factor=3.0, warn=False)
print(f"P = {imp.P_ic/1e9:.0f} GPa, m_plasma = {imp.m_plasma:.3e} kg, "
      f"T_e = {imp.plasma.T_eV:.2f} eV, Zbar = {imp.plasma.Zbar:.3f}")

# --- Stage 2: expansion into vacuum, freeze-out ------------------------
exp = simulate_expansion(imp, t_end=2e-5, n_shell=24)
print(f"Q_frozen = {exp.Q_final:.3e} C")
print(f"transition at n_e = {exp.transition['n_e']:.3e} m^-3")

# --- Stage 3: radiated EMP ---------------------------------------------
em = simulate_emp(exp, r_sensor=0.30, theta_deg=90.0, f_max=5e9,
                  closure="debye")
print(f"peak |E| = {em.peak_field:.3e} V/m")
band = emission_at_frequency(exp, 916e6, r_sensor=0.30)
print(f"916 MHz -> {band['E_peak']:.3e} V/m")

# --- Stage 4: spacecraft coupling --------------------------------------
chg = simulate_charging(exp, standoff=0.10, dielectric_thickness=1.27e-4)
cpl = couple_to_structure(em, loop_area=1e-2, wire_length=0.10,
                          shielding_dB=40.0)
print(f"peak differential = {chg.peak_differential_V:.1f} V, "
      f"induced (shielded) = {cpl.shielded_V:.3e} V")
```

Each stage consumes the previous stage's result object and nothing else, so
you can replace any one of them. See §10.

---

## 5. Complete parameter reference

### 5.1 `run_scenario` — everything, with defaults

| parameter | default | meaning |
|---|---|---|
| `projectile_material` | — | name (`"Fe"`) or a `Material` |
| `target_material` | — | name (`"Al"`) or a `Material` |
| `mass` | — | projectile mass [kg] |
| `velocity` | — | impact speed [m/s] |
| `angle_deg` | `0.0` | incidence from surface normal [deg] |
| `n_decay` | `2.0` | shock-decay exponent, `P ~ r^-n` |
| `core_scale` | `1.0` | isobaric-core radius / projectile radius |
| `plume_expansion_factor` | `3.0` | Stage-1 → Stage-2 handoff density |
| `t_end` | `None` | expansion end time [s]; auto if `None` |
| `r_sensor` | `0.30` | EMP sensor standoff [m] |
| `theta_deg` | `90.0` | angle from dipole axis; 90° = broadside max |
| `f_max` | `5e9` | top of the reported spectrum [Hz] |
| `separation_factor` | `1.0` | charge displacement / Debye length |
| `closure` | `"debye"` | `"debye"` (conservative) or `"fletcher"` (upper bound) |
| `bands` | `(315e6, 916e6)` | narrowband antenna frequencies [Hz] |
| `standoff` | `0.10` | distance to the charged surface [m] |
| `loop_area` | `1e-2` | worst-case harness loop area [m²] |
| `wire_length` | `0.10` | exposed conductor length [m] |
| `shielding_dB` | `40.0` | enclosure shielding effectiveness [dB] |
| `verbose` | `False` | print the summary as it runs |

### 5.2 The four parameters that are *not* fixed by first principles

These are the ones to vary when you quote an uncertainty. Measured leverage
(Fe 1 pg → Al at 50 km/s, from `examples/03_sensitivity.py`):

| parameter | defensible range | why it is uncertain | spread in E@916 MHz |
|---|---|---|---|
| `n_decay` | 1.5 – 3.0 | hydrocode fits give 1.2 near-field, ~3 far-field | 2.1× |
| `core_scale` | 0.5 – 1.5 | isobaric-core size is only approximately the projectile radius | 2.1× |
| `plume_expansion_factor` | 2 – 5 | where the plume stops being strongly coupled is not sharp | 8.6× |
| `separation_factor` | 0.5 – 2 | the O(1) coefficient in `delta = xi * lambda_D` | 4.0× |

Combined with the closure choice, **absolute EMP amplitude is uncertain by
roughly two orders of magnitude.** Charge yield is good to ~1 decade;
vaporisation thresholds and velocity *scalings* are the reliable outputs.

### 5.3 Stage-level parameters not exposed by `run_scenario`

| function | parameter | default | meaning |
|---|---|---|---|
| `simulate_expansion` | `n_shell` | `24` | Lagrangian shells in the Gaussian profile |
| | `n_out` | `400` | output time samples |
| | `aspect0` | `1.0` | initial `R_z / R_r` |
| | `include_bulk_drift` | `True` | give the plume a centre-of-mass velocity |
| | `rtol`, `atol` | `1e-8`, `1e-14` | ODE tolerances |
| `simulate_emp` | `n_freq` | `4096` | spectrum resolution |
| | `n_epochs` | `60` | plume epochs sampled as dipole sources |
| | `coherent` | `True` | coherent (upper) vs random-phase (lower) sum |
| `simulate_charging` | `dielectric_thickness` | `1.27e-4` | 5 mil Kapton [m] |
| | `dielectric_eps_r` | `3.4` | Kapton; use ~4 for coverglass |
| | `secondary_yield` | `0.3` | secondary-electron emission coefficient |
| `couple_to_structure` | `antenna_gain_dBi` | `0.0` | for the received-power estimate |

---

## 6. Result-object reference

### `Scenario`
`impact`, `expansion`, `emp`, `charging`, `coupling`, `anomaly` (dict),
`bands` (dict keyed by frequency), `warnings` (list of str), `.summary()`.

Any of `expansion`/`emp`/`charging`/`coupling` may be `None` if the chain
stopped early — always check, and read `warnings` to find out why.

### `ImpactResult`
| field | meaning |
|---|---|
| `P_ic`, `r_ic` | peak shock pressure [Pa], isobaric-core radius [m] |
| `m_vapour`, `m_plasma`, `m_melt_target` | mass inventories [kg] |
| `m_vapour_projectile`, `m_vapour_target` | split by origin [kg] |
| `plasma` | `IonisationState` — `.T`, `.T_eV`, `.n_e`, `.Zbar`, `.fractions` |
| `Q_free` | free charge at formation [C] |
| `material_mix` | the synthetic mixed-species `Material` |
| `u_atom` | internal energy per heavy particle above free atoms [J] |
| `E_res_projectile`, `E_res_core` | release waste heat [J/kg] |
| `n_h0`, `rho_plume0`, `r_plume0`, `v_expansion` | Stage-2 initial condition |
| `thresholds` | phase-change thresholds for both materials |
| `diagnostics` | shell integrals, energy budget, two-phase charge, …|
| `.vaporisation_efficiency`, `.plasma_efficiency`, `.charge_per_mass` | ratios |

### `ExpansionResult`
Arrays over time `t`: `R_r`, `R_z`, `v_r`, `v_z`, `T`, `T_eV`, `Zbar`,
`Zbar_eq` (local Saha equilibrium, for comparison), `n_h`, `n_e`, `omega_pe`,
`nu_ei`, `lambda_De`, `Gamma_coupling`, `Q_free`.

Dicts: `freeze` (mass-weighted freeze-out plus per-shell arrays), `transition`
(the `nu_ei = omega_pe` crossing: `t`, `n_e`, `T_eV`, `f_pe`, `lambda_De`, …),
`shells`, `diagnostics` (includes `energy["relative_drift"]` — the ODE
conservation check).

`.Q_final` is the surviving charge. Use it, not `impact.Q_free`, for
comparison with experiment.

### `EMPResult`
`f`, `E_spec` (complex), `t`, `E_t`, `B_t`, `r_sensor`, `theta_deg`,
`peak_field`, `energy_radiated`, `sources` (list of `ChargeSeparation`),
`.psd`, `.band_field(f_lo, f_hi)`.

### `emission_at_frequency(...)` → dict
`E_peak` (exact dipole, near + far field), `E_far_field_only`, `n_e`, `t`,
`lambda_De`, `Q_separated`, `delta`, `p0`, `omega`, `gamma`, `Q_factor`,
`kr`, `far_field`, `dipole_approx_valid`, `reached`.

Check `reached` before using `E_peak` — it is `False` if the plume never
passes through the resonant density.

### `ChargingResult` / `CouplingResult`
`peak_differential_V`, `esd_risk`, `phi_float`, `V_dielectric`, `Q_surface`;
`V_loop_peak`, `V_monopole_peak`, `shielded_V`, `received_power_dBm`,
`upset_risk`. Both have `.summary()`.

`V_dielectric` is clipped at the dielectric breakdown voltage — beyond that
the material discharges and the capacitor model does not apply. Check
`charging.diagnostics["breakdown_saturated"]`; the unclipped values are kept
in `diagnostics["V_dielectric_unclipped"]` for diagnosis.

---

## 7. Recipes

### 7.1 Bracket the EMP amplitude honestly

```python
from hvi_emp import run_scenario

for closure in ("debye", "fletcher"):
    sc = run_scenario("Fe", "W", mass=1e-16, velocity=50e3,
                      t_end=1e-4, closure=closure)
    print(closure, f"{sc.bands[916e6]['E_peak']:.3e} V/m")
```

Report the range, not either endpoint. Close et al.'s measured 1.9 mV/m falls
inside it.

### 7.2 Charge yield versus impact speed

```python
import numpy as np
from hvi_emp import velocity_sweep

res = velocity_sweep("Fe", "Al", mass=1e-12,
                     velocities=np.linspace(30e3, 72e3, 12), t_end=1e-5)

# Fit only above the vaporisation threshold -- including the threshold region
# gives a much steeper exponent that is not comparable with published fits.
hi = (res["Q_frozen"] > 0) & (res["v"] >= 40e3)
beta = np.polyfit(np.log(res["v"][hi]), np.log(res["Q_frozen"][hi]), 1)[0]
print(f"beta (above 40 km/s) = {beta:.2f}")

lo = (res["Q_frozen"] > 0) & (res["v"] < 45e3)
beta_lo = np.polyfit(np.log(res["v"][lo]), np.log(res["Q_frozen"][lo]), 1)[0]
print(f"beta (threshold region) = {beta_lo:.2f}")
```

`velocity_sweep` returns `v`, `Q_stage1`, `Q_frozen`, `T_eV`, `Zbar`,
`vapour_efficiency`, `E_916MHz`, `Q_empirical`.

The two exponents differ by a factor of two or more, which is the point:
`beta` is range- *and* definition-dependent, and published values (3.4–3.5)
come from fits that span the threshold. Read THEORY §6 before quoting it.

### 7.3 Find the threshold speed for a material pair

```python
import numpy as np
from hvi_emp import Projectile, simulate_impact, get_material

proj_mat, tgt = get_material("Fe"), get_material("Al")
for v in np.arange(10e3, 60e3, 1e3):
    r = simulate_impact(Projectile(proj_mat, 1e-12, v), tgt, warn=False)
    if r.m_plasma > 0:
        print(f"plasma threshold: {v/1e3:.0f} km/s")
        break
```

### 7.4 Sweep the incidence angle

```python
from hvi_emp import run_scenario

for a in (0, 30, 45, 60):
    sc = run_scenario("Fe", "Al", mass=1e-12, velocity=60e3,
                      angle_deg=a, t_end=1e-5)
    q = sc.expansion.Q_final if sc.expansion else 0.0
    print(f"{a:3d} deg: P = {sc.impact.P_ic/1e9:7.0f} GPa, Q = {q:.3e} C")
```

Only the normal velocity component couples into the shock; beyond 60° the
framework warns you that this is degrading.

### 7.5 Check whether your vacuum chamber is a vacuum *for the plume*

```python
from hvi_emp import stopping_distance, torr_to_number_density

for torr in (1e-6, 1e-3, 1.0):
    n_b = torr_to_number_density(torr)
    d = stopping_distance(n_b, n_plume0=1e27, R0=1e-5)
    print(f"{torr:.0e} Torr -> plume stops after {d*1e3:.2f} mm")
```

Below ~10⁻⁵ Torr the plume expands freely; at 1 Torr it is arrested within a
millimetre, and the plasma-oscillation mechanism cannot operate.

### 7.6 Confirm the dust cloud does not shield the EMP

```python
from hvi_emp import dust_optical_depth

for lam, label in ((550e-9, "550 nm"), (0.327, "916 MHz")):
    d = dust_optical_depth(n_dust=1e16, radius=1e-7, path=0.05, wavelength=lam)
    print(f"{label:>8}: x = {d['size_parameter']:.2e}, "
          f"tau = {d['optical_depth']:.2e}, "
          f"significant = {d['significant']}")
```

### 7.7 Scan a spacecraft configuration for ESD risk

```python
from hvi_emp import run_scenario

for standoff in (0.02, 0.05, 0.10, 0.30, 1.00):
    sc = run_scenario("Olivine", "Al", mass=1e-6, velocity=59e3,
                      standoff=standoff)
    d = sc.charging.diagnostics
    print(f"standoff {standoff*100:5.1f} cm: "
          f"V_diff = {sc.charging.peak_differential_V:8.1f} V, "
          f"ESD = {str(sc.charging.esd_risk):<5} "
          f"breakdown = {str(d['breakdown_saturated']):<5} "
          f"(uncapped {abs(d['V_dielectric_unclipped']).max():.2e} V)")
```

The reported potential saturates at the dielectric breakdown voltage
(2540 V for 127 µm Kapton). That clip is physical: past it the dielectric
discharges, and the meaningful statement is "it broke down", not a voltage.
The uncapped capacitor-model value is kept in `diagnostics` for diagnosis
only — for a surface engulfed by the plume it reaches ~5×10⁸ V, which is
five orders of magnitude beyond what the material can hold.

### 7.8 Batch runs to JSON

```bash
for v in 30e3 40e3 50e3 60e3 72e3; do
  python -m hvi_emp --quiet run --velocity $v --t-end 1e-5 --json run_$v.json
done
```

---

## 8. Adding your own materials

Eight are built in: `Al`, `Fe`, `W`, `Cu`, `SiO2`, `Kapton`, `Olivine`,
`Dolomite` (`python -m hvi_emp materials` prints them with their parameters).

```python
from hvi_emp import ALUMINIUM, Material, Projectile, simulate_impact
from hvi_emp.constants import EV

NICKEL = Material(
    name="Ni",
    rho0=8874.0,          # kg/m^3
    c0=4602.0,            # m/s   intercept of the linear Us-Up fit
    s=1.437,              # -     slope of the linear Us-Up fit
    gamma0=1.93,          # -     Gruneisen parameter at rho0
    cv_solid=444.0,       # J/(kg K)
    T_melt=1728.0, T_vap=3186.0,          # K
    L_fusion=2.91e5, L_vap=6.31e6,        # J/kg
    E_cohesive=4.44 * EV,                 # J per atom
    A=58.693, Z=28,
    E_ion=(7.6398 * EV, 18.1688 * EV, 35.19 * EV),   # NIST, J
    g_ion=(9.0, 6.0, 9.0, 10.0),                     # ground-term weights
    work_function=5.15 * EV,
    us_up_valid_to=1.5e4,   # m/s, above which the linear fit is extrapolated
)

r = simulate_impact(Projectile(NICKEL, 1e-12, 50e3), ALUMINIUM, warn=False)
print(f"{r.plasma.T_eV:.2f} eV, Zbar = {r.plasma.Zbar:.3f}")
```

**Where to get the data.** `c0`, `s`, `gamma0` from *LASL Shock Hugoniot Data*
(Marsh 1980) or Steinberg (LLNL UCRL-MA-106439); thermophysics from the CRC
Handbook; ionisation potentials and ground-state degeneracies from the NIST
Atomic Spectra Database.

**Sanity-check a new material before trusting it.** Run these three checks on
anything you add:

```python
from hvi_emp import ALUMINIUM, IRON, Material, TUNGSTEN
from hvi_emp.constants import EV
from hvi_emp.eos import (cohesive_energy_vinet, compression_limit_velocity,
                         phase_thresholds)

# The titanium parameters below are real (Marsh 1980) and deliberately chosen:
# titanium is a material this framework should NOT be used for, and the checks
# are what tell you so.
TITANIUM = Material(
    name="Ti_demo", rho0=4510.0, c0=5220.0, s=0.767, gamma0=1.23,
    cv_solid=523.0, T_melt=1941.0, T_vap=3560.0,
    L_fusion=2.96e5, L_vap=8.88e6, E_cohesive=4.85 * EV,
    A=47.867, Z=22,
    E_ion=(6.8281 * EV, 13.5755 * EV, 27.4917 * EV),
    g_ion=(21.0, 28.0, 21.0, 10.0), work_function=4.33 * EV)

for mat in (ALUMINIUM, IRON, TUNGSTEN, TITANIUM):
    # 1. Vinet binding energy vs the tabulated sublimation energy. Derived
    #    from rho0, c0, s alone -- a free check with no fitted parameters.
    ratio = cohesive_energy_vinet(mat) / mat.E_sublimation

    # 2. Compression limit of the linear Us-Up fit. Materials with s < 1
    #    become unphysical at accessible speeds.
    lim = compression_limit_velocity(mat)

    # 3. Phase-change thresholds, to compare with any shock data you have.
    th = phase_thresholds(mat)

    verdict = "OK" if (0.5 < ratio < 2.0 and th["P_vap"] < 1e13) else "REJECT"
    print(f"{mat.name:<9} binding ratio {ratio:5.2f}  "
          f"limit {lim/1e3:7.1f} km/s  "
          f"P_melt {th['P_melt']/1e9:7.0f} GPa  "
          f"P_vap {th['P_vap']/1e9:8.0f} GPa   {verdict}")
```

```
Al        binding ratio  0.85  limit     inf km/s  P_melt      94 GPa  P_vap     1504 GPa   OK
Fe        binding ratio  0.45  limit     inf km/s  P_melt     198 GPa  P_vap     1352 GPa   REJECT
W         binding ratio  1.60  limit     inf km/s  P_melt     356 GPa  P_vap     2839 GPa   OK
Ti_demo   binding ratio  9.77  limit    22.4 km/s  P_melt     inf GPa  P_vap      inf GPa   REJECT
```

Read that table as follows.

A **binding-energy ratio** far from 1 means the linear `Us`-`Up` fit is poor
for that material — usually a phase transition folded into the fit. Iron's
0.45 is the known α→ε problem (THEORY §2.3); it is retained because iron is
unavoidable for meteoroid work, but its vaporisation thresholds carry a
correspondingly larger error. Titanium's 9.77 is a hard rejection: `s = 0.767`
gives `K0' = 2.07`, so the `(K0'-1)²` denominator in the Vinet binding energy
nearly vanishes and the cold curve is meaningless.

A **finite compression limit** is a ceiling, not a warning. Above it the fit
implies infinite density, and `hugoniot_state` raises a `ValueError` naming the
limit rather than failing obscurely. Titanium hits it at 22.4 km/s particle
velocity — roughly 45 km/s impact onto a light target, well inside the
meteoroid range.

**`P_melt = inf`** means the threshold is unreachable within the fit's
validity. For titanium the compression limit arrives before the material can
be melted on release, so the framework can say nothing about it at all.

For materials like these, substitute a tabular EOS (SESAME/ANEOS) in
`eos.hugoniot_state` — the rest of the chain is unaffected.

---

## 9. Troubleshooting

**`ValueError: No vapour produced by this impact`**
Below the complete-vaporisation threshold there is no superheated plasma, so
Stage 2 has nothing to expand. This is physics, not a bug. `run_scenario`
handles it gracefully and returns a `Scenario` with `expansion is None` and an
explanatory warning; only the low-level `simulate_expansion` raises. Raise the
impact speed, or use a lower-threshold target such as `Dolomite`.

**`no collisional->collisionless transition inside the integration window`**
Increase `t_end` (try 10–100×). The default is auto-scaled but can be short
for large, slow plumes.

**`RuntimeWarning: particle velocity ... exceeds the validated linear Us-Up range`**
Expected above ~30 km/s. Silence with `warnings.simplefilter("ignore")` or
`--quiet`, but note the caveat: the linear EOS is being extrapolated and a
tabular EOS should be substituted for quantitative work.

**`Gamma = ... > 1: the plasma starts strongly coupled`**
Informational. Raise `plume_expansion_factor` to hand off later at lower
density if you want to stay in the weakly-coupled regime — but check the
sensitivity, because that parameter has a lot of leverage (§5.2).

**`emission_at_frequency` returns `reached: False`**
The plume never passes through the resonant density for that frequency.
Increase `t_end` (for low frequencies, which need a very dilute late plume) or
pick a frequency inside the range spanned by `exp.n_e`.

**`python setup.py` prints "error: no commands supplied"**
That file is not the installer — it is a compatibility shim. Use
`pip install -e .`. (It now prints usage instead of the bare setuptools
error.)

**`lammps python module not available` from example 05**
Expected if the LAMMPS bindings are not installed — the example writes a
complete input deck instead and tells you its absolute path and how to run it
with the standalone `lmp` binary. To enable in-process runs:
`pip install lammps mpich`, or `mamba install -c conda-forge lammps` in a
conda environment. `hvi_emp.solvers.lammps_diagnostics()` distinguishes "not
installed" from "installed but the shared library will not load" (usually a
missing MPI runtime) and gives the matching fix.

**`ImportError: attempted relative import with no known parent package`**
You are running a file from inside `hvi_emp/` directly, e.g.
`python lammps_stage1.py`. Those are package modules, not scripts: the `..`
imports only resolve when Python knows the parent package. Do not move them
out of `hvi_emp/solvers/`. Use `python -m hvi_emp.solvers.lammps_stage1`,
`python -m hvi_emp lammps --help`, `python examples/05_lammps_impact.py`, or
import the module — all from the project root. The solver modules now detect
direct execution and print these options instead of the raw traceback.

**`ModuleNotFoundError: No module named 'matplotlib'` running an example**
matplotlib is optional and used only for the figures. The examples now detect
its absence, print all the numerical results, and skip the PNG with a note —
so if you saw a traceback, take the newer package. To get the figures:
`pip install matplotlib`, or `mamba install -c conda-forge matplotlib`, or
`pip install -e ".[plots]"`.

**`ImportError: Cannot import packaging.licenses` when installing**
Your environment has setuptools ≥ 77 with `packaging < 24.2` — a very common
combination (Homebrew Python, conda, any venv whose setuptools was upgraded
in isolation). Current `pyproject.toml` avoids the PEP 639 SPDX licence field
entirely, so this cannot occur; if you hit it on an older copy, either
`pip install -U "packaging>=24.2"` or take the newer package.

**`SetuptoolsDeprecationWarning: project.license as a TOML table is deprecated`**
An older copy of the package. The licence field is now omitted from
`[project]` altogether (see the comment in `pyproject.toml` explaining why
neither spelling is safe).

**`SetuptoolsWarning: install_requires overwritten in pyproject.toml`**
An old `setup.cfg` duplicated the dependency list. `setup.cfg` is now empty
and `pyproject.toml` is the single source of truth.

**Package installs as `UNKNOWN 0.0.0`**
The build backend is older than setuptools 61 and cannot read `[project]`.
`setup.py` now detects this and fails with instructions instead of installing
something wrong. Fix with `pip install -U "setuptools>=61" pip`, or
`pip install --no-build-isolation .` if your environment already has a newer
setuptools than the isolated build environment can fetch. You can also skip
installation entirely — the package is a plain directory.

**First run is slow (~3 s), later ones are fast**
The Saha table is built once per material and cached. Mixed plumes are
quantised to 10% composition steps precisely to keep the number of tables
small.

**Results changed after I edited a material**
Clear the caches — `phase_thresholds` and `spinodal_volume` are
`lru_cache`d on the `Material`, and the ionisation tables are cached by name.
Restart the interpreter, or use a different `name` for the edited material.

---

## 10. Extending the framework

Each stage's interface is a small physical state vector, so stages can be
swapped independently.

**Using the production-solver bridges.** `hvi_emp.solvers` already implements
the three substitutions below for LAMMPS (Stage 1, runs in-process), WarpX
(Stage 3, deck generation + openPMD post-processing) and OpenMHD (Stage 2
magnetised, `model.f90` generation) — see [`SOLVERS.md`](SOLVERS.md) before
rolling your own.

**Substituting a real hydrocode for Stage 1.** Build an `ImpactResult` (or a
duck-typed object) with the fields `simulate_expansion` actually reads:
`m_plasma`, `material_mix`, `plasma` (an `IonisationState`), `r_plume0`,
`v_expansion`. Everything else in `ImpactResult` is diagnostics.

**Substituting a PIC code for Stage 3.** `simulate_emp` and
`emission_at_frequency` read only `t`, `n_e`, `T_eV`, `R_r`, `R_z`, `nu_ei`,
`v_r`, `v_z`, `material` and `transition` from the `ExpansionResult`. Write
those out as PIC initial conditions, or replace Stage 3 entirely and keep
Stages 1–2 as the initial-condition generator.

**Adding a charge-separation closure.** Add a branch to
`emp.separation_from_shell`; it is ~15 lines. The two existing closures are
deliberately the conservative and aggressive extremes.

**Adding a validation case.** Append a `Check(name, model, reference, unit,
tolerance, source, note)` to the relevant function in `validation.py`. Keep
the citation in `source` — every entry in that file is somebody else's
measurement, and that is what makes the suite meaningful.

**Before committing a change**, run both gates:

```bash
python -m pytest tests/ -q      # must stay 61/61 — these are conservation laws
python -m hvi_emp validate      # must stay 31/31
```

The test suite checks identities that *cannot* be traded off (Rankine-Hugoniot,
`P_c = -dE_c/dV`, Saha charge conservation, ODE energy conservation,
`B = E/c`, the far-field `1/r` limit). A failure there is a bug, not a
modelling choice.

---

## 11. Reading the results responsibly

**Trust these:**

* shock pressures and vaporisation thresholds (validated to ~1.0 for Al with
  no free parameters);
* relative comparisons — which impact, material or geometry is worse;
* velocity and mass *scalings*;
* the qualitative pathway verdict in `anomaly_assessment`.

**Treat these as order-of-magnitude:**

* absolute charge yield (~1 decade);
* absolute EMP field amplitude (~2 decades, hence the closure bracket);
* induced voltages, which inherit all of the above plus assumed geometry.

**Do not use the framework for:**

* root-cause determination of a specific anomaly on its own;
* impacts above ~30 km/s without substituting a tabular EOS;
* incidence beyond ~60° from normal;
* millimetre-and-larger impactors, where radiative cooling and
  optically-thick plume physics enter;
* the emitted *frequency* — the predicted spectral peak is orders of magnitude
  above what is measured, an open problem that Fletcher & Close report from
  their PIC simulations too.

THEORY §7 lists all nine known physics gaps. When you quote a number from this
framework, quote the uncertainty from §5.2 with it.
