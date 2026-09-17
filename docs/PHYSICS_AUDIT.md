# Physics audit

`scripts/physics_audit.py` stress-tests statements that **cannot be false** —
conservation laws, thermodynamic identities, limiting behaviour, cross-module
agreement — over a wide sweep of materials, velocities and densities.

```bash
python scripts/physics_audit.py       # 516 checks; exits non-zero on failure
```

It is not the test suite. The test suite guards behaviour we chose; the audit
looks for behaviour physics forbids. Run it after touching any of `eos.py`,
`ionization.py`, `impact.py`, `expansion.py` or `emp.py`.

**Current status: 516 pass, 0 fail, 1 warning, 2 notes.**

---

## What it found

Three real defects, all in Stage 1, all of which produced visibly unphysical
output. Each now has a regression test in `tests/test_hvi_emp.py`.

### 1. Composition was quantised to 10%, and the physics moved in steps

`mix_materials` rounded the projectile/target mass fraction to one decimal
place *before* building the synthetic mixture, so the mean atomic mass jumped
in coarse steps (26.98 → 85.00 → 67.00 → 55.28 → 47.06 amu) as impact speed
rose smoothly.

Consequence: **plume temperature fell as impact speed rose**, four times over
28–62 km/s. A velocity sweep produced a sawtooth in T_e, Z̄ and charge.

The quantisation existed to keep the ionisation-table cache small (each table
costs ~1.5 s to build, and an exact composition per velocity means a fresh
table per velocity). That is a real cost, but it was paid in the wrong
currency.

**Fix.** The mixture is now exact. The cache problem is solved where it
belongs — `ionization.MixtureTable` builds tables on a coarse composition grid
*lazily* and interpolates between neighbours. Z̄ and the energy inversion are
smooth in composition, so interpolating between compositions 10% apart is
accurate to well under a percent, while composition itself enters
continuously.

### 2. The projectile's plasma mass was all-or-nothing

```python
m_plasma_p = proj.mass if E_res_p >= th_p["E_vap"] else 0.0
```

Below the complete-vaporisation threshold the projectile contributed *no*
plasma; above it, *all* of it. For W→Al this jumped from 0 to the entire
projectile mass between 35.4 and 35.6 km/s, putting a step into plasma mass,
charge, temperature and every downstream EMP quantity — in the middle of the
velocity range the framework is used for.

The justification in the comment was that the planar impact approximation
shocks the projectile uniformly. But the *target* gets a proper shell integral
over its decaying pressure field, and the projectile sits in the other half of
that same field.

**Fix.** `_projectile_integrals` integrates the projectile against the same
peak-pressure field, over the exact geometry of a ball touching a plane:

    dm/dr = rho0 (2 pi r^2 - pi r^3 / a),   0 <= r <= 2a

which integrates to exactly (4/3)π a³ ρ₀. No fitting, no free parameter.

### 3. The isobaric core had a kink, and it was worth exactly 5/16

Even after (2), a step remained — smaller, but still a step. Cause: the peak
pressure was written piecewise, constant inside the isobaric core and a power
law outside. A *perfectly flat* core means every gram inside it crosses the
vaporisation threshold at the same impact speed. For a sphere touching a
plane the core holds exactly 5/16 = 0.3125 of the projectile mass — and that
31.25% appeared all at once, which is precisely the step height observed.

**Fix.** `peak_pressure_profile` uses

    P(r) = P_ic / (1 + (r/r_ic)^n)

which has both correct asymptotes (→ P_ic for r ≪ r_ic, → P_ic (r/r_ic)^−n for
r ≫ r_ic), is monotone and smooth, and introduces no new parameter. The
piecewise form remains available as `smooth=False`.

---

## What the fixes changed

| quantity | before | after | reference |
|---|---|---|---|
| W→Al complete-vaporisation threshold | 36 km/s | **26 km/s** | Fletcher 10–20 km/s |
| ratio to Fletcher | 2.06 | **1.49** | — |
| spurious T_e drops over 28–62 km/s | 4 | **0** | — |
| charge exponent β at formation (Fe→Al) | 3.2 | 4.5 | 3.48 measured |
| charge exponent β frozen (Fe→Al) | 5.0 | 6.5 | 3.48 measured |

The threshold moved substantially **toward** Fletcher, which is the headline
result: a graded projectile is both more physical and closer to the reference.

The charge-yield exponent moved **away**, and that is now the framework's
largest open discrepancy — see below.

---

## Two more defects, in the two-temperature model

Both were found by asking the 2T model for its *conserved* quantity rather
than for its answer, and both had survived the 201-test suite and all 509
audit checks — because nothing had ever looked.

### 4. An ion-temperature floor was manufacturing energy

`unpack` clamped both temperatures with `max(y[6], 50.0)`. Once the ions
cooled to 50 K the ODE kept integrating them downward while `rhs` kept
*reading* 50 K, so the ion pressure `P_i = n_h k T_i` never went to zero and
went on doing work against the expansion for the rest of the run.

It invented **+1.31% of the plume's total energy**, and the giveaway was that
the figure was *exactly* 1.307% at rtol = 10⁻⁶, 10⁻⁸, 10⁻¹⁰ and 10⁻¹². Drift
that ignores tolerance is not integration error; it is a term in the model.

**Fix.** Evolve `ln T_e` and `ln T_i` instead of `T_e` and `T_i`. Positivity
becomes structural, no clamp is needed, and no clamp is applied. Both
temperatures are now free to decay through arbitrarily small values — T_i
reaches 8×10⁻⁶ eV by 10 µs where it used to sit at the 50 K floor. Drift
falls from 1.31% to below 10⁻⁶.

The electron equation still divides by Z̄ when inverting for dT_e. Below
`Z_FLOOR = 1e-9` the model now switches to the physical limit (a vanishing
electron population cools adiabatically with the flow) rather than dividing
by a floored denominator. That is a change of closure in a regime where the
electron gas carries no energy, not a clamp on the state.

### 5. `v_z` meant two different things on the same dataclass

`expansion.py` returns `v_z = vz + v_cm` — expansion rate *plus* the plume's
bulk drift. `expansion_2t.py` returned the bare expansion rate. Same field,
same `ExpansionResult` type, different physics.

Read naively, the 2T plume appeared to expand at 2.40 km/s against 27.54 for
1T — an 11× discrepancy that looked like a broken momentum equation. The real
difference is 2.40 vs 2.56 km/s, i.e. **6%**, entirely explained by
recombination energy going to the electrons alone instead of being shared
with the ions.

A related consequence: the first energy audit used `v_z` directly and so
reported a 28.8% energy loss that did not exist. The kinetic energy of a
Gaussian plume is `½m(2Ṙ_r² + Ṙ_z²)` — **two** transverse axes — computed
from the drift-free rates. `energy_budget()` now does this once, correctly,
so it cannot be got wrong ad hoc again.

**Fix.** 2T returns `vz + v_cm`, matching the 1T contract. Three regression
tests pin it, including one asserting that `include_bulk_drift=False` removes
exactly that offset.

A third, smaller instance of the same disease: the 2T model had its own
private copy of the collisional→collisionless search, which omitted the
`f_pe` key that Stage 3 reads by name. The first end-to-end 2T scenario
crashed in `simulate_emp`. It now calls `expansion._find_transition`, so
there is one definition and one contract.

---

## The open discrepancy, and what closed it

**Charge-yield velocity exponent: 1T model 6.70, measured 3.48.**

It decomposes cleanly:

| stage | contribution to β |
|---|---|
| Stage 1, at formation | ~4.5 |
| Stage 2, added by freeze-out | ~+2.0 |
| total, frozen (what an RPA measures) | ~6.5 |

The freeze-out amplification is self-consistent, not a coding error:
three-body recombination goes as n_e² T^−4.5, and the plume temperature here
rises very nearly linearly with impact speed, so the surviving charge fraction
climbs from 5.6% at 36 km/s to 21% at 72 km/s — an exponent of +2.0 on its
own.

Two candidate mechanisms were named here *before* either was implemented.
Both have now been built and measured, and they did not contribute equally.

**Radiative cooling: ruled out.** `radiation.py` implements bremsstrahlung and
radiative recombination with a Kramers escape factor. The radiative cooling
time exceeds the expansion time by ~54× at plume formation and by six further
orders of magnitude as the plume thins. Effect on β: **−0.26**. The term is
kept (it is correct, and it matters for denser or slower plasmas) but it does
not explain the discrepancy. Note this is a *lower* bound: line radiation is
omitted, and for a partially-ionised metal vapour near 1 eV it is normally the
dominant channel. The conclusion survives a 10× underestimate; it would not
survive 100×.

**A two-temperature plume: this is the mechanism.** `expansion_2t.py` evolves
T_e and T_i separately with non-equilibrium ionisation per mass shell.

| Stage-2 closure | β (40–66 km/s) |
|---|---|
| at formation (Stage 1 only) | 4.68 |
| frozen, 1-temperature | **6.70** |
| frozen, 2-temperature | **3.65** |
| measured (Close 2013) | **3.48** |

Decomposing the 2T result by switching its terms off one at a time:

| variant | β | charge surviving at 40 km/s | at 66 km/s |
|---|---|---|---|
| full 2T | 3.65 | 0.686 | 0.408 |
| no recombination heating | 4.09 | 0.005 | 0.004 |
| no e–i equilibration | 4.07 | 1.038 | 0.763 |
| neither | 4.43 | 0.010 | 0.009 |
| (1-temperature, for reference) | 6.70 | 0.061 | 0.168 |

Two things to read out of this.

*Recombination heating dominates the surviving charge.* Without it, 0.5% of
the charge survives; with it, 41–69%. Each three-body capture returns its
binding energy to the electron gas, the rate goes as T_e^−4.5, and the
resulting negative feedback throttles further recombination. In a
one-temperature plume that energy is shared with ions carrying ~2.7× the heat
capacity per particle, the electrons stay cold, and the plasma
over-recombines.

*But no single term explains β.* Every 2T variant lands in 3.65–4.43, all far
below the 1T value of 6.70. Recombination heating and e–i equilibration each
contribute about −0.4 and are not additive. The flattening comes from
two-temperature physics as a whole, not from one favoured term — which is the
more believable outcome, and is what makes the sign flip meaningful: 1T
freeze-out *adds* 2.0 to β, 2T freeze-out *subtracts* 1.0.

The remaining 4.4% gap (3.65 vs 3.48) is well inside the framework's other
uncertainties, and the comparison is still not strictly like for like: Close's
exponent is fitted **across** the vaporisation threshold over 3–66 km/s,
whereas this is an above-40 km/s fit.

**This is not treated as "case closed".** The 1T model remains the default,
the check remains marked `OPEN`, and `Check` still carries its `known_open`
field: such a check runs, prints its number, and is visibly flagged, but does
not count as a regression. A validation suite that must be 100% green is a
suite that will be tuned until it is.

### Using it

```python
sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3,
                  expansion_model="2T")     # default is "1T"
```

2T costs ~4× the runtime (15 s vs 3.7 s for a full scenario). It is not the
default because it changes every downstream number and the 1T path is what
every existing comparison in `README.md` was measured against — but it is the
better physics, and it is what to use for quantitative work.

One downstream consequence worth knowing before switching: the narrowband
field at 916 MHz and 0.30 m moves from 1.7×10⁻⁴ V/m (1T) to 5.5×10⁻³ V/m
(2T), against 1.9×10⁻³ V/m measured. The 1T value is 11× low; the 2T value is
2.9× high. Both remain inside the framework's stated two-order-of-magnitude
closure bracket, so this is a genuine improvement but not a validation.

```bash
python -m hvi_emp validate      # prints the open issue and its full reasoning
```

---

## Remaining warning

`IonisationTable` differs from a direct `saha_solve` by 7.2% at n = 10²⁰ m⁻³,
T = 0.3 eV — the extreme low-density, low-temperature corner of the grid, where
Z̄ ≈ 0.007. The absolute error is 5×10⁻⁴ in Z̄ and it has no influence on any
result, but it is reported rather than suppressed.

---

## What the audit checks

| group | examples |
|---|---|
| EOS thermodynamics | P_c = −dE_c/dV to 10⁻³ over 0.55–2.5 V₀ |
| Rankine–Hugoniot | P = ρ₀U_s u_p, ρ = ρ₀U_s/(U_s−u_p), E = u_p²/2 |
| Release | waste heat monotone in u_p, bounded by the shock energy, non-negative |
| Impedance matching | u_p,proj + u_p,tgt = v_impact; pressure continuity |
| Saha | Σf_j = 1, Σ j n_j = n_e, Z̄ = n_e/n_h, populations ≥ 0 |
| Ionisation limits | Z̄ monotone in T, → 0 at 300 K |
| Formulary | ω_pe and λ_D against NRL to 0.2% and 0.5% |
| Impact budgets | f_vap ∈ [0,1], m_vap ≤ m_proj, m_plasma ≤ m_vap, Q = Q_super + Q_two |
| Monotonicity | f_vap, m_plasma, T_e, Q all non-decreasing in velocity |
| Expansion | R increasing, density falling, sub-luminal, T > 0 |
| EMP | B = E/c in the far field, 1/r scaling, sin θ dipole pattern |

---

## Two more defects: the plume was a sphere, and half of it was underground

Both were found by looking at a ParaView render rather than at a number, and
both had survived every test and every audit check because nothing had ever
asked about the plume's *shape*.

### 6. `aspect0 = 1.0` is a fixed point, so the plume could only ever be a sphere

With `R_r = R_z` the two momentum equations `dv/dt = P/(rho R)` are
identical, so the ratio is conserved exactly: `R_z/R_r` stays 1.000 for the
whole run, to nine digits. The plume was not *found* to be spherical; it was
spherical by construction, and no amount of running it longer would change
that.

That matters because in the Anisimov model the asymptotic angular
distribution is

    N(theta) ~ [k^2 sin^2(theta) + cos^2(theta)]^(-3/2),   k = Rdot_z/Rdot_r

and `k = 1` makes `N` **flat** -- an isotropic plume. Laboratory measurements
of dust-impact plasma plumes report a **cosine law**, which is forward-peaked.
So the old default was not merely unlikely; it was excluded by experiment.

Least-squares fitting that cosine out to 80 degrees gives `k = 1.33`, which
this model reaches from `aspect0 = 0.5`. That is the new default.

**This is a calibration, and it is labelled as one.** One parameter fitted to
one measured distribution. It is not derived from the shock solution -- and
in fact this framework's own Stage 1 disagrees with it, because
`peak_pressure_profile` depends on distance alone, so every iso-pressure
surface including the vaporisation boundary is a hemisphere with aspect 1.
Real plumes are forward-peaked because free-surface rarefaction vents
material along the normal, and that physics is simply absent here. `aspect0`
is where it gets encoded, and `scenario.aspect0` exposes it for sweeping.

### 7. The vapour mass was spread over a sphere when the cloud is a hemisphere

Stage 1 computes `r_plume0 = (3V/2pi)^(1/3)` -- explicitly the radius of a
*hemisphere*, because vapour is released from a crater into the half-space
above a solid target. Stage 2 then spread that same mass over a **full** 3-D
Gaussian. Measured consequence: **~50% of the plume sat at z < 0, inside the
target**, for the first nanosecond, and the density was low by a factor of
two everywhere and at all times.

The cloud stays a half-Gaussian for the whole run, not only while it touches
the surface: a radially expanding cloud has no velocity component through the
plane containing its centre, so no material ever crosses into the lower half.
Equivalently, the real half evolves exactly as half of a freely expanding
sphere carrying twice the mass. Hence `HEMISPHERE_FACTOR = 2.0`, with no free
parameter.

### What the two changed

| quantity | before | after | reference |
|---|---|---|---|
| R_z/R_r, asymptotic | 1.000 (sphere) | **1.37** | 1.33 from the cosine law |
| peak n_e | 1.14e26 | **4.36e26** | ~1e27 (hydrocode) |
| frozen Q, 50 km/s | 1.03e-7 C | 2.43e-8 C | — |
| charge exponent beta | 6.383 | **6.393** | 3.48 -- still OPEN |
| validation checks passing | 30/31 | **30/31** | — |

The important line is the last two. **beta moves by 0.16%**, so these changes
do not quietly improve the framework's headline open discrepancy -- they are
not tuning dressed up as physics. And nothing that was passing now fails.

Peak n_e improves from 8.8x below the hydrocode value to 2.3x below it. That
remaining factor is **not** claimed to be resolved.

---

## The "IonisationTable within 5% of saha_solve" warning was misdiagnosed

That warning stood for a long time, reported as a 7.2% table-interpolation
error at the low-density corner, and dismissed in this document as having
"no influence on any result". Chasing it properly found something else.

**The old check measured the wrong thing, twice.**

It sampled an arbitrary 10×12 grid of points, which mostly miss cell
centres — exactly where bilinear interpolation is worst. Sampling at cell
centres, the true worst relative error is not 7.2% but **6×10⁵%**.

And relative error is meaningless where Z̄ → 0. That 6×10⁵% is Z̄ = 3×10⁻⁵
against 5×10⁻²: an absolute error of 0.05 in a regime with essentially no
free electrons, which propagates to nothing. What matters is the **absolute**
error, because n_e = Z̄ n_h.

**The residual is not an interpolation error at all.** `saha_solve` is
discontinuous where continuum lowering abruptly unbinds the ground state:

| material | T | n | Z̄ jumps |
|---|---|---|---|
| W | 0.27 eV | 3.9×10²⁸ | 0 → 0.150 |
| Fe | 0.27 eV | 4.2×10²⁸ | 0 → 0.155 |
| Al | 0.27 eV | 2.4×10²⁸ | 0 → 0.073 |

No interpolant represents a step, and the measurements confirm it: refining
the density grid makes the worst-case relative error **worse**, not better
(78% → 194% for W going from 64 to 320 points), because a finer grid simply
samples closer to the discontinuity. `n_pts` stays at 64 for that reason.

**A second, milder step sits inside the operating region.** At n ≈ 3×10²⁷,
T ≈ 0.73 eV the true Z̄ rises from 0.043 to 0.074 within a single grid cell —
`d ln Z̄ / d ln n = +6.8`, the *opposite* sign to ordinary Saha, so it is
pressure ionisation switching on. A W→Al plume at 30 km/s passes through it.
Worst absolute error there is 1.9×10⁻² in Z̄, and again refinement does not
help: 1.9×10⁻² at n_pts=64, 2.5×10⁻² at 128, 1.6×10⁻² at 160 — non-monotone,
the signature of a step rather than of under-sampling.

### What changed

The audit now:

* samples **cell centres**, not arbitrary points;
* gates on **absolute** error in Z̄ along real plume trajectories, at 0.01 —
  the region every published number depends on;
* **reports** the whole-grid worst case as a `note`, which is printed and
  recorded but never passes or fails, because no threshold on a
  discontinuous function is anything but arbitrary;
* locates the pressure-ionisation step explicitly, so it appears as itself
  rather than as a mysterious interpolation warning.

The one remaining warning is W→Al at 30 km/s, and it is **not** silenced by
widening the tolerance. It is a real defect in the ionisation model, and the
fix is a continuous continuum-lowering treatment — the same work as the
pending pressure-ionisation task that the Fletcher threshold gap also points
at.

For the default Fe→Al case the error along the trajectory is **0.12%**, so
nothing in the headline validation table is affected.

### A route to both open items

Both outstanding physics tasks — the EOS ceiling above ~20 km/s and the
pressure-ionisation step — now have a way to be *measured* rather than only
argued about. M2C (`hvi_emp.solvers.m2c_stage1`) ships:

* **Tillotson and ANEOS-Birch-Murnaghan-Debye** equations of state, i.e. the
  tabular EOS the linear Us–Up fit needs above its validated range; and
* a **multi-species non-ideal Saha solver** with Ebeling and Griem continuum
  lowering, solved *coupled to the flow* rather than tabulated.

The bridge sets `MaxChargeNumber` from the number of stages in each
material's YAML, so M2C solves the same ionisation ladder this framework
does and the two Z̄ values are directly comparable. Running W→Al at 30 km/s
through M2C with `depression_model="Ebeling"` and then `"Griem"` answers, for
that one case, whether the step is physical or an artefact of the tabulated
treatment here.

That is a measurement, not a fix: it tells you what the right answer is at a
point, and does not by itself give `ionization.py` a continuous closure.
Setting it up is `docs/SOLVERS.md` §5.
