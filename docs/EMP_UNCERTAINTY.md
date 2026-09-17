# The EMP amplitude: where the uncertainty is, and what to do about it

```python
run_scenario(..., closure="calibrated")   # for absolute amplitudes
run_scenario(..., closure="debye")        # conservative, first-principles
run_scenario(..., closure="fletcher")     # aggressive, first-principles
```

## The uncertainty is one number, not many

The framework's EMP amplitude spans a factor of **184**. It is tempting to
read that as accumulated error from many approximations. It is not. It is
one quantity: the electron–ion charge separation `delta`.

Every other term in the radiated field is either measured, derived, or
cancels in the ratio. The two first-principles closures are the **same
physics** with different choices of the velocity that sets `delta`:

```
debye      delta = xi * lambda_D            = xi * v_te / omega_pe
fletcher   delta = xi * sqrt(m_i/m_e) * v_bulk / omega_pe
```

and they are *identical* when the ion velocity is taken to be the ion sound
speed, because `sqrt(m_i/m_e) c_s = v_te` exactly. Fletcher & Close apply the
mass-ratio factor to the **bulk expansion** speed instead. At 50 km/s that is
15× the sound speed, and that single modelling choice is the entire bracket.

| closure | E at 916 MHz, 0.30 m | vs measured |
|---|---|---|
| `debye` (ξ = 1) | 1.29×10⁻⁴ V/m | 15× **low** |
| `calibrated` (ξ = 15) | 1.93×10⁻³ V/m | **1.02×** |
| `fletcher` | 2.37×10⁻² V/m | 12.5× **high** |
| measured, Close 2013 | 1.90×10⁻³ V/m | — |

## What was tried and did not work

**An energy-conservation bound on the aggressive closure.** This looked
promising: the Fletcher drift implies 118 eV of directed energy per electron
against 1.5 eV of thermal energy, which sounds impossible.

It isn't. At Z̄ ≈ 0.01 there are ~100 ions per electron, so the ion bulk
reservoir holds ~11,800 eV per electron and 118 eV is about **1% of it**.
Implausible, but not forbidden. Energy conservation does not constrain
`delta`, and the bracket cannot be narrowed that way.

This is written down because a negative result that is not recorded gets
re-attempted.

## What calibration buys, and what it costs

The radiated field is **exactly linear in `delta`**, so a single measurement
determines ξ with no fitting freedom at all:

> **ξ = 15 reproduces Close et al. (2013) to 2%.**

`closure="calibrated"` uses it. What this changes:

**Gained.** Absolute amplitudes are usable. The 184× bracket becomes a point
prediction.

**Lost.** The framework no longer *predicts* the absolute amplitude at that
one point — it reproduces it by construction. Anyone quoting an absolute
number must say it rests on a calibration to Close 2013.

**Unaffected, and this is the important part.** Every *scaling* remains a
prediction, because none of them were used to fit ξ:

* with impact speed,
* with projectile mass and diameter,
* with projectile and target material,
* with observation frequency,
* with sensor distance and angle.

So a calibrated run that gets the *velocity dependence* right is still
telling you something. One that gets it wrong is still falsifying the model.

## It is a falsifiable claim, not a fudge factor

ξ = 15 says something specific and checkable: **the charge separation is
about 15 Debye lengths.** That is a large sheath — a few λ_D is typical —
but not absurd for a plasma expanding into vacuum with a non-Maxwellian
electron tail.

A particle-in-cell run measures `delta` directly. `solvers/picongpu_stage3`
already generates the input set, and the probe particles it places are there
to record exactly this. The check is:

```bash
python examples/09_gpu_solver_decks.py     # writes the PIConGPU input set
# build and run on the A5000, then read back the probe data
```

Three outcomes, all informative:

1. **PIC gives δ ≈ 15 λ_D.** The calibration is confirmed from first
   principles and the framework becomes predictive again — the strongest
   possible result.
2. **PIC gives a different δ.** Use it. The calibration was a placeholder and
   the PIC value replaces it, with the discrepancy worth understanding.
3. **PIC gives δ that depends on parameters the closure treats as fixed.**
   Then ξ is not a constant and the closure needs a functional form, which
   is a more interesting result than either.

Until that run happens, `CALIBRATED_XI` carries its provenance in
`CALIBRATION_SOURCE` and this document, and the validation suite reports the
closure in use with every amplitude.

## Which closure to use

| you want | use |
|---|---|
| an absolute amplitude to design against | `calibrated`, and cite the calibration |
| a conservative lower bound | `debye` |
| an upper bound for margin analysis | `fletcher` |
| to test the model against new data | `debye` or `fletcher` — using `calibrated` against Close 2013 is circular |

The default remains `debye`, so nothing silently changed underneath existing
results. Choosing `calibrated` is a deliberate act, which is appropriate for
a number that rests on one measurement.
