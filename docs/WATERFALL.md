# The hydrocode → PIC waterfall

This is the strategy of Fletcher (2021), NRL/MR/6757--20-10,138, §2:

> we break simulations of hypervelocity impact plasmas into two parts with
> the first providing initial and boundary conditions for the second in a
> waterfall fashion. A hydrocode models the impact, solid mechanics, phase
> change, high energy density (HED) physics, and plasma generation, while a
> particle-in-cell (PIC) code takes that result into the collisionless regime
> to model the EMP.

Until this change the package did not do that. M2C ran, and wrote
`electron_density` and `mean_charge_number` every output interval, but no
code read them: every PIC and MHD bridge was initialised from the reduced
analytic chain (`from_expansion(ExpansionResult)`). The hydrocode and the PIC
were never connected.

## Three commands

```bash
# 0. once: stock M2C aborts on Tillotson + TemperatureDependsOnDensity = Yes
python scripts/patch_m2c_tillotson.py $M2C_HOME && make -C $M2C_HOME -j 16

# 1. hydrocode (M2C), Tillotson EOS where constants are verified
python scripts/write_m2c_deck.py                    # -> runs/m2c/al_al_v32kms/input.st
cd runs/m2c/al_al_v32kms
mpirun -np 64 $M2C_HOME/m2c input.st > m2c.log 2>&1 &   # writes results/solution.pvd
python ../../../scripts/watch_m2c.py m2c.log --follow --abort --pid $!

# 2. handoff at the collisional -> collisionless transition
cd ../../..                                         # back to the repo root
python scripts/m2c_handoff.py runs/m2c/al_al_v32kms/results

# 3. PIC initialised from the hydrocode's own plume
python scripts/warpx_from_handoff.py runs/m2c/al_al_v32kms/handoff/plume --ion Al
cd runs/m2c/al_al_v32kms/warpx && mpirun -np <N> python warpx_from_handoff.py
```

## Stage 1 — the hydrocode must survive the expanded states

The deck used `ExtendedMieGruneisen` for every solid: a compression-branch
fit with no vapour physics. The 50 km/s run that died after 12.4 h had
Riemann states at 5 % and 40 % of solid density and an implied sound speed
over 1000 c — the EOS evaluated where it has no physics. Fletcher used
SESAME tables, which cover vapour; the open alternative M2C ships is
**Tillotson**, now the default (`M2CConfig(eos="auto")`) for any material
with a verified constant set.

Verified so far: **Al**, Tillotson (1962) as tabulated by Melosh (1989),
Table AII.3. Checked against an independent data set: the Tillotson sound
speed at rest, √(A/ρ₀) = 5277 m/s, against the library's Mie–Grüneisen
c₀ = 5386 m/s (2 %).

**Tungsten is not verified.** No Tillotson set for W could be traced to a
primary source. `eos="auto"` keeps W on Mie–Grüneisen and says so in the
deck notes; `eos="tillotson"` refuses. Because Fletcher's W projectile
vaporises fully above 10–20 km/s, an unsourced W EOS would decide the
answer, so supply one with a citation:

```python
from hvi_emp.solvers.m2c_stage1 import TillotsonParams, register_tillotson
register_tillotson("W", TillotsonParams(rho0=..., E0=..., a=..., b=..., A=...,
                   B=..., alpha=..., beta=..., E_iv=..., E_cv=...,
                   source="<reference>"))
```

Until then, run the series **Al → Al** (the replication plan already
recommended this order).

### Three silent traps in M2C's Tillotson, found in its source

All three run without error and give the wrong answer.

1. **Omitted keys default to water** (Brundage 2013, Table 1). The deck now
   writes all 14 `TillotsonModel` keys, and a test enforces it.
2. **Default temperature law gives negative temperatures.** With
   `TemperatureDependsOnDensity = No`, M2C computes T = T₀ + (e − E₀)/c_v,
   where E₀ is Tillotson's energy *scale* (5 MJ/kg for Al). Material at rest
   (e = 0) sits at **−5274 K**, and that feeds the Saha solver. The deck uses
   `Yes`: T = T₀ + (e − e_cold(ρ))/c_v (Brundage 2013).
3. **ρ_IV can switch off the plume's pressure.** M2C's incipient-vaporisation
   density is not a tabulated Tillotson constant. Below it, every state with
   e < E_cv uses the cold-expanded formula — including partially vaporised
   material — which goes tens of GPa into tension and is clamped by
   `PressureCutOff` to 1 Pa. (The `p = 1` states in the failed run's Riemann
   log are this clamp.) The default is now the lowest ρ_IV M2C allows without
   the cold curve turning over, ρ_IV = 1.05 ρ₀(1 − A/2B) = 0.443 ρ₀ for Al:
   the closest M2C gets to the form the constants were fitted for. At 0.6 ρ₀
   and 12 MJ/kg this gives +7.3 GPa of vapour pressure instead of 1 Pa.

## Stage 2 — the handoff

`hvi_emp.handoff` implements the join and nothing the report does not
support.

**Which cells are plume.** Target or projectile material, in front of the
target surface, moving away from it, with free electrons — the plasma that
"separates from the target entirely". Ionised material moving back toward the
target is excluded and its charge is reported
(`charge_excluded_backflow_C`).

**When.** Per cell, the electron plasma frequency against the electron–ion
Coulomb collision frequency (NRL formulary), the two curves of Fletcher's
Fig. 2. The handoff frame is the first in which at least half the plume's
free electrons are collisionless (`--fraction`). The transition is then
treated as instantaneous, as Fletcher and Close et al. do; **the fraction
still collisional is recorded**, because it is the size of that assumption.
Electron–neutral collisions are computed as a diagnostic and flagged when
they dominate, but do not change the criterion.

**What.** The report's three quantities — total charge, temperature,
expansion velocity — plus the plume on the M2C mesh (n_e, n_heavy, Z̄, T,
velocity, cell volumes). Code-neutral `.npz` + `.json`, no new dependency.

### A result the tool reports rather than papers over

If the plume cools adiabatically, the handoff never happens. For monatomic
adiabatic expansion n ∝ s⁻³ and T ∝ s⁻², so ν_ei ∝ n T^(−3/2) is **constant**
while ω_pe ∝ s^(−3/2) falls: the plume gets *more* collisional as it expands.
Fletcher's Fig. 2 has collisions falling faster than the plasma frequency,
which needs the electron temperature to stay up — Close et al.'s
isothermal expansion. `m2c_handoff.py` detects a falling collisionless
fraction and says so. Whether a real M2C plume behaves like this is
something the first real run will tell us.

### The reader

`m2c_output.read_m2c_vtr` reads M2C's snapshots against the byte layout of
PETSc's `DMDAVTKWriteAll_VTR` (one `<Piece>` per MPI rank, UInt64 byte-count
prefixes, x-fastest Float64, points at cell centres), tolerates M2C writing
`electron_density` twice, and rebuilds cylindrical cell volumes from the
centres. Units come back SI through the same table the deck writer uses.

## Stage 3 — PIC from the handoff

`hvi_emp.solvers.warpx_handoff` writes a WarpX **RZ** run (M2C x → WarpX z,
M2C y → r; the target surface becomes a conducting wall).

* Every macroparticle descends from an M2C cell, sampled inside it.
* Electrons and ions are **co-located pairs**: the run starts exactly
  quasi-neutral.
* Both carry the cell's single-fluid velocity plus a Maxwellian. The old
  bridge imposed an electron drift of √(m_i/m_e)·v_ion by hand — the Fletcher
  & Close (2017) assumption the PIC exists to test. Here the separation must
  come out of the PIC.
* The charge in the particles equals the handed-over charge to round-off.

### Getting from the handoff to the measured bands

At handoff the plume is dense — for the synthetic mm-scale cloud, Debye
lengths of 0.2 nm and a peak density of 5×10²⁷ m⁻³. Nothing there
oscillates at 315 or 916 MHz: those need n_e = ε₀m_eω²/e² = 1.2×10¹⁵ and
1.0×10¹⁶ m⁻³. Starting EM PIC at the handoff would take ~35 million steps,
and the plan says so.

Fletcher & Close (2017) met the same gap and started their PIC where the
plume resonates with the band: "an impact plasma with a plasma frequency near
the RF emission frequency that was measured in experiments has a peak density
of 10¹⁶ m⁻³ and a length scale of 40 mm". In between, the plume is
collisionless and quasi-neutral, so it expands ballistically.
`--advance band` (default) does that to the hydrocode's own plume, keeping its
spatial and velocity structure where they imposed an analytic profile.
`--advance none` starts at the handoff.

The advance omits post-transition ambipolar acceleration. The run records
the residual electron thermal energy against the ion kinetic energy (1.03×
for the synthetic case) and the resulting bound on the velocity error
(up to 1.43×).

**Scale matters.** A mm projectile (Fletcher's series) carries coulombs of
free charge, so it reaches the 916 MHz density only at **metre scale, about a
millisecond** after impact. F&C's 40 mm scale corresponds to ~10⁻⁷ C, which is
dust-accelerator territory. Explicit PIC resolving λ_D at their epoch costs
~2×10¹² cell-steps (overnight on one GPU); `--scheme auto` picks implicit when
explicit exceeds `--budget`, resolving the edge plasma frequency, where the
radiating shell is, and reports the explicit figure.

## What has and has not been run

| | status |
|---|---|
| M2C Tillotson deck | written and grammar-checked against M2C source; first run aborted at start-up in stock M2C (docs/M2C_FAILURE_MODES.md §4) |
| M2C Tillotson patch | M2C's own `VarFcnTillot.h` compiled and driven from 1e-6 to 5 rho0: aborts unpatched, passes patched. **Full M2C not rebuilt here** |
| VTR reader | tested against PETSc's byte layout, multi-rank; **not yet on a real M2C file** |
| handoff extractor | tested on synthetic clouds with known integrals |
| WarpX particles and plan | tested |
| WarpX PICMI script | compiles; **not executed** (no pywarpx here). Boundary names and implicit-RZ support are the first things to check |

The synthetic M2C output (`m2c_synthetic`) is a prescribed Gaussian cloud in
M2C's file format. It tests plumbing, not physics, and `extract` refuses it
unless told otherwise.

## References

* Fletcher, A. C. (2021). *Characterizing Electromagnetic Pulses from
  Hypervelocity Impact Plasmas.* NRL/MR/6757--20-10,138.
* Fletcher, A. C. & Close, S. (2017). Particle-in-cell simulations of an RF
  emission mechanism associated with hypervelocity impact plasmas. *Phys.
  Plasmas* 24, 053102.
* Tillotson, J. H. (1962). *Metallic equations of state for hypervelocity
  impact.* General Atomic GA-3216.
* Melosh, H. J. (1989). *Impact Cratering: A Geologic Process.* Oxford UP.
* Brundage, A. L. (2013). Implementation of Tillotson equation of state for
  hypervelocity impact of metals, geologic materials, and liquids. *Procedia
  Eng.* 58, 461–470.
* NRL Plasma Formulary (2019).
* M2C source: `IoData.cpp`, `VarFcnTillot.h`, `Output.cpp`
  (github.com/kevinwgy/m2c). PETSc source: `src/dm/impls/da/grvtk.c`.
