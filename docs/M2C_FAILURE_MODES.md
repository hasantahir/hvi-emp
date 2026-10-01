# M2C failure modes seen in this project

## 1. dt collapse (the expensive one)

**Signature**

```
Step 129521: t = 8.414028e-10, dt = 1.885042e-22, cfl = 1.0000e-01.
             Computation time: 4.4740e+04 s.
```

The run keeps printing `Step N`, so it looks alive. It is not. `dt` has
fallen ~7 orders of magnitude below this run's own average step
(6.5e-15 s), and dt collapse does not recover.

**Why it is unambiguous**

`dt` is not a free parameter -- the solver picks it from the CFL condition,
`dt = CFL * dx / (|u| + c)`. Compare two points in the *same* run and the
cell size cancels:

```
(|u| + c)_now / (|u| + c)_early  =  dt_early / dt_now  =  3.7e7
```

Even granting the early state a generous 100 km/s, the final state implies
over 1000 c. An equation of state cannot return that. The state it is being
handed is not physical.

Quote an *absolute* speed only with the **finest** cell of the graded mesh.
Using the nominal spacing over-states it by the grading factor -- about 50x
here, in the alarming direction.

**Detection**

```
python scripts/watch_m2c.py run.log --t-end 9.22e-9
python scripts/watch_m2c.py run.log --follow --abort --pid $!   # in a job
```

## 2. Riemann solver failure

```
Warning: Riemann solver failed to find a bracketing interval or to
         converge on 487 edge(s).
Warning: Division-by-zero while using the secant method ...
         dir = 0.921473,-0.388441,0, f0 = 0, f1 = 0
```

`f0 == f1 == 0` means the pressure function is flat in the secant bracket:
the two states are not physically distinguishable to the EOS. Seen with
states like

```
left:  0.000124656, -3.11169e+06, 36.8047,  matid 2
right: 0.00107383,  -3.11169e+06, 109.298,  matid 1
```

In M2C's mm-g-s units those densities are 0.125 and 1.07 g/cm3 -- 5% and
40% of solid aluminium. That is **expanded, partly vaporised material**, and
it is exactly where a Mie-Grueneisen form fitted on the compression branch
stops being valid. One state carries `p = 1` exactly, which looks like a
floor being hit rather than a computed pressure.

## 3. Level sets overlapping (the fatal one, usually last)

```
*** Error: Node (519,18,0) belongs to two material subdomains.
           phi[0(matid:1)] = -5.600041e-03, phi[1(matid:2)] = -2.026143e-02.
```

Both level sets are negative at one node: both materials claim it. This
aborts the run, but it is normally a **consequence**. Check the log for
`L-S Reinitialization` failures and for dt collapse first -- if either
started thousands of steps earlier, the overlap is the symptom and not the
disease.

## 4. Signal 6 (abort) with a Tillotson deck

```
[reznor:...] Signal code:  (-6)
[ 2] /lib64/libc.so.6(abort+0x127)
[ 5] .../m2c(+0x359d4)
[ 6] .../m2c(+0x36a1c)
[ 7] .../m2c(+0x2c9b9)            <- main
prterun noticed that process rank 26 ... exited on signal 6 (Aborted).
```

Three M2C frames, the last one `main`: the abort is in set-up, not in the
time loop. It is `assert(!err)` in `VarFcnTillot.h`, and it happens on every
rank (the rank number is just whichever reported first) with any core count.

It is caused by `TemperatureDependsOnDensity = Yes`, which our deck needs.
Stock M2C has three bugs on that path, all reproduced by compiling its own
`VarFcnTillot.h`:

| when | what |
|---|---|
| start-up | the constructor builds the cold curve before setting `elat = eCV - eIV`, then divides by it in the Case 1\|2 blend (Al's cold curve passes e_IV above rho0/2) |
| shock past 2 rho0 | on-demand extension starts with the previous step; asking for less than one step more makes `runge_kutta_45` return -1 |
| plume below ~rho0/2 | marching backwards, `runge_kutta_45`'s first step is the whole interval; and below rho_IV the curve stalls on e = e_CV (Case 3 / Case 2 switch) |

Running M2C itself on the deck (built in the sandbox from the same source)
then found more, each the next thing to stop the run:

| t | symptom | cause |
|---|---|---|
| step 2 | `Exit 255`, often no message | p -> e has no solution below rho_IV for pressures between Case 3 at e_CV- and Case 2 at e_CV+ (-32 to +8.3 GPa for Al); `exit(-1)` on one rank |
| 28 ns | `Exit 255` | the exact Riemann solver evaluates the EOS at trial rho <= 0 before rejecting it; Tillotson exits where other EOSs return |
| 72 ns | signal 6 | the solver's `rho<=0 \|\| c^2<0` stage test lets NaN through |
| 81 ns | `Exit 255` | blend inverse with its root on e_IV to round-off |

`Exit 255` with nothing on screen: M2C's `exit_mpi()` is `exit(-1)`, and
`print_error` prints from rank 0 only, so a failure on another rank can be
silent. `grep -n "Error" m2c.log` first; then rerun the same deck on fewer
ranks.

The patch now covers all of these (VarFcnTillot.h and
ExactRiemannSolverBase.cpp). Fix and check:

```bash
python scripts/patch_m2c_tillotson.py $M2C_HOME --selftest   # FAIL (aborts) before
python scripts/patch_m2c_tillotson.py $M2C_HOME              # patch (keeps .orig)
python scripts/patch_m2c_tillotson.py $M2C_HOME --selftest   # PASS
make -C $M2C_HOME -j 16                                      # rebuild M2C
```

The patch holds e_cold at e_CV below the density where it reaches it (0.35
rho0 for Al): the cohesive plateau. Vapour then gets T = T0 + (e - e_CV)/cv.
Expanded material below e_CV gets T < 0, which the Saha solver treats as no
ionisation -- right for a cold two-phase mixture, and the same as stock M2C
would give if it did not abort.

## 5. dt collapse from the near-vacuum ambient

Two mechanisms, both from the 10^12 density contrast between metal and the
1e-4 Pa "vacuum", both measured on the Al->Al 32 km/s deck:

* **Pressure left behind.** When a metal level set leaves a cell, M2C gives
  the cell to the ambient gas, and it can keep the metal's pressure. One
  cell held 711 GPa at 2e-9 kg/m^3: c = 2.5e10 m/s, dt 4e-11 -> 1e-16 s at
  8.5 ns. Fix in the deck: `PressureUpperLimit` on the ambient, capping its
  sound speed at max(4 v_impact, 100 km/s) -- above anything the gas reaches
  physically.
* **Runaway at the density floor.** Metal cells clipped to DensityCutOff
  (1e-6 rho0) keep their pressure and accelerate to ~1000 km/s (76 ns in a
  coarse run). Fix: a denser ambient. In the same coarse run 1 Pa lasted to
  250 ns and 100 Pa was healthy at 450 ns (where the test stopped).
  `write_m2c_deck.py` now defaults to 100 Pa and prints the gas mass in front
  of the target as a fraction of the projectile (0.4 % for 1 mm Al), which is
  why it cannot slow the plume inside the mesh.

## Order to investigate

Cheapest first, and each one rules something out:

1. **Same deck at ~20 km/s.** Inside the EOS's calibrated range. If it
   survives, the EOS out at 50 km/s is the cause and everything above is
   downstream. This is the single most informative experiment.
2. **Same velocity, coarser mesh, short `t_end`.** Does the collapse arrive
   at the same simulated **time** or the same step **count**? Time points at
   physics (a state the EOS cannot represent); count points at numerics
   (accumulating interface error).
3. **Only then** tune level-set reinitialisation. Tuning it first can
   postpone the abort without fixing anything, which is worse than the
   abort: the run then produces plausible output from an unphysical state.

## Open issues this touches

- **#34** EOS validity ceiling above 20 km/s. The reference case runs at
  50 km/s.
- **#35** Pressure ionisation. Already surfaced in a real run as Ebeling
  depression dI = 42.9 eV against Al III's I = 28.4 eV -- a 1.51x overshoot.
