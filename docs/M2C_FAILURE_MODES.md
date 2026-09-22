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
