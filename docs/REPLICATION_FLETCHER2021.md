# Replicating Fletcher (2021), NRL/MR/6757‑20‑10,138

*"Characterizing Electromagnetic Pulses from Hypervelocity Impact Plasmas"*

A practical plan for reproducing that report on a workstation, given that its
code is unobtainable. Written for a 128‑core Xeon with 512 GB and an NVIDIA
GPU; sizing scales obviously from there.

**Start here:**

```bash
python examples/07_fletcher2021_replication.py --cores 128
```

That generates every solver input described below and immediately tests the
claims the reduced chain can answer. Nothing needs to be installed first.

---

## 1. The obstacle, stated plainly

Fletcher used **ALEGRA** (Sandia National Laboratories) with **SESAME** EOS
tables and the **LMD conductivity model** for ionisation. None of the three is
publicly obtainable:

| component | status |
|---|---|
| ALEGRA | Sandia-controlled; not distributed outside authorised users |
| SESAME tables | LANL-controlled; licence required |
| LMD conductivity model | published method, no open implementation |

So a *bit-for-bit* replication is impossible. What follows is a
physics-equivalent replication with open or free-to-academics tools, and an
explicit account of what that costs you.

---

## 2. What is in the report, and what can reproduce it

**Verified from the report (NRL/MR/6757--20-10,138, AD1123389):** ALEGRA,
**SESAME** EOS tables, **Johnson–Cook** constitutive and fracture models,
**LMD conductivity model with pressure ionisation**, no resolidification.
Materials are a **tungsten projectile on an aluminium target** (chosen to
mimic experiments where W raises the charge yield through its density and low
work function). Six normal impacts at **12, 22, 32, 42, 52, 62 km/s**,
snapshots at **12 µs**. Figs. 6–7 add resistive MHD at a 30° angle, 30 km/s.


| # | Result in the report | Reproducible with | Fidelity |
|---|---|---|---|
| Fig. 4 | Mass density at 12 µs, W→Al, 12–62 km/s | **M2C** | good — this is what a multi-material hydrocode is for |
| Fig. 5 | Pressure at 2.5 µs, same six speeds | **M2C** | good |
| §2 | Complete-vaporisation threshold, 10–20 km/s | M2C + this framework | **disputed — see §5** |
| §2 | Plasma T, n, expansion velocity | M2C's coupled Saha solver, or this framework's | moderate → good |
| Fig. 6 | 30° oblique Al→Al at 30 km/s, **B ∥ surface**, field distortion | **M2C** (3-D hydro) + **OpenMHD** (field) | partial — see §4 |
| Fig. 7 | Normal W→Al, **B ⊥ surface**, diamagnetic cavity | **OpenMHD** | good for the cavity, not the crater |
| §2 | Ionisation state / pressure ionisation (LMD) | M2C's non-ideal Saha with continuum lowering | **was the weakest link — see §3.1** |
| §3 | CubeSat / ISS experiment concept, MASTER flux | not a simulation; nothing to replicate | — |

---

## 3. Tool by tool

### 3.1 M2C — the ALEGRA substitute (Figs. 4–6)

**This plan previously routed Figs. 4–5 through iSALE. It no longer does.**
Repeated access requests to the iSALE developers went unanswered, and waiting
on a queue that is not moving is not a plan. M2C (Zhao, Ma, Islam, Narkhede &
Wang, *Comput. Phys. Commun.* 2026;
[github.com/kevinwgy/m2c](https://github.com/kevinwgy/m2c)) replaces it, and
the substitution is an upgrade rather than a compromise on three counts:

| | iSALE | M2C |
|---|---|---|
| access | apply, wait, non-commercial licence | `git clone`, GPLv3 |
| oblique impact (Fig. 6) | iSALE2D is axisymmetric; needs iSALE-3D | 3-D Cartesian natively |
| ionisation | none — pure hydrodynamics | multi-species non-ideal Saha, coupled to the flow |
| tungsten EOS | no standard table (the old blocker) | Tillotson and ANEOS-Birch-Murnaghan-Debye built in |

The third row matters most. ALEGRA's distinguishing feature in this report is
the **LMD conductivity model with pressure ionisation**, and the old plan
listed the ionisation state as its "weakest link" precisely because iSALE
could not touch it. M2C solves ionisation *inside* the hydrodynamic step with
Ebeling or Griem continuum lowering, which is the same class of physics. That
converts §5's threshold disagreement from an argument into a measurement.

The second row removes the other structural blocker. Fig. 6 is a **30°
oblique** impact, and neither the reduced chain nor iSALE2D can represent it
at all (see `THEORY.md`, "Oblique incidence"). M2C switches to a 3-D Cartesian
mesh automatically when `angle_deg > 0`.

**Get it:**

```bash
git clone https://github.com/kevinwgy/m2c.git ~/src/m2c
cd ~/src/m2c && mkdir -p build && cd build && cmake .. && make -j
export M2C_HOME=$PWD
python -m hvi_emp.solvers.m2c_stage1 --check ~/src/m2c   # verify the grammar
```

It needs PETSc and MPI. Full instructions in
[`INSTALLING_SOLVERS.md`](INSTALLING_SOLVERS.md) §6b.

**Generated for you:** `fletcher2021/m2c/v{12,22,...,62}kms/`, each with a
complete `input.st` in M2C's mm-g-s-K-A units and a `README.txt` carrying its
own run command and cost estimate.

**Sizing.** M2C is MPI-parallel within a single run, which inverts the old
plan's bottleneck: iSALE2D was serial, so 128 cores bought concurrency across
the six cases but no speed within one. With M2C the six cases run
*sequentially*, each using the whole machine.

Per-case estimates come from `M2CConfig.estimate_resources()` and are
**order-of-magnitude only** — derived from cell counts and an explicit CFL
step, not measured. At 64 cores the example prints roughly 10 h for the
12 km/s case rising to 40 h at 62 km/s, because the CFL step shrinks as the
impact gets faster while the end time stays fixed. That is a *longer* campaign
than the old iSALE plan's ~12 h wall clock, and the comparison is not
like-for-like: iSALE2D ran six serial cases concurrently and computed no
ionisation, whereas M2C solves the Saha system in every cell every step. If
wall clock matters more than the plasma state, drop `cells_per_radius` — cost
goes as its square in 2-D.

The 3-D oblique case for Fig. 6 is a different scale of job: at the bridge's
3-D defaults it is ~12M cells and ~2 GB; at the full 2-D resolution ~461M
cells and ~86 GB, which a 540 GB workstation holds. The bridge picks
dimension-appropriate defaults and states the cost rather than silently
generating the larger one.

Memory is not the binding constraint — **time is**. With non-ideal Saha the
measured cost is ~1187 us per cell-step per core, so 461M cells is far beyond
a workstation regardless of RAM. Size 3-D runs by wall time, not by GB.

**What is lost.** iSALE has Lagrangian tracers and a validated crater-scaling
heritage (Pierazzo et al. 2008); M2C's published validation is on shock-tube
and fluid-structure problems plus the hypervelocity-impact case in its own
paper. If you specifically need a benchmarked *crater volume*, iSALE remains
the better instrument and the bridge for it is still in the package. For
Fletcher's figures — density and pressure fields, and the plasma state — M2C
is at least as good and gets you the ionisation for free.

### 3.2 OpenMHD — the diamagnetic cavity (Figs. 6–7)

The reduced chain has no magnetic field at all (THEORY §7.2, gap #4), and M2C
is pure hydrodynamics plus ionisation. OpenMHD covers the field physics.

**Generated:** `fletcher2021/openmhd/fig6_parallel/` and
`fig7_perpendicular/`, each with `model.f90`, `RUN.md` (build + SI conversion)
and `CASE.md` (what to look for).

```bash
git clone https://github.com/zenitani/OpenMHD    # Fortran 90 + MPI
```

This one *does* use your core count properly — OpenMHD is MPI-parallel.

**The honest gap:** ALEGRA-MHD solves the crater *and* the field in one code.
The substitute splits them: M2C gives the plume, OpenMHD expands that plume
into a magnetised ambient. You therefore lose the field's back-reaction on
crater formation. For the cavity itself that is a small error (the field is
dynamically irrelevant at crater pressures — β ≫ 1 there); for anything
involving currents *in* the crater it is not.

Each case ships with the analytic pressure-balance prediction
(cavity radius, formation time, characteristic frequency) so the run has
something to be tested against rather than just admired.

### 3.3 WarpX — the PIC half of the waterfall

Fletcher's whole framing is "hydrocode → PIC". `fletcher2021/warpx/` contains
the PICMI script and native input for the EMP stage, initialised from the
computed plume state. **This is where your GPU earns its keep** — WarpX has
first-class CUDA support and the 2D case becomes minutes.

See [`SOLVERS.md`](SOLVERS.md) §3 for the cost/epoch trade-off, which is the
same scale-separation wall Fletcher & Close (2017) hit.

### 3.4 LAMMPS — an independent check at nm scale

Not part of his report, but useful: the MD bridge resolves the
shock/vaporisation physics from first principles at nm scale, giving an
independent handle on the vaporisation threshold that neither a hydrocode nor
the reduced model provides. `pip install lammps mpich`, then
`examples/05_lammps_impact.py`.

---

## 4. Suggested order of work

Nothing here waits on anyone's approval any more, which is the main practical
change from the previous version of this plan.

1. **Today.** Run `examples/07_fletcher2021_replication.py`. You get the
   velocity-series predictions, all solver inputs, and cost estimates. Then
   `git clone` M2C and build it — half an hour, no application.
2. **First real result.** Run the **12 and 22 km/s** M2C cases. They are the
   cheapest and they discriminate on §5's threshold disagreement, which is
   the one question in this report worth settling.
3. **The rest of the series.** 32–62 km/s for Figs. 4–5, in **Al→Al** first:
   M2C has tungsten EOS options that iSALE did not, but Al→Al removes one
   variable while you are still validating the build.
4. **Fig. 6.** The 30° oblique case, now reachable — 3-D Cartesian, the one
   thing neither the reduced chain nor iSALE2D could do.
5. **The field physics.** Build OpenMHD (an afternoon) for Figs. 6–7; build
   or install WarpX and run the PIC case on the GPU.
6. **Post-process.** M2C writes `MeanCharge` and `ElectronDensity` directly,
   so the ionisation state ALEGRA got from LMD comes straight out of the run
   rather than being reconstructed; feed it into Stages 2–4 for the EMP.

---

## 5. A disagreement worth chasing — now quantified

Fletcher, p.5, for a **tungsten projectile on an aluminium target**:

> "We found a threshold velocity that lies between 10 km/s and 20 km/s at
> which the projectile is entirely vaporized (consistent with Zel'dovich and
> Raizer [30]) and the plasma transitions from partially ionized to fully
> ionized."

This framework puts complete vaporisation of tungsten at **36 km/s** — about
twice as fast. `python examples/11_fletcher_threshold.py` computes the whole
comparison; the resolution is a single number.

### The two criteria

| criterion | question it answers | crosses at |
|---|---|---|
| **Shock energy** (Zel'dovich & Raizer, cited by Fletcher) | did the shock *deposit* enough energy to vaporise? `E_shock = u_p²/2 ≥ E_vap` | **14 km/s** |
| **Release waste heat** (this framework, Ahrens & O'Keefe) | does enough energy *survive decompression* to leave it as vapour? | **36 km/s** |

**Only ~12% of the shock's internal energy is retained after isentropic
release**; the rest is returned as expansion work. That factor is the entire
disagreement. 14 km/s sits squarely inside Fletcher's 10–20 km/s band.

### Which is right

The waste-heat criterion is the standard one in shock physics, and this
framework reproduces Ahrens & O'Keefe's aluminium thresholds with it to within
2% with no free parameters (374 vs 380 GPa incipient, 1504 vs 1500 GPa
complete). By the classical criterion, ~36 km/s for tungsten is right.

But Fletcher is not using the classical criterion alone. ALEGRA runs SESAME
tables with the **LMD conductivity model, which includes pressure ionisation**,
and the report says plainly that "pressure ionization significantly increases
the ionization state in early phases of impact". Pressure ionisation converts
energy into ionisation *while the material is still compressed* — energy that
is then not recoverable as expansion work. That changes the release path in a
way Saha-on-release cannot capture, and it is listed as a known gap in
[`THEORY.md`](THEORY.md) §7.

So the two numbers bracket the answer, the mechanism that would close the gap
is named and physical, and — usefully — it is testable.

### A concrete prediction to test first

The old version of this section proposed running iSALE, which has no
pressure-ionisation model, and inferring the cause from where it landed. M2C
lets you do something better: **switch the mechanism on and off directly.**

Run the series three times:

| run | setting | if the threshold is… |
|---|---|---|
| A | `ionisation="ideal"` | near **36 km/s** — matches this framework, so the gap is not continuum lowering |
| B | `ionisation="non-ideal", depression_model="Ebeling"` | near **10–20 km/s** — continuum lowering is the mechanism, and ALEGRA's LMD model is doing the same job |
| C | `depression_model="Griem"` | a different answer from B — the *choice* of lowering model matters as much as its presence, which is itself worth reporting |

If A and B agree with each other but both sit near 36 km/s, the cause is
definitional rather than physical: Fletcher judges "entirely vaporized" from
density snapshots 12 µs after impact, and hot dense fluid above the critical
point can look like vapour without having crossed the complete-vaporisation
energy.

Every outcome is a real result, and unlike the iSALE version this one does not
depend on inferring a mechanism from a single number. **Run the 12 and 22 km/s
cases first** — cheapest, and they discriminate.

### Our numbers at his six speeds (W → Al)

| v [km/s] | f_vap (projectile) | m_plasma/m_p | T_e [eV] | Z̄ | v_exp [km/s] |
|---|---|---|---|---|---|
| 12 | 0.000 | 0.000 | 0.03 | 0.00 | 6 |
| 22 | 0.208 | 0.000 | 0.03 | 0.00 | 11 |
| 32 | 0.752 | 0.134 | 0.86 | 0.08 | 16 |
| 42 | 1.000 | 1.285 | 1.28 | 0.30 | 21 |
| 52 | 1.000 | 1.525 | 1.75 | 0.48 | 26 |
| 62 | 1.000 | 1.852 | 1.87 | 0.49 | 31 |

The report says "for 22 km/s and above, there is a significant amount of gas
and plasma generated that expands outwards at speeds of several km/s". Our
plasma appears at 32 km/s — one velocity step later, exactly the offset the
threshold comparison predicts.

## 6. What cannot be replicated at all

Be clear about this before committing time:

* **Bit-for-bit agreement with ALEGRA.** Different code, different EOS,
  different constitutive models. Expect agreement in trends and
  order-of-magnitude, not in digits.
* **The LMD conductivity model.** Its pressure ionisation is more aggressive
  than equilibrium Saha with Debye–Hückel continuum lowering, which is why
  this framework under-predicts Zbar generally (THEORY §7.2, gap #2). Nothing
  open reproduces it.
* **Coupled MHD + crater formation in one code** (§3.2).
* **Figures 8–10** — the MASTER debris-flux and CubeSat-design sections are
  not simulations.
* **His absolute EMP amplitudes**, which he does not publish in that report
  anyway; the amplitude comparison lives in Fletcher & Close (2017) and
  Close et al. (2013), and is handled by the closure bracket in THEORY §6.

---

## 7. Hardware notes for a 128-core / 512 GB / GPU workstation

| task | how it uses the machine | wall clock |
|---|---|---|
| reduced-chain velocity series | 1 core, seconds | instant |
| M2C 2D series (6 speeds) | MPI, all cores per case, run sequentially | ~10–40 h/case, rising with speed |
| M2C 3D oblique (Fig. 6) | MPI, ~64 cores, ~12 GB | hours–days |
| OpenMHD 2D cavity | MPI, ~16–32 cores | hours |
| WarpX 2D PIC | **GPU** | minutes–hours |
| LAMMPS nm-scale check | MPI, 16–64 cores | minutes |

Core count is the constraint on the 2-D work; memory becomes one only for the
3-D oblique case. The GPU is idle except for WarpX — if you want to use
it harder, WarpX 3D or a Debye-resolved 2D run is the place.

---

## 8. References

- Fletcher, A.C., *Characterizing Electromagnetic Pulses from Hypervelocity Impact Plasmas*, NRL/MR/6757‑20‑10,138 (2021). **[the target]**
- Fletcher & Close, *Phys. Plasmas* **24**, 053102 (2017) — the PIC half.
- Fletcher, Close & Mathias, *Phys. Plasmas* **22**, 093504 (2015) — the ionisation threshold he cites.
- Zhao, Ma, Islam, Narkhede & Wang, *Comput. Phys. Commun.* (2026) — M2C, the hydrocode this plan now uses.
- Islam, Zhao, Wang et al. (2023) — plasma formation in ambient fluid from hypervelocity impacts; the chamber-gas ionisation M2C exposes and this framework omits.
- Collins, Melosh & Ivanov, *Meteorit. Planet. Sci.* **39**, 217 (2004) — iSALE, retained as an optional bridge.
- Wünnemann, Collins & Melosh, *Icarus* **180**, 514 (2006) — iSALE porosity.
- Pierazzo et al., *Meteorit. Planet. Sci.* **43**, 1917 (2008) — the aluminium crater benchmark, if you do end up using iSALE.
- Zenitani, OpenMHD, <https://github.com/zenitani/OpenMHD>.
- Fedeli et al., SC22 — WarpX.
- Zel'dovich & Raizer (2002) — the vaporisation criterion he invokes.
