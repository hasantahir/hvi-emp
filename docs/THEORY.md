# HVI-EMP: theory, assumptions and validity

A physically-traceable reduced-order framework for hypervelocity impact plasma
generation and the resulting electromagnetic pulse.

This document derives every closure used in the code, states where each one
breaks, and records what has been validated against published data and what
has not. Section 7 is the honest-limitations section; read it before quoting
any absolute number from this framework.

---

## 1. The problem and why it is split into four stages

A piece of orbital debris striking a spacecraft at 10–72 km/s deposits enough
energy to vaporise and ionise both itself and part of the target. The plasma
expands into vacuum, goes collisionless, and — under the mechanism proposed by
Close *et al.* (2010) — radiates an electromagnetic pulse. That pulse, or the
plasma itself, is a candidate cause for a class of on-orbit anomalies
(Olympus-1, Landsat 5, ADEOS-II) in which mechanical damage was ruled out.

The process spans roughly

| quantity | range |
|---|---|
| length | 10⁻⁶ m (early plume) → 10⁻¹ m (sensor standoff) |
| time | 10⁻¹⁵ s (plasma period) → 10⁻³ s (crater formation) |
| density | 10²⁸ m⁻³ (near-solid) → 10¹⁵ m⁻³ (radiating shell) |
| physics | solid mechanics → HED → strongly coupled plasma → kinetic plasma → free-space EM |

No single code resolves all of it. Fletcher (NRL/MR/6757‑20‑10,138, 2021)
describes the standard approach as a "waterfall": a hydrocode for the impact,
handing initial conditions to a PIC code for the radiation. This framework
uses the same decomposition but replaces both expensive codes with validated
reduced-order models, so the whole chain runs in seconds and every closure is
inspectable.

```
Stage 1  impact.py      shock → release → vaporisation → ionisation inventory
Stage 2  expansion.py   self-similar plume → freeze-out → collisionless transition
Stage 3  emp.py         charge separation → coherent oscillation → radiation
Stage 4  coupling.py    surface charging → ESD → induced currents
```

Each stage's output is a small, physically meaningful state vector, so a stage
can be replaced by a higher-fidelity code (a real hydrocode for Stage 1, a PIC
for Stage 3) without touching the rest.

---

## 2. Stage 1 — impact, shock, release

### 2.1 Jump conditions and the linear shock relation

Across a steady shock into material at rest, with $P_0 \approx 0$:

$$\rho_0 U_s = \rho (U_s - u_p), \qquad P = \rho_0 U_s u_p, \qquad
E - E_0 = \tfrac{1}{2} P (V_0 - V) = \tfrac{1}{2} u_p^2$$

closed empirically by $U_s = c_0 + s\,u_p$. The energy identity
$E = u_p^2/2$ is exact and is asserted in the test suite.

**Validity.** The linear $U_s$–$u_p$ fit is calibrated to gas-gun and
explosive data, typically $u_p \lesssim 10$–15 km/s. Above that, electronic
excitation and shell ionisation stiffen the Hugoniot and the linear
extrapolation over-predicts pressure. `Material.us_up_valid_to` records the
limit per material and the code emits a `RuntimeWarning` past it;
`pipeline.run_scenario` promotes that to a scenario-level warning. For
quantitative work above ~30 km/s a tabular EOS (SESAME, ANEOS) should replace
`eos.hugoniot_state`.

### 2.2 Impedance matching (planar impact approximation)

Continuity of pressure and velocity at the projectile/target interface gives a
quadratic in the target particle velocity $u$:

$$(\rho_{0t}s_t - \rho_{0p}s_p)u^2
+ (\rho_{0t}c_{0t} + 2\rho_{0p}s_p v + \rho_{0p}c_{0p})u
- (\rho_{0p}s_p v^2 + \rho_{0p}c_{0p}v) = 0$$

**Validity.** One-dimensional. It gives the *peak* (isobaric-core) state
correctly and is the standard starting point (Melosh 1989 §4). It says nothing
about crater shape or the late-time flow field. It also says nothing about
**jetting**, which turns out to matter more than the omission suggests —
see §2.2b.

**Oblique impacts** use only the normal velocity component. This is accurate
to ~20% within 60° of normal; beyond that, downrange ricochet and
decapitation of the projectile matter and the framework warns.

### 2.2b Jetting — why plasma appears far below the bulk threshold

The impedance match above describes the *bulk*, and the bulk is not the
fastest thing in the problem. When two surfaces meet at a shallow angle, the
contact point runs outward at $v/\tan\alpha$ — far faster than either body
moves — and a thin sheet of material is squirted out ahead of it. For a
**sphere** the contact angle sweeps continuously from 0° to 90° during
penetration, so every impact passes through the jetting regime whether or not
anyone models it.

This is not a small correction. Jean & Rollins (*AIAA J.* **8**, 1742, 1970)
measured aluminium jets at **30–38 km/s from 4–6 km/s impacts**, and copper
jets "as high as 50 km/sec". Jetted material is shocked as though the impact
were ~6× faster than it is.

That resolves what would otherwise be a flat contradiction. Jean & Rollins saw
a hot, strongly ionised, continuum-radiating plasma at *every* velocity they
fired, from 2 km/s up. This framework puts the bulk complete-vaporisation
threshold for Al on Al at 37 km/s, and Close et al. (2013) report no RF below
15–20 km/s. All three are correct: the jet ionises where the bulk does not.

`hvi_emp.jetting` implements two results, both with no free parameters:

| quantity | model | validated against |
|---|---|---|
| critical angle for jet onset | $\tan\alpha_c = v/U_s$ (Walsh 1953) | J&R Table 1, mean error **1.9°** over 1–6.3 km/s |
| steady jet speed | $v_j = v\cot(\alpha_c/2)$ | J&R Table 2, ~13% over 10–40° |

plus one measured ratio, `FAST_JET_FACTOR = 1.73 ± 0.18`, relating the
transient detachment jet to the steady one (J&R Table 3).

**Why this does not overturn the RF threshold.** Mass. Jean & Rollins note
that their fast jet is *not luminous* because "too little material is
involved". A jet can make a spectroscopically obvious plasma and still carry
far too little charge to radiate a detectable pulse. **The module predicts jet
velocity, not jet mass**, and that is the honest limit of closed-form jetting
theory. Resolving the mass fraction needs a multi-material hydrocode that
tracks the jet as its own material — `solvers.m2c_stage1`.

**What it does license.** The jet arrives somewhere, and when it does it is a
hypervelocity impactor in its own right. `downrange_threshold()` asks what the
jet does to the *next* surface: a 5 km/s strike on a bumper delivers a ~33
km/s jet to whatever is behind it, and the framework's own EOS says that
completely vaporises an aluminium rear wall from an impact speed about **5.4×
lower** than the bulk would need. This is the mechanism behind ESA's Zero
Debris concern about impacts affecting *internal* items (Technical Booklet
§3.5), and it is a mechanical-to-electrical path the reduced chain would
otherwise miss entirely.

**Where it stops being trustworthy.** $v\cot(\alpha_c/2) \to 2U_s$ as
$v\to0$ rather than to zero, so below ~3 km/s the formula claims a 21 km/s jet
from a 1 km/s impact. That is not an artefact of our critical angle — feeding
J&R's own *measured* 12.5° at 1 km/s into the same formula still gives 9× — it
is inherent to evaluating a steady-flow result exactly where the jet is
beginning to form and carries almost no mass. The module warns rather than
returning the number silently.

### 2.3 The cold curve — and why a Hugoniot-referenced Mie-Grüneisen fails

Off-Hugoniot states use Mie-Grüneisen referenced to the **0 K cold curve**:

$$P(V,E) = P_c(V) + \frac{\Gamma(V)}{V}\big(E - E_c(V)\big), \qquad
\Gamma\rho = \Gamma_0\rho_0$$

The cold curve is the **Vinet** ("universal binding energy") form, calibrated
with no free parameters from the shock data:

$$K_0 = \rho_0 c_0^2, \qquad K_0' = 4s - 1, \qquad \eta = \tfrac{3}{2}(K_0'-1)$$
$$P_c(V) = 3K_0 x^{-2}(1-x)\,e^{\eta(1-x)}, \qquad x = (V/V_0)^{1/3}$$
$$E_c(V) = \frac{4K_0V_0}{(K_0'-1)^2}\Big[1 - \big(1-\eta(1-x)\big)e^{\eta(1-x)}\Big]$$

The textbook alternative — referencing Mie-Grüneisen to the **Hugoniot** — is
unusable here, and this was a real bug caught during development. For $V>V_0$
the Hugoniot pressure is identically zero, so $P = (\Gamma/V)E$, giving
$\mathrm{d}E/\mathrm{d}V = -(\Gamma_0/V_0)E$ and therefore *unbounded*
exponential cooling on release: the model predicted zero waste heat and zero
plasma at every impact speed. The Vinet cold curve has a genuine tensile
branch and a finite binding energy, so release terminates physically.

**A free check on the construction.** Vinet's asymptotic binding energy
$4K_0V_0/(K_0'-1)^2$ should equal the sublimation energy. It is derived
entirely from $\rho_0$, $c_0$, $s$ and compared with tabulated cohesive
energies:

| | Vinet [J/kg] | tabulated [J/kg] | ratio |
|---|---|---|---|
| Al | 1.03 × 10⁷ | 1.21 × 10⁷ | 0.85 |
| Cu | 3.97 × 10⁶ | 5.30 × 10⁶ | 0.75 |
| W | 7.47 × 10⁶ | 4.67 × 10⁶ | 1.60 |
| Fe | 3.35 × 10⁶ | 7.40 × 10⁶ | 0.45 |

Iron is the outlier because its tabulated $c_0 = 3955$ m/s comes from a fit
that folds in the α→ε phase transition, so $K_0 = \rho_0c_0^2 = 123$ GPa
understates the true bulk modulus (170 GPa). This is a known deficiency of the
single linear $U_s$–$u_p$ fit for iron and it propagates into iron's
vaporisation thresholds (Section 6).

### 2.4 Release and waste heat

Because $\Gamma/V = \Gamma_0/V_0$ is constant, the thermal energy
$\Delta E = E - E_c$ obeys, along an isentrope,

$$\frac{\mathrm{d}(\Delta E)}{\mathrm{d}V} = -P + P_c = -\frac{\Gamma_0}{V_0}\Delta E
\quad\Longrightarrow\quad
\Delta E(V) = \Delta E(V_H)\,e^{-\Gamma_0 (V - V_H)/V_0}$$

an exact closed form. Release terminates at

$$V_{\rm end} = \min\big(V_{P=0},\ V_{\rm spinodal}\big),
\qquad V_{\rm spinodal} = \arg\min_V P_c(V)$$

*Physically:* if the release adiabat can reach zero pressure while still on
the condensed branch, it does, and the leftover $\Delta E$ is the classical
waste heat. If the thermal pressure is too large to be cancelled anywhere on
the condensed branch, the material passes the mechanical stability limit,
fragments into vapour, and the remaining thermal energy is handed to Stage 2
rather than being converted to further $P\,\mathrm{d}V$ work inside the
target. Terminating at $V_0$ instead (the naive "release to ambient density")
was tried and under-predicts the vaporisation thresholds by 2–3× — see the
comparison in Section 6.

### 2.5 Phase state by the lever rule

$$E_{\rm melt} = c_v(T_m - 298) + L_f, \quad
E_{\rm boil} = E_{\rm melt} + c_v(T_v - T_m), \quad
E_{\rm vap} = E_{\rm boil} + L_v$$

with vapour fraction $(E_{\rm res} - E_{\rm boil})/L_v$ clipped to $[0,1]$.
Partial vaporisation is resolved rather than being a binary switch.

### 2.6 Shock decay and the plasma-forming mass

Outside the isobaric core the peak pressure decays as $P(r) = P_{ic}(r/r_{ic})^{-n}$
with $n \approx 1.5$–3 (Melosh 1989 §5.4; Pierazzo *et al.* 1997). Hemispherical
shells are integrated for melt and vapour mass.

**The distinction that matters.** Two very different populations come out of
the crater:

* **plasma-forming mass** — waste heat above $E_{\rm vap}$, so it is a
  *superheated* vapour with energy left over for ionisation;
* **condensable vapour/dust** — in the two-phase region, arriving at the
  boiling point with no superheat.

Mass-weighting these together (a single-temperature plume) is badly wrong:
shell mass grows as $r^2$ while waste heat falls as a steep power of $r$, so
the average is dominated by the *coldest* marginally-vaporised material. Doing
so during development gave plume temperatures of 0.03–0.7 eV where the
literature reports 2–2.5 eV — an order of magnitude too cold, and enough to
suppress the ionisation to nothing. Separating the two populations recovers
$T_e \approx 2$ eV at 50 km/s.

### 2.7 Mixed-composition plumes

The plume is projectile plus target vapour in a ratio that varies continuously
with impact speed. Selecting "whichever dominates by mass" introduces a
spurious *discontinuity* in $T_e$ and $\bar Z$ at the crossover, because the
mean atomic mass jumps by a factor of two. `materials.mix_materials` instead
builds a number-weighted synthetic material with

$$\frac{1}{\bar m} = \frac{x_a}{m_a} + \frac{x_b}{m_b}$$

and number-weighted ionisation potentials and degeneracies. **Limitation:** a
genuine two-species Saha solve would track each element's ladder separately
sharing one electron reservoir. The single effective ladder is good when the
two first ionisation potentials are within a couple of eV (true for most
metal-on-metal pairs) and degrades otherwise; the composition appears in the
material name so the approximation stays visible.

---

## 3. Ionisation

### 3.1 Saha ladder

$$\frac{n_{j+1}n_e}{n_j} = 2\frac{g_{j+1}}{g_j}
\left(\frac{2\pi m_e kT}{h^2}\right)^{3/2}
\exp\!\left(-\frac{\chi_j - \Delta\chi_j}{kT}\right)$$

closed by $n_e = n_h\sum_j j f_j$ and solved by a bracketed root-find on
$\log n_e$, with the ladder evaluated in log space for numerical range.

### 3.2 Continuum lowering

At near-solid density the isolated-atom potentials are strongly depressed.
Debye-Hückel screening with an ion-sphere floor (Stewart–Pyatt-like):

$$\Delta\chi_j = \frac{(j+1)e^2}{4\pi\varepsilon_0\max(\lambda_D, R_{\rm ion})}$$

At $n_e = 10^{27}$ m⁻³ and 1 eV this depresses Fe I from 7.90 eV to ~2.9 eV.
This is the "pressure ionisation" that Fletcher (2021) implements via the LMD
conductivity model. **Note:** LMD gives systematically *higher* ionisation
than equilibrium Saha with Debye-Hückel lowering, which is the leading
candidate for the framework's under-prediction of $\bar Z$ relative to
hydrocode results (Section 7).

### 3.3 Energy partition

$$E_{\rm res}\,m_{\rm atom} = E_{\rm cohesive} + \sum_j f_j\!\!\sum_{k<j}\chi_k
+ \tfrac{3}{2}kT(1+\bar Z)$$

Ionisation acts as a thermostat: pouring more energy in raises $\bar Z$ more
than it raises $T$, which is why plume temperatures cluster around 1–3 eV over
a wide velocity range.

### 3.4 Neglected physics

Radiative losses and electronic excitation are omitted. For a micron-scale
plume at 1–3 eV over ~1 µs the plume is optically thin and bremsstrahlung
losses are $\ll 1\%$ of the internal energy, so this is safe. It would **not**
be safe for a millimetre-scale impactor, where the plume is optically thick
early and radiative cooling matters.

### 3.5 Strong coupling

$$\Gamma = \frac{e^2}{4\pi\varepsilon_0 a_{\rm ws}\,kT_e}$$

At the Stage-1→2 handoff, $\Gamma \sim 0.7$–1 for typical parameters. Both
ideal Saha and the Spitzer collision rate are approximations there. The
Coulomb logarithm is floored at 2, which is the conventional patch but not a
theory. `pipeline.run_scenario` reports $\Gamma$ as a scenario warning when it
exceeds 1.

### 3.6 Tabulated inversion

Solving the ladder inside the expansion ODE costs a nested root-find per
evaluation and dominated runtime. `IonisationTable` tabulates $\bar Z(n,T)$
and $u(n,T)$ on a 64 × 160 log grid, then serves bilinear interpolants;
$u$ is monotone in $T$ at fixed $n$ so the inverse is a bracketed search on a
column. ~10³× faster, ~0.5–3% accurate (asserted in the test suite).

---

## 4. Stage 2 — expansion into vacuum

### 4.1 Self-similar ellipsoidal gas dynamics

After Anisimov, Bäuerle & Luk'yanchuk (1993), for an axisymmetric plume with
semi-axes $(R_r, R_r, R_z)$ and a Gaussian density profile:

$$\ddot R_i = \frac{P}{\rho R_i}, \qquad P = n_h kT(1+\bar Z)$$
$$\frac{\mathrm{d}u}{\mathrm{d}t} = -\frac{P}{n_h}\frac{\mathrm{d}\ln V}{\mathrm{d}t},
\qquad u = \tfrac{3}{2}kT(1+\bar Z) + E_{\rm ion}(n,T)$$

The ionisation energy sits **inside** $u$, so the effective adiabatic index is
computed self-consistently rather than fixed at 5/3. A recombining plasma
returns ionisation energy to the gas and cools much more slowly than $\gamma =
5/3$ would predict.

Energy conservation is verified: the test suite asserts total (kinetic +
internal) plume energy drifts by < 5% over the whole integration, and the
late-time solution is asserted to be ballistic ($R \propto t$ to 0.5%).

**Validity.** Self-similarity assumes the density profile *shape* is
preserved — well satisfied for free expansion into vacuum after a few initial
radii, and the standard treatment for laser-ablation plumes. It does not
resolve internal shocks, returning rarefactions, the plume/target-surface
interaction, or an ambient magnetic field. Fletcher (2021) shows a diamagnetic
cavity forming in the geomagnetic field for a normal impact; that is a real
low-frequency emission mechanism this framework does not model.

### 4.2 Shell-resolved freeze-out — and why single-zone fails

$$\tau_{\rm rec}^{-1} = \alpha_3 n_e^2 + \alpha_{\rm rr} n_e,
\qquad \alpha_3 = 8.75\times10^{-39}\,T_e^{-4.5}\ \mathrm{m^6/s},
\qquad \alpha_{\rm rr} = 2.7\times10^{-19}Z^2T_e^{-0.75}\ \mathrm{m^3/s}$$

Because $\tau_{\rm rec} \sim n_e^{-2}$, it varies by *many* orders of
magnitude across the plume's density profile. The tenuous outer shells freeze
out early at high charge state; the dense core stays in Saha equilibrium and
recombines almost completely.

A single-zone model evaluated at the peak density therefore predicts
essentially **zero** surviving charge — which was the behaviour during
development, and is in flat contradiction with retarding-potential-analyser
measurements. The Gaussian profile is discretised into 24 Lagrangian mass
shells, each with its own density history and its own freeze-out test; the
surviving charge is summed over shells. The test suite asserts that outer
shells freeze before inner ones.

**Approximation:** shells share the common temperature history from the
single-zone energy equation (justified by fast electron thermal conduction).
A fully multi-zone energy treatment would let shells have separate adiabats.

### 4.3 The collisional → collisionless transition

$$\nu_{ei} = 2.91\times10^{-12}\,n_e Z\ln\Lambda\,T_e^{-3/2}\ \mathrm{s^{-1}},
\qquad \omega_{pe} = \sqrt{\frac{n_ee^2}{\varepsilon_0 m_e}}$$

Since $\nu_{ei}\propto n_e$ and $\omega_{pe}\propto n_e^{1/2}$, an expanding
plasma *must* cross $\nu_{ei} = \omega_{pe}$. This crossing is the trigger for
the Close mechanism, and its location sets the emitted frequency. The
framework locates it at plume scales of ~10⁻⁵ m, matching Fig. 2 of Fletcher
(2021).

---

## 5. Stage 3 — charge separation and radiation

### 5.1 The mechanism

Close *et al.* (2010): at the collisionless transition, electrons at the plume
edge are free to stream ahead of the ions; the ambipolar field is a restoring
force; the displaced shell oscillates coherently at the local $\omega_{pe}$;
a coherently oscillating macroscopic dipole radiates.

### 5.2 Source strength — two closures

A shell at density $n_j$ carries coherently displaced charge
$Q_j = e\,n_j A_j \lambda_{D,j}$ displaced by $\delta_j$, giving $p_j = Q_j\delta_j$.

* **`debye`** (default, conservative): $\delta = \xi\lambda_D$, from ambipolar
  force balance. $\xi$ is the framework's one genuinely free O(1) parameter.
* **`fletcher`** (upper bound): electrons acquire a bulk drift
  $\sqrt{m_i/m_e}$ times the **ion bulk drift**, turned around after
  $\omega_{pe}^{-1}$, as *assumed* by Fletcher & Close (2017).

**A useful identity.** If Fletcher & Close's $\sqrt{m_i/m_e}$ factor is applied
to the ion *sound speed* rather than the bulk drift, then
$\sqrt{m_i/m_e}\,c_s = \sqrt{kT_e/m_e} = v_{te}$, and $v_{te}/\omega_{pe}$ is
exactly the Debye length — the two closures coincide. They differ *only*
because the factor is applied to the much larger bulk expansion velocity (tens
of km/s), which is what makes their predicted field 1–2 orders of magnitude
larger. The two closures bracket the measurement (Section 6).

### 5.3 Damping

$$\Gamma = \underbrace{\tfrac{1}{2}\nu_{ei}}_{\text{collisional}}
+ \underbrace{\sqrt{\tfrac{\pi}{8}}\frac{\omega_{pe}}{(k\lambda_D)^3}
e^{-1/2(k\lambda_D)^2 - 3/2}}_{\text{Landau}}
+ \underbrace{\tfrac{1}{2}\omega_{pe}\lambda_D/L}_{\text{dephasing}}$$

The dephasing term — shells at different densities oscillating at different
frequencies and losing phase coherence — dominates, and is what converts a
narrow plasma line into the observed *broadband* pulse.

### 5.4 Spectrum computed analytically

Each shell contributes $p_j(t) = p_j e^{-\Gamma_j t}\sin(\omega_j t)$, whose
Fourier transform is a Lorentzian
$P_j(\omega) = p_j\omega_j/[(\Gamma_j + i\omega)^2 + \omega_j^2]$. Summing
analytically avoids time-stepping at 10⁻¹⁵ s for 10⁻⁸ s (10⁷ samples); the
waveform is recovered by inverse FFT onto whatever band is of interest.

### 5.5 Radiation — near field included

$$|E_\theta| = \frac{p}{4\pi\varepsilon_0}
\sqrt{\left(\frac{k^2}{r} - \frac{1}{r^3}\right)^2 + \left(\frac{k}{r^2}\right)^2}\,\sin\theta$$

The near-field ($1/r^3$) and induction ($1/r^2$) terms are **not** negligible
for real experiment geometry: at 315 MHz, $\lambda = 0.95$ m, so a patch
antenna at 0.30 m sits at $kr \approx 2$ — formally the transition zone.
Reporting only the far-field term there is a ~10% error, and much larger at
lower frequencies or closer sensors. The dipole approximation for the *source*
remains valid ($L \ll \lambda$) and is checked and reported.

### 5.6 The frequency problem — stated, not hidden

The spectral peak sits near $\omega_{pe}$ at the transition, which for a fresh
plume is 10¹³–10¹⁴ rad/s — far above the 315/916 MHz where Close *et al.*
(2013) detect emission. **Fletcher & Close report exactly the same
discrepancy** ("produces emission at a frequency higher than that detected in
experiments") and it remains the central open problem in the field. This
framework reproduces the discrepancy rather than papering over it.

`emp.emission_at_frequency` therefore provides the *resonant-shell* view: an
antenna at frequency $f$ responds to whichever part of the plume has
$\omega_{pe} = 2\pi f$ at that instant, which is a late, tenuous shell
(~10¹⁶ m⁻³ for 916 MHz). Below the local plasma frequency a wave is
evanescent, so this is also the only part of the plume from which that
frequency can escape. Both the broadband spectral peak and the resonant-shell
amplitude are reported.

---

## 6. Validation

`python -m hvi_emp.validation` runs 36 comparisons against published data;
all currently pass. Highlights:

### Shock phase-change thresholds (no free parameters)

Release terminated at the spinodal, versus Ahrens & O'Keefe / Melosh:

| material | quantity | model [GPa] | reference [GPa] | ratio |
|---|---|---|---|---|
| Al | complete melt | 94 | 110 | 0.86 |
| Al | incipient vaporisation | 374 | 380 | **0.98** |
| Al | complete vaporisation | 1504 | 1500 | **1.00** |
| Cu | complete melt | 179 | 270 | 0.66 |
| Cu | incipient vaporisation | 448 | 580 | 0.77 |
| Cu | complete vaporisation | 1675 | 1500 | 1.12 |
| Fe | incipient vaporisation | 373 | 890 | 0.42 |
| Fe | complete vaporisation | 1352 | 2000 | 0.68 |
| W | complete vaporisation | 2839 | 4000 | 0.71 |

Terminating release at $V_0$ instead gives Al incipient vaporisation at
163 GPa (2.3× low) and complete at 571 GPa (2.6× low) — which is how the
spinodal endpoint was selected.

### Plasma state

| quantity | model | reference | source |
|---|---|---|---|
| $T_e$, W→Al 40 km/s | 1.1 eV | 2–2.5 eV | Fletcher & Close (2017) §III |
| asymptotic plume speed | 22 km/s | ~30 km/s | Fletcher & Close (2017) |
| early peak density | 6.6 × 10²⁵ m⁻³ | ~10²⁷ m⁻³ | Fletcher & Close (2017) |

### Charge yield scaling

| quantity | model | reference |
|---|---|---|
| $\beta$ in $Q_{\rm frozen} \sim v^\beta$, above 40 km/s | 5.0 | 3.4–3.5 (Close 2013; McBride & McDonnell 1999) |
| $\beta$ for $Q$ at formation, above 40 km/s | **3.2** | 3.4–3.5 |
| $\beta$ across the threshold (12–40 km/s) | 8–13 | (a prediction, not fitted) |
| absolute $Q$, Fe→Al 1 pg 50 km/s | 6 × 10⁻⁷ C | 6 × 10⁻⁸ C (empirical prefactor, ±1 decade) |

The velocity exponent is the robust, repeatedly-measured feature, and the
framework lands in the right range without being fitted to it. Two caveats
that must be stated together with that claim:

* The exponent depends on **which** charge is being counted. The charge at
  formation gives $\beta = 3.2$, close to the measured value; after
  shell-resolved freeze-out it steepens to $\beta \approx 5$, because faster
  impacts freeze out at higher charge state as well as producing more vapour.
  Published experiments collect ions with an RPA some distance from the
  crater, so the measured quantity is closer to the frozen charge — meaning
  the model's frozen-charge exponent is genuinely ~1.4× steeper than measured,
  not merely a definitional mismatch.
* The exponent is **strongly range-dependent**: fitting across the
  vaporisation threshold gives 8–13. Published fits span 3–66 km/s and
  therefore average across a threshold that the model places at ~26 km/s for
  Fe→Al. A like-for-like comparison would require re-fitting the published
  data over the same restricted range.

The prefactor is high by ~10×, within the scatter of published prefactors.

### EMP amplitude — bracketed, not claimed

Close *et al.* (2013) measured **1.9 mV/m at 0.30 m, 916 MHz**, from 1–100 fg
iron projectiles at 3–66 km/s. Fletcher & Close's DG-PIC predicted 9.8 mV/m
(5× high, as they note).

| | E at 916 MHz, 0.30 m |
|---|---|
| this framework, `debye` closure | 3.8 × 10⁻⁵ V/m |
| **measurement (Close 2013)** | **1.9 × 10⁻³ V/m** |
| geometric mean of the two closures | 7.4 × 10⁻⁴ V/m |
| Fletcher & Close DG-PIC | 9.8 × 10⁻³ V/m |
| this framework, `fletcher` closure | 1.4 × 10⁻² V/m |

The measurement lies inside the closure bracket. That bracket — not either
endpoint — is the honest statement of what this framework can say about
absolute EMP amplitude.

### Chamber conditions and dust

* At 10⁻⁶ Torr (Close *et al.*) the plume stopping distance is ~3 cm — a real
  vacuum over the region that matters, as they intended.
* At 1 Torr (Bianchi *et al.* 1984) it is ~0.3 mm: the plume is collisionally
  arrested within a fraction of a millimetre, so their ~300 kHz emission
  **cannot** be the coherent plasma-oscillation mechanism. This is consistent
  with their own argument that the plasma scale is far too small for a dipole
  at that frequency, and it supports attributing slow-impact RF to
  micro-cracking (Maki *et al.* 2004, 2005) instead.
* Ejecta dust: optical depth ~6.5 at 550 nm (the visible impact flash) and
  ~6 × 10⁻⁷ at 916 MHz, with size parameter $x \approx 2\times10^{-6}$. The
  debris cloud is completely transparent to the EMP. Any observed RF
  attenuation comes from the **plasma**, not the dust — a quantitative answer
  to a frequently-asserted worry (Tishkovets *et al.* 2011 framework, Rayleigh
  limit).

---

## 7. Honest limitations

**Read this before quoting any absolute number.**

### 7.1 The four free parameters and their leverage

Measured by `examples/03_sensitivity.py` over physically defensible ranges,
for Fe 1 pg → Al at 50 km/s:

| parameter | range | spread in $m_{\rm vap}$ | in $T_e$ | in $Q$ | in E@916 MHz |
|---|---|---|---|---|---|
| `n_decay` | 1.5–3.0 | 3.5× | 1.4× | 1.4× | 2.1× |
| `core_scale` | 0.5–1.5 | 6.6× | 1.6× | 1.3× | 2.1× |
| `plume_expansion_factor` | 2–5 | 1.0× | 1.2× | 6.4× | 8.6× |
| `separation_factor` | 0.5–2 | — | — | — | 4.0× |

Combined, the **absolute EMP amplitude is uncertain by roughly two orders of
magnitude**, and the closure choice (§5.2) adds another two. Charge yield is
good to ~1 decade; the *velocity scaling* and the phase-change thresholds are
the reliable outputs.

### 7.2 Known physics gaps

1. **The frequency discrepancy** (§5.6). Unresolved in the literature as well
   as here.
2. **Ionisation state under-predicted** relative to hydrocodes. Fletcher
   (2021) reports full ionisation above 15–20 km/s via the LMD conductivity
   model; equilibrium Saha with Debye-Hückel lowering gives $\bar Z \sim
   0.3$–0.6 at those speeds. The likely cause is that LMD's pressure
   ionisation is more aggressive than Debye-Hückel continuum lowering.
3. **Strongly coupled regime.** At handoff $\Gamma \sim 1$; ideal Saha and
   Spitzer collisions are both approximations, and $\ln\Lambda$ is floored at
   2 by convention rather than by theory.
4. **No magnetic field.** Fletcher (2021) shows a diamagnetic cavity forming
   in the geomagnetic field, collapsing as the plume thins — a genuine
   low-frequency emission mechanism entirely absent here.
5. **No dust–plasma coupling.** Parker Solar Probe results suggest the dust
   component affects plume dynamics and emission; the framework treats dust
   only as a (negligible) RF scatterer.
6. **No target bias.** Close *et al.* found RF detection rates rise
   substantially for *biased* targets; this framework models an unbiased,
   grounded surface.
7. **Single-temperature electrons and ions.** No $T_e \ne T_i$, no non-Maxwellian
   tails, both of which a PIC code would resolve.
8. **Iron's EOS** is compromised by the α→ε transition folded into its linear
   $U_s$–$u_p$ fit (§2.3).
9. **Stage 4 thresholds are generic**, not qualification data. `UPSET_THRESHOLDS`
   carries sources and should be replaced with numbers for the actual design.

### 7.3 What the framework is good for

* Ranking scenarios — which impacts matter, which do not.
* Parameter studies and sensitivity bounding, cheaply.
* Reproducing measured *scalings* (velocity exponent, thresholds).
* Generating physically consistent initial conditions for a real PIC run.
* Making the open problems in the field explicit and testable.

### 7.4 What it is not for

* Root-cause determination for a specific anomaly on its own.
* Absolute EMP amplitude to better than ~2 orders of magnitude.
* Any impact above ~30 km/s without substituting a tabular EOS.
* Oblique impacts beyond ~60° from normal.
* Millimetre-and-larger impactors, where radiative cooling and
  optically-thick plume physics enter.

---

## 8. References

**Impact physics and EOS**
- Melosh, H.J., *Impact Cratering: A Geologic Process*, OUP (1989).
- Zel'dovich & Raizer, *Physics of Shock Waves and High-Temperature Hydrodynamic Phenomena*, Dover (2002).
- Ahrens & O'Keefe, "Shock melting and vaporization of lunar rocks and minerals", *The Moon* **4**, 214 (1972).
- Pierazzo, Vickery & Melosh, *Icarus* **127**, 408 (1997).
- Vinet, Ferrante, Rose & Smith, *J. Geophys. Res.* **92**, 9319 (1987).
- Marsh (ed.), *LASL Shock Hugoniot Data*, UC Press (1980).

**Impact plasmas and RF emission**
- Close, Colestock, Cox, Kelley & Lee, *J. Geophys. Res.* **115**, A12328 (2010).
- Close *et al.*, *Phys. Plasmas* **20**, 092102 (2013).
- Fletcher, Close & Mathias, *Phys. Plasmas* **22**, 093504 (2015).
- Fletcher & Close, *Phys. Plasmas* **24**, 053102 (2017). *(uploaded)*
- Fletcher, A.C., "Characterizing Electromagnetic Pulses from Hypervelocity Impact Plasmas", NRL/MR/6757‑20‑10,138 (2021). *(uploaded, AD1123389)*
- Crawford, D.A., *Procedia Engineering* **103**, 89 (2015). *(uploaded)*
- Crawford & Schultz, *Int. J. Impact Eng.* **14**, 205 (1993); **23**, 169 (1999).
- Jean & Rollins, *AIAA J.* **8**, 1742 (1970) — impact flash, jetting, the critical angle and jet-velocity tables. *(uploaded; the oldest data in the validation suite)*
- Bianchi *et al.*, *Nature* **308**, 830 (1984). *(uploaded)*
- Maki *et al.*, *Adv. Space Res.* **34**, 1085 (2004); *J. Appl. Phys.* **97**, 104911 (2005).
- Starks *et al.*, *Int. J. Impact Eng.* **33**, 781 (2006).

**Jetting and shaped charges**
- Birkhoff, MacDougall, Pugh & Taylor, *J. Appl. Phys.* **19**, 563 (1948) — the steady-jet theory.
- Walsh, Shreffler & Willig, *J. Appl. Phys.* **24**, 349 (1953) — limiting conditions for jet formation; the critical-angle criterion.
- Jean & Rollins, *AIAA J.* **8**, 1742 (1970) — the measurements `jetting.py` is validated against.

**Standards and policy**
- ESA, *Zero Debris Technical Booklet* (2025) — §3.4 on small, untrackable particles that cannot be avoided; §3.5 A.II–III on predicting how hypervelocity impacts affect internal items. *(uploaded)*

**Plasma theory**
- Anisimov, Bäuerle & Luk'yanchuk, *J. Appl. Phys.* **73**, 8337 (1993).
- Gurevich, Pariiskaya & Pitaevskii, *Sov. Phys. JETP* **22**, 449 (1966).
- Mora, *Phys. Rev. Lett.* **90**, 185002 (2003).
- Stewart & Pyatt, *ApJ* **144**, 1203 (1966).
- NRL Plasma Formulary (2019).
- Kodis, R.D., "Propagation and scattering in plasmas" (1965). *(uploaded)*
- Ginzburg, *Propagation of Electromagnetic Waves in Plasma*, Gordon & Breach (1961).

**Scattering and spacecraft effects**
- Tishkovets, Petrova & Mishchenko, *JQSRT* **112**, 2095 (2011). *(uploaded)*
- Bohren & Huffman, *Absorption and Scattering of Light by Small Particles*, Wiley (1983).
- Caswell, McBride & Taylor, "OLYMPUS end of life anomaly — a Perseid meteoroid impact event?", *Int. J. Impact Eng.* **17**, 139 (1995). *(uploaded)*
- Foschini, L., *Europhys. Lett.* **43**, 226 (1998).
- Goel & Close, *Adv. Space Res.* (spacecraft anomaly statistics).
- NASA-HDBK-4002A, *Mitigating In-Space Charging Effects*.
- Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999), Ch. 9.


## Oblique incidence: the plume is axisymmetric and should not be

`angle_deg` enters the model in exactly one place: it reduces the impact
speed to its normal component, `v_normal = v cos(theta)`. Everything
downstream — the shock, the vaporised mass, the plume geometry and the
direction of the centre-of-mass drift — is computed as though the impact
were normal, just weaker.

Measured, Fe on Al at 50 km/s:

| angle | v_normal | m_plasma | R_z/R_r | drift direction |
|---|---|---|---|---|
| 0° | 50.0 km/s | 1.55e-12 kg | 1.366 | surface normal |
| 30° | 43.3 km/s | 1.03e-12 kg | 1.386 | surface normal |
| 45° | 35.4 km/s | 5.04e-13 kg | 1.422 | surface normal |

The aspect ratio barely moves (that residual drift is a temperature effect,
not a geometric one), and the plume drifts straight up the surface normal at
every angle. **The model cannot produce an asymmetric or downrange-directed
plume**, which is one of the best-established features of real oblique
hypervelocity impacts.

Consequences worth knowing before using an oblique scenario:

* Any slice perpendicular to the surface normal is a perfect circle, at every
  incidence angle. That symmetry is a property of the model, not a result.
* The radiated dipole is oriented along the surface normal regardless of
  incidence, so the predicted angular pattern of the EMP is wrong for
  oblique impacts even where the amplitude is roughly right.
* `conf/scenario/debris_1mm.yaml` uses 45°, because that is the realistic
  case — and it is therefore the scenario most affected by this limitation.

Fixing it needs a genuinely 3-D treatment: either a hydrocode or an
asymmetric closure for the initial plume with the downrange momentum carried
through Stage 2. Neither is a small change, and neither should be faked with
a tilt applied to a symmetric Gaussian.

### What to do about it now

The hydrocode route is available. **M2C** (`hvi_emp.solvers.m2c_stage1`) is
3-D Cartesian, so an oblique impact is a geometry change rather than a
modelling impossibility, and the bridge switches to a 3-D mesh automatically
whenever `angle_deg > 0`:

```bash
python examples/13_m2c_oblique_impact.py m2c_runs
```

iSALE does *not* help here — iSALE2D is axisymmetric, which is the same
restriction this section describes. M2C is also GPLv3 with no application
process, so it is the practical choice as well as the capable one; see
`docs/SOLVERS.md` §5.

This does not repair the reduced chain. An M2C run tells you what the
asymmetry actually looks like for one case; it does not give Stage 2 a
downrange closure. Until such a closure exists, an oblique scenario run
through the reduced chain should be read as *the normal-incidence answer at
reduced speed*, which is what it is.
