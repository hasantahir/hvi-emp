"""Stage 3 -- charge separation, coherent oscillation and EMP radiation.

Mechanism
---------
Close et al., J. Geophys. Res. 115, A12328 (2010) proposed, and Fletcher &
Close, Phys. Plasmas 24, 053102 (2017) tested with a DG-PIC code, the
following chain:

1.  The expanding plume crosses from collisional to collisionless
    (`expansion` Stage 2).  Because nu_ei ~ n_e but omega_pe ~ n_e^(1/2), this
    crossing is unavoidable.
2.  Electrons at the plume edge, no longer collisionally locked to the ions,
    stream ahead over roughly a Debye length.
3.  The resulting ambipolar field is a restoring force: the displaced electron
    shell oscillates at the *local* electron plasma frequency.
4.  Because the displacement is coherent over the shell, the oscillation
    carries a macroscopic dipole moment, and an oscillating dipole radiates.

Implementation
--------------
**Source.**  A shell at density n_j carries a coherently displaced charge
``Q_j = e n_j A_j lambda_D,j`` displaced by ``delta_j = xi lambda_D,j``, giving
a dipole moment ``p_j = Q_j delta_j``.  The displacement scale is set by the
Debye length because that is where the ambipolar field balances the electron
pressure; ``xi`` (`separation_factor`) is the one O(1) parameter of the model
and is exposed explicitly.  An alternative, more aggressive closure follows
Fletcher & Close in assuming the electrons acquire a bulk drift
sqrt(m_i/m_e) times the ion drift; both are selectable, and the difference is
a direct measure of the model uncertainty.

**Spectrum, analytically.**  Each shell contributes a damped sinusoid
``p_j(t) = p_j exp(-Gamma_j t) sin(omega_j t)`` whose Fourier transform is a
Lorentzian.  Summing analytically over the density profile avoids having to
time-step at 1e-15 s for 1e-8 s (1e7 samples); the time-domain waveform is
then recovered by inverse FFT over whatever band the user actually cares
about.  The radiated far field of a dipole is

    E_theta(r, t) = mu_0 sin(theta) p_ddot(t - r/c) / (4 pi r),   B = E/c

so in the frequency domain ``E(omega) = -mu_0 sin(theta) omega^2 P(omega)
/ (4 pi r)``.

**Damping** combines (i) electron-ion collisions nu_ei/2, (ii) Landau damping
when k lambda_D is not small, and (iii) inhomogeneous dephasing -- shells at
different densities oscillate at different frequencies and lose phase
coherence.  The dephasing term dominates and is what converts a narrow
plasma line into the observed broadband pulse.

**Why the predicted frequency is high.**  The peak of the spectrum sits near
omega_pe at the transition, which for a fresh impact plume is 1e13-1e14 rad/s
-- far above the 315/916 MHz at which Close et al. (2013) detect emission.
Fletcher & Close report exactly the same discrepancy ("produces emission at a
frequency higher than that detected in experiments") and it remains the
central open problem in the field.  `emission_at_frequency` therefore also
provides the *resonant-shell* view: an antenna at frequency f responds to
whichever part of the plume has omega_pe = 2 pi f at that instant, which is a
late, tenuous shell.  Both numbers are reported so the discrepancy is visible
rather than hidden.

References
----------
Close, Colestock, Cox, Kelley & Lee, J. Geophys. Res. 115, A12328 (2010).
Close et al., Phys. Plasmas 20, 092102 (2013) (315/916 MHz measurements).
Fletcher & Close, Phys. Plasmas 24, 053102 (2017).
Jackson, *Classical Electrodynamics*, 3rd ed., Ch. 9 (dipole radiation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import (C_LIGHT, E_CHARGE, EPS_0, EV, K_B, M_ELECTRON, MU_0)
from .expansion import ExpansionResult
from .ionization import debye_length, plasma_frequency
from ._compat import trapezoid


# ---------------------------------------------------------------------------
# Charge separation
# ---------------------------------------------------------------------------

@dataclass
class ChargeSeparation:
    """Coherently displaced charge in one plume shell."""
    n_e: float           # m^-3
    T_eV: float
    area: float          # m^2, shell area participating coherently
    lambda_De: float     # m
    Q: float             # C, separated charge
    delta: float         # m, displacement
    p0: float            # C m, dipole moment
    omega: float         # rad/s
    gamma: float         # 1/s, total damping rate
    E_ambipolar: float   # V/m


#: Charge separation in Debye lengths, CALIBRATED against one measurement.
#:
#: What this number is
#: -------------------
#: The EMP amplitude is uncertain by a factor of ~184 in this framework, and
#: the entire factor is one quantity: the electron-ion separation `delta`.
#: The two first-principles closures are the *same* physics with different
#: choices of the velocity that sets it --
#:
#:     debye     delta = xi * lambda_D              (= xi * v_te / omega_pe)
#:     fletcher  delta = xi * sqrt(m_i/m_e) * v_bulk / omega_pe
#:
#: and they coincide exactly when the ion velocity is taken to be the ion
#: sound speed, because sqrt(m_i/m_e) c_s = v_te. Fletcher & Close apply the
#: mass-ratio factor to the *bulk expansion* speed instead, which at 50 km/s
#: is 15x the sound speed. That single choice is the whole bracket.
#:
#: Measured against Close et al. (2013), 916 MHz at 0.30 m:
#:
#:     debye    (xi=1)   1.29e-4 V/m   15x   LOW
#:     fletcher          2.37e-2 V/m   12.5x HIGH
#:     measured          1.90e-3 V/m
#:
#: E is exactly linear in delta, so one measurement fixes xi:
#: **xi = 15 reproduces the measurement to 2%.**
#:
#: What was tried and did not work
#: -------------------------------
#: An energy-conservation bound on the aggressive closure. It implies 118 eV
#: of drift energy per electron against 1.5 eV of thermal energy -- but at
#: Zbar ~ 0.01 there are ~100 ions per electron, so the ion bulk reservoir
#: holds ~11,800 eV per electron and 118 eV is ~1% of it. Implausible, but
#: not forbidden, so energy does not constrain it. Recorded because a
#: negative result that is not written down gets re-attempted.
#:
#: What this is and is not
#: -----------------------
#: This is a **calibration against a single published measurement**, not a
#: derivation. Using it, the framework no longer *predicts* the absolute
#: amplitude at that point -- it reproduces it by construction. What remains
#: predictive is every *scaling*: with impact speed, projectile mass,
#: material, observation frequency and sensor distance, none of which were
#: used to fit xi.
#:
#: It is also a falsifiable claim about the plasma: it says the charge
#: separation is ~15 Debye lengths. A PIC run (`solvers/picongpu_stage3`)
#: measures delta directly and either confirms that or does not.
CALIBRATED_XI = 15.0

#: Where CALIBRATED_XI came from, carried with the result so a number can be
#: traced to its source.
CALIBRATION_SOURCE = (
    "Close et al., Phys. Plasmas 20, 092102 (2013): 1.9e-3 V/m at 916 MHz, "
    "0.30 m, Fe on Al. Single point; xi is linear in E so it is exactly "
    "determined by it.")


def separation_from_shell(n_e: float, T_eV: float, area: float,
                          nu_ei: float, scale_length: float,
                          separation_factor: float = 1.0,
                          closure: str = "debye",
                          m_ion: float = 4.48e-26,
                          v_drift: float = 0.0) -> ChargeSeparation:
    """Build the dipole source for one shell.

    Parameters
    ----------
    n_e, T_eV : float
        Local electron density [m^-3] and temperature [eV].
    area : float
        Coherent shell area [m^2].
    nu_ei : float
        Electron-ion collision frequency [1/s] (collisional damping).
    scale_length : float
        Density scale length [m], used for the ambipolar field and for the
        inhomogeneous-dephasing damping rate.
    separation_factor : float
        The O(1) coefficient xi in delta = xi * lambda_D.
    closure : {"debye", "calibrated", "fletcher"}
        "debye"      -- displacement of order the Debye length (conservative;
                        follows from ambipolar force balance). 15x below the
                        one available measurement.
        "calibrated" -- the same form with xi = CALIBRATED_XI = 15, fixed by
                        Close et al. (2013). Reproduces that measurement to
                        2%. Use this for absolute amplitudes, and read
                        CALIBRATED_XI's docstring before quoting one.
        "fletcher" -- electrons acquire a bulk drift sqrt(m_i/m_e) times the
                      *ion bulk* drift, as assumed by Fletcher & Close (2017),
                      and are turned around after ~1/omega_pe.  Much larger
                      dipole; read it as an upper bound.
    m_ion : float
        Ion mass [kg], used by the "fletcher" closure.
    v_drift : float
        Ion bulk drift speed [m/s], used by the "fletcher" closure.

    Notes
    -----
    A useful identity: if the ion drift in Fletcher & Close's assumption is
    taken to be the *ion sound speed* c_s = sqrt(kT_e/m_i), then
    sqrt(m_i/m_e) c_s = sqrt(kT_e/m_e) = v_te, and v_te / omega_pe is exactly
    the Debye length.  The two closures then coincide.  They differ only
    because Fletcher & Close apply the sqrt(m_i/m_e) factor to the much larger
    *bulk expansion* velocity (tens of km/s), which is what makes their
    predicted field ~1-2 orders of magnitude larger.
    """
    if n_e <= 0 or T_eV <= 0 or area <= 0:
        return ChargeSeparation(n_e, T_eV, area, np.inf, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0)
    lam = debye_length(n_e, T_eV)
    w_pe = plasma_frequency(n_e)
    v_te = np.sqrt(T_eV * EV / M_ELECTRON)

    if closure == "debye":
        delta = separation_factor * lam
    elif closure == "calibrated":
        # Same functional form as "debye", with xi fixed by measurement
        # rather than assumed to be O(1). See CALIBRATED_XI.
        delta = separation_factor * CALIBRATED_XI * lam
    elif closure == "fletcher":
        # electrons acquire a bulk drift sqrt(mi/me) x the ion bulk drift and
        # are turned around by the ambipolar field after ~1/omega_pe.  Falls
        # back to the sound speed (== the Debye result) if no drift is given.
        v_i = v_drift if v_drift > 0 else np.sqrt(T_eV * EV / m_ion)
        v_e = min(np.sqrt(m_ion / M_ELECTRON) * v_i, 0.1 * C_LIGHT)
        delta = separation_factor * v_e / w_pe
    else:
        raise ValueError(
            f"unknown closure {closure!r}; expected 'debye' (conservative), "
            f"'calibrated' (xi from Close 2013) or 'fletcher' (aggressive)")

    Q = E_CHARGE * n_e * area * lam
    p0 = Q * delta
    E_amb = T_eV * EV / (E_CHARGE * max(scale_length, lam))

    # damping: collisional + Landau + inhomogeneous dephasing
    g_coll = 0.5 * nu_ei
    k = 2.0 * np.pi / max(scale_length, lam)
    kl = k * lam
    if kl < 1e-3:
        g_landau = 0.0
    else:
        # Landau damping of a Langmuir wave, standard weak-damping form
        g_landau = (np.sqrt(np.pi / 8.0) * w_pe / kl**3
                    * np.exp(-1.0 / (2.0 * kl**2) - 1.5))
    # dephasing: shells within one scale length differ in omega_pe by
    # ~ omega_pe * (lambda/L) / 2 (since omega ~ n^1/2)
    g_deph = 0.5 * w_pe * lam / max(scale_length, lam)

    gamma = g_coll + g_landau + g_deph
    return ChargeSeparation(n_e, T_eV, area, lam, Q, delta, p0, w_pe,
                            gamma, E_amb)


# ---------------------------------------------------------------------------
# EMP result
# ---------------------------------------------------------------------------

@dataclass
class EMPResult:
    """Radiated field from the impact plasma."""
    f: np.ndarray             # Hz, positive frequencies
    E_spec: np.ndarray        # V/m/Hz, complex spectral density at the sensor
    t: np.ndarray             # s, time relative to the retarded trigger
    E_t: np.ndarray           # V/m
    B_t: np.ndarray           # T
    r_sensor: float           # m
    theta_deg: float
    sources: list             # list[ChargeSeparation]
    energy_radiated: float    # J
    peak_field: float         # V/m
    diagnostics: dict = field(default_factory=dict)

    @property
    def psd(self) -> np.ndarray:
        """|E(f)|^2, one-sided [V^2 m^-2 Hz^-2]."""
        return np.abs(self.E_spec) ** 2

    def band_field(self, f_lo: float, f_hi: float) -> float:
        """RMS field [V/m] in a band, from Parseval on the one-sided spectrum."""
        m = (self.f >= f_lo) & (self.f <= f_hi)
        if not m.any():
            return 0.0
        df = np.gradient(self.f)[m]
        return float(np.sqrt(2.0 * np.sum(self.psd[m] * df) * np.mean(df)))

    def summary(self) -> str:
        return "\n".join([
            f"EMP at r = {self.r_sensor:.3f} m, theta = {self.theta_deg:.0f} deg",
            f"  peak |E|            {self.peak_field:.3e} V/m",
            f"  peak |B|            {self.peak_field/C_LIGHT:.3e} T",
            f"  radiated energy     {self.energy_radiated:.3e} J",
            f"  spectral peak       "
            f"{self.f[np.argmax(self.psd)]:.3e} Hz",
            f"  band 100-500 MHz    {self.band_field(1e8, 5e8):.3e} V/m",
            f"  band 0.5-2 GHz      {self.band_field(5e8, 2e9):.3e} V/m",
        ])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def simulate_emp(exp: ExpansionResult,
                 r_sensor: float = 0.30,
                 theta_deg: float = 90.0,
                 f_max: float = 5.0e9,
                 n_freq: int = 4096,
                 separation_factor: float = 1.0,
                 closure: str = "debye",
                 coherent: bool = True,
                 n_epochs: int = 60,
                 rng_seed: int = 0) -> EMPResult:
    """Radiated EMP at a sensor, from the Stage-2 expansion history.

    The plume is sampled at `n_epochs` times from the collisional ->
    collisionless transition onward.  At each epoch the outer shell is a
    dipole oscillator at the local omega_pe; the analytic Lorentzian spectra
    are summed and inverse-transformed onto the requested band.

    Parameters
    ----------
    exp : ExpansionResult
    r_sensor : float
        Sensor standoff [m].  Close et al. (2013) used patch antennas at
        0.30 m, which is why that is the default.
    theta_deg : float
        Angle from the dipole axis (the surface normal).  90 deg is broadside
        (maximum), 0 deg is on-axis (null).
    f_max, n_freq : float, int
        Band and resolution of the reported spectrum and waveform.
    separation_factor, closure :
        Passed to `separation_from_shell`.
    coherent : bool
        Sum shell contributions with a common phase (coherent, upper bound) or
        with random phases (incoherent, lower bound).  Reality is in between;
        running both brackets the answer.
    """
    if not exp.transition:
        raise ValueError(
            "No collisional->collisionless transition found in the expansion "
            "history; there is no EMP source. Extend t_end or check Stage 1.")

    t0 = exp.transition["t"]
    sel = np.where(exp.t >= t0)[0]
    if sel.size < 4:
        raise ValueError("Too few expansion samples after the transition.")

    idx = np.unique(np.linspace(sel[0], len(exp.t) - 1,
                                min(n_epochs, sel.size)).astype(int))
    rng = np.random.default_rng(rng_seed)

    f = np.linspace(0.0, f_max, n_freq)
    w = 2.0 * np.pi * f
    P = np.zeros_like(w, dtype=complex)
    sources: list[ChargeSeparation] = []

    m_ion = exp.material.m_atom
    for i in idx:
        n_e = exp.n_e[i]
        if n_e <= 0 or exp.T_eV[i] <= 0:
            continue
        Rr, Rz = exp.R_r[i], exp.R_z[i]
        # Coherent area: the leading hemisphere of the ellipsoid.
        area = 2.0 * np.pi * Rr * Rz
        L = np.sqrt(Rr * Rz)                      # density scale length
        v_bulk = float(np.hypot(exp.v_r[i], exp.v_z[i]))
        cs = separation_from_shell(
            n_e, exp.T_eV[i], area, exp.nu_ei[i], L,
            separation_factor=separation_factor, closure=closure,
            m_ion=m_ion, v_drift=v_bulk)
        if cs.p0 <= 0 or cs.omega <= 0:
            continue
        sources.append(cs)

        # Fourier transform of p(t) = p0 exp(-g t) sin(w0 t) H(t):
        #   P(w) = p0 w0 / ((g + i w)^2 + w0^2)
        g, w0 = cs.gamma, cs.omega
        denom = (g + 1j * w) ** 2 + w0**2
        contrib = cs.p0 * w0 / denom
        if coherent:
            phase = np.exp(-1j * w * (exp.t[i] - t0))
        else:
            phase = np.exp(1j * rng.uniform(0, 2 * np.pi))
        P += contrib * phase

    if not sources:
        raise ValueError("No radiating shells found.")

    sin_th = np.sin(np.radians(theta_deg))
    # E(w) = -mu0 sin(theta) w^2 P(w) / (4 pi r)
    E_spec = -MU_0 * sin_th * w**2 * P / (4.0 * np.pi * r_sensor)

    # --- time domain via inverse FFT (real signal) -----------------------
    df = f[1] - f[0]
    n_t = 2 * (n_freq - 1)
    E_t = np.fft.irfft(E_spec, n=n_t) * (n_freq - 1) * 2.0 * df
    t = np.arange(n_t) / (n_t * df)
    B_t = E_t / C_LIGHT

    # --- radiated energy: Larmor integrated over the source spectrum -----
    # dU/dw = mu0 w^4 |P(w)|^2 / (6 pi c)  (one-sided)
    dUdw = MU_0 * w**4 * np.abs(P) ** 2 / (6.0 * np.pi * C_LIGHT)
    U = float(trapezoid(dUdw, w))

    return EMPResult(
        f=f, E_spec=E_spec, t=t, E_t=E_t, B_t=B_t,
        r_sensor=r_sensor, theta_deg=theta_deg, sources=sources,
        energy_radiated=U, peak_field=float(np.max(np.abs(E_t))),
        diagnostics={
            "t_transition": t0, "n_sources": len(sources),
            "closure": closure, "coherent": coherent,
            "separation_factor": separation_factor,
            "omega_peak": float(sources[0].omega),
            "f_pe_transition": exp.transition["f_pe"],
            "dipole_moments": np.array([s.p0 for s in sources]),
            "omegas": np.array([s.omega for s in sources]),
            "gammas": np.array([s.gamma for s in sources]),
            "P_spec": P,
        },
    )


# ---------------------------------------------------------------------------
# Exact (near + far field) dipole radiation
# ---------------------------------------------------------------------------

def dipole_field(p_amp: float, omega: float, r: float,
                 theta_deg: float = 90.0) -> dict:
    """Exact field of a harmonic point dipole, all terms retained.

    Jackson (3rd ed.) eq. 9.18.  For theta = 90 deg,

        |E| = (p / 4 pi eps0) sqrt[ (k^2/r - 1/r^3)^2 + (k/r^2)^2 ] sin(theta)

    The near-field (1/r^3) and induction (1/r^2) terms are *not* negligible
    for the geometry of real impact experiments: at 315 MHz the free-space
    wavelength is 0.95 m, so a patch antenna 0.30 m away sits at kr ~ 2 and is
    formally in the transition zone.  Reporting only the far-field 1/r term
    there is a ~10% error, and much larger for lower frequencies or closer
    sensors.
    """
    if r <= 0 or omega <= 0 or p_amp == 0:
        return {"E": 0.0, "B": 0.0, "kr": 0.0, "far_field": False}
    k = omega / C_LIGHT
    sin_t = np.sin(np.radians(theta_deg))
    radial = np.sqrt((k**2 / r - 1.0 / r**3) ** 2 + (k / r**2) ** 2)
    E = p_amp * radial * sin_t / (4.0 * np.pi * EPS_0)
    # magnetic field: (1/r) and (1/r^2) terms only
    B = (MU_0 * omega * p_amp * sin_t / (4.0 * np.pi)
         * np.sqrt((k / r) ** 2 + (1.0 / r**2) ** 2))
    return {"E": float(E), "B": float(B), "kr": float(k * r),
            "far_field": bool(k * r > 3.0),
            "far_field_only_E": float(p_amp * k**2 * sin_t
                                      / (4.0 * np.pi * EPS_0 * r))}


# ---------------------------------------------------------------------------
# Resonant-shell view: what an antenna at a given frequency actually sees
# ---------------------------------------------------------------------------

def resonant_density(f_Hz: float) -> float:
    """Electron density [m^-3] whose plasma frequency equals f_Hz."""
    w = 2.0 * np.pi * f_Hz
    return w**2 * EPS_0 * M_ELECTRON / E_CHARGE**2


def emission_at_frequency(exp: ExpansionResult, f_Hz: float,
                          r_sensor: float = 0.30,
                          theta_deg: float = 90.0,
                          separation_factor: float = 1.0,
                          closure: str = "debye",
                          bandwidth_fraction: float = 0.1) -> dict:
    """Field a narrowband antenna at `f_Hz` would see.

    Only the part of the plume with omega_pe ~ 2 pi f radiates into that band
    and can escape (a wave below the local plasma frequency is evanescent).
    This function locates the epoch at which the plume's peak electron density
    passes through the resonant value and evaluates the dipole there.

    This is the correct comparison with the Close et al. (2013) patch-antenna
    measurements at 315 MHz and 916 MHz.
    """
    n_res = resonant_density(f_Hz)
    n_e = exp.n_e
    if n_res > np.nanmax(n_e) or n_res < np.nanmin(n_e[n_e > 0]):
        return {"f": f_Hz, "n_resonant": n_res, "reached": False,
                "E": 0.0, "note": "resonant density never reached"}

    # n_e falls monotonically after the peak; find the crossing.
    k = int(np.argmin(np.abs(np.log(np.maximum(n_e, 1e-300))
                             - np.log(n_res))))
    Rr, Rz = exp.R_r[k], exp.R_z[k]
    area = 2.0 * np.pi * Rr * Rz
    L = np.sqrt(Rr * Rz)
    cs = separation_from_shell(n_e[k], exp.T_eV[k], area, exp.nu_ei[k], L,
                               separation_factor=separation_factor,
                               closure=closure, m_ion=exp.material.m_atom,
                               v_drift=float(np.hypot(exp.v_r[k], exp.v_z[k])))
    w0 = cs.omega
    fld = dipole_field(cs.p0, w0, r_sensor, theta_deg)

    # Fraction of the pulse energy that falls inside the antenna band.  The
    # emission is a damped oscillation of half-width gamma (rad/s), i.e. a
    # Lorentzian of FWHM gamma/pi in Hz; an antenna of bandwidth bw collects
    # the Lorentzian integral over that band.
    bw = bandwidth_fraction * f_Hz
    q_factor = w0 / (2.0 * cs.gamma) if cs.gamma > 0 else np.inf
    frac = float(min(1.0, (2.0 / np.pi)
                     * np.arctan(np.pi * bw / max(cs.gamma / np.pi, 1e-30))))
    return {
        "f": f_Hz, "n_resonant": n_res, "reached": True,
        "t": float(exp.t[k]), "index": k,
        "n_e": float(n_e[k]), "T_eV": float(exp.T_eV[k]),
        "R_r": float(Rr), "R_z": float(Rz),
        "lambda_De": cs.lambda_De, "Q_separated": cs.Q,
        "delta": cs.delta, "p0": cs.p0,
        "omega": w0, "gamma": cs.gamma, "Q_factor": q_factor,
        "E_peak": fld["E"], "E_far_field_only": fld["far_field_only_E"],
        "E_in_band": fld["E"] * np.sqrt(frac), "band_fraction": frac,
        "B_peak": fld["B"], "kr": fld["kr"], "far_field": fld["far_field"],
        "wavelength": C_LIGHT / f_Hz,
        "dipole_approx_valid": bool(L < 0.1 * C_LIGHT / f_Hz),
    }


# ---------------------------------------------------------------------------
# Scaling checks
# ---------------------------------------------------------------------------

def field_decay(E_ref: float, r_ref: float, r: float,
                exponent: float = 1.0) -> float:
    """Field at range r given a reference.

    A true far-field dipole falls as 1/r.  Fletcher & Close measure d^-0.73 in
    their 2D simulations (between the cylindrical d^-1/2 and spherical d^-1
    limits, as expected for a 2D geometry), and Fletcher (2021) quotes r^-1/2
    for the direction perpendicular to the plume axis.  `exponent` lets a
    measured decay law be substituted.
    """
    return E_ref * (r_ref / r) ** exponent


__all__ = ["CALIBRATED_XI", "CALIBRATION_SOURCE",
    "ChargeSeparation", "EMPResult", "separation_from_shell", "simulate_emp",
    "dipole_field", "resonant_density", "emission_at_frequency",
    "field_decay",
]
