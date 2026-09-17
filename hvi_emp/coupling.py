"""Stage 4 -- coupling to the spacecraft: charging, ESD, and induced currents.

Three distinct damage pathways, all of which have been invoked for real
on-orbit anomalies:

1.  **Direct plasma charging.**  The plume is a conducting, quasi-neutral
    plasma in contact with spacecraft surfaces.  Electrons reach a surface
    faster than ions (their thermal speed is sqrt(m_i/m_e) larger), so an
    isolated surface charges negative until the fluxes balance at the floating
    potential

        phi_f = -(k T_e / e) ln sqrt(m_i / (2 pi m_e))

    This is the Langmuir sheath result used by Crawford (Procedia Eng. 103, 89,
    2015) inside CTH, and it is the mechanism behind the electrostatic charge
    separation measured at the AVGR.

2.  **Differential charging and ESD.**  Spacecraft exteriors mix grounded
    conductors with dielectrics (coverglass, Kapton MLI, thermal paint).  A
    transient plasma charges them differently; if the resulting differential
    potential exceeds the dielectric breakdown threshold a discharge is
    triggered.  This is precisely the Olympus-1 scenario proposed by Caswell,
    McBride & Taylor (Int. J. Impact Eng. 17, 139, 1995): a Perseid impact
    "may have generated a plasma triggering a discharge of charged surfaces
    entering the grounded spacecraft via the umbilical and an external
    sensor."

3.  **EMP coupling to cabling.**  The radiated pulse induces (i) a loop
    voltage V = -A dB/dt and (ii) an open-circuit voltage V = h_eff E on a
    wire acting as a short monopole.  Because the pulse is broadband, part of
    its energy lies above the shielding effectiveness roll-off of a practical
    Faraday enclosure -- Fletcher (2021): "some of the electromagnetic energy
    could slip past the standard protection".

Thresholds
----------
The upset/damage thresholds used here are representative published values,
not qualification data for any specific hardware.  They are collected in
`UPSET_THRESHOLDS` with sources so they can be replaced with the numbers for
the actual design under study.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import (C_LIGHT, E_CHARGE, EPS_0, EV, K_B, M_ELECTRON, MU_0)
from .emp import EMPResult
from .expansion import ExpansionResult
from .materials import Material


# ---------------------------------------------------------------------------
# Reference thresholds
# ---------------------------------------------------------------------------

UPSET_THRESHOLDS = {
    "esd_dielectric_V": (500.0,
                         "Differential potential at which surface ESD is "
                         "commonly assumed to trigger on spacecraft "
                         "dielectrics. NASA-HDBK-4002A, Sec. 5."),
    "esd_breakdown_field_V_per_m": (2.0e7,
                                    "Bulk dielectric strength of Kapton/"
                                    "coverglass, ~20 MV/m (manufacturer data)."),
    "cmos_upset_V": (0.5,
                     "Induced transient on a digital input sufficient to "
                     "cause a logic upset in 3.3 V CMOS (noise margin)."),
    "rf_frontend_damage_dBm": (10.0,
                               "Typical LNA input damage level, +10 dBm."),
}


# ---------------------------------------------------------------------------
# Surface charging
# ---------------------------------------------------------------------------

def floating_potential(T_e_eV: float, m_ion: float,
                       secondary_yield: float = 0.0) -> float:
    """Floating potential [V] of an isolated surface in a plasma.

    Balancing the ion flux against the *net* electron flux -- incident
    electrons minus the secondaries they knock back out -- gives

        (1 - delta) J_e0 exp(e phi / k T_e) = J_i0
        phi_f = -(k T_e/e) ln[ (1 - delta) sqrt(m_i / (2 pi m_e)) ]

    Secondary emission therefore makes the surface *less* negative: emitted
    secondaries carry negative charge away, partially cancelling the incoming
    electron current.  Once delta exceeds the value that makes the bracket
    unity the surface would float positive, and the potential is then set by
    the space-charge-limited emission problem rather than by this balance; we
    clamp at zero and flag that regime instead of extrapolating.
    """
    if T_e_eV <= 0 or m_ion <= 0:
        return 0.0
    d = min(max(secondary_yield, 0.0), 0.99)
    ratio = (1.0 - d) * np.sqrt(m_ion / (2.0 * np.pi * M_ELECTRON))
    return float(-T_e_eV * np.log(max(ratio, 1.0)))


def thermal_current_density(n_e: float, T_e_eV: float, m: float,
                            charge: float = -E_CHARGE,
                            potential: float = 0.0) -> float:
    """Random thermal current density [A/m^2] onto a biased surface.

    Electrons (repelled by a negative surface) are Boltzmann-suppressed:
        J_e = (1/4) e n_e v_th,e exp(e phi / k T_e)
    Ions are collected at the (unretarded) thermal flux:
        J_i = (1/4) e n_i v_th,i
    """
    if n_e <= 0 or T_e_eV <= 0:
        return 0.0
    v_th = np.sqrt(8.0 * T_e_eV * EV / (np.pi * m))
    J = 0.25 * abs(charge) * n_e * v_th
    if charge < 0 and potential < 0:
        J *= np.exp(potential / T_e_eV)     # electron retardation
    return float(J)


@dataclass
class ChargingResult:
    """Surface charging produced by the passing plume."""
    t: np.ndarray
    n_e_at_surface: np.ndarray     # m^-3
    T_eV: np.ndarray
    phi_float: np.ndarray          # V
    J_ion: np.ndarray              # A/m^2
    J_electron: np.ndarray         # A/m^2
    Q_surface: np.ndarray          # C/m^2, accumulated on a dielectric
    V_dielectric: np.ndarray       # V, across the dielectric
    peak_differential_V: float
    esd_risk: bool
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        thr = UPSET_THRESHOLDS["esd_dielectric_V"][0]
        lines = [
            "Surface charging:",
            f"  peak floating potential   {np.min(self.phi_float):10.2f} V",
            f"  peak deposited charge     "
            f"{np.max(np.abs(self.Q_surface)):10.3e} C/m^2",
            f"  peak differential V       {self.peak_differential_V:10.2f} V",
            f"  ESD threshold ({thr:.0f} V)      "
            f"{'EXCEEDED' if self.esd_risk else 'not exceeded'}",
        ]
        if self.diagnostics.get("breakdown_saturated"):
            v_bd = self.diagnostics.get("V_breakdown", 0.0)
            lines.append(
                f"  dielectric breakdown      REACHED -- potential clipped at "
                f"{v_bd:.0f} V; beyond this the dielectric discharges and the "
                "capacitor model no longer applies")
        else:
            lines.append("  dielectric breakdown      not reached")
        return "\n".join(lines)


def simulate_charging(exp: ExpansionResult,
                      standoff: float = 0.10,
                      dielectric_thickness: float = 1.27e-4,
                      dielectric_eps_r: float = 3.4,
                      secondary_yield: float = 0.3,
                      area: float = 1.0) -> ChargingResult:
    """Charging of a dielectric surface a distance `standoff` from the impact.

    The plume density at the surface is taken as the Gaussian profile
    evaluated at that standoff, so a surface outside the plume sees very
    little.  Charge accumulates by the *difference* of ion and electron
    fluxes; the dielectric acts as a capacitor of areal capacitance
    ``eps0 eps_r / d``, and the resulting voltage is compared with the ESD
    threshold.

    Parameters
    ----------
    standoff : float
        Distance from the impact point to the surface element [m].
    dielectric_thickness : float
        Coverglass / Kapton thickness [m].  Default 5 mil (127 um) Kapton.
    dielectric_eps_r : float
        Relative permittivity (3.4 for Kapton, ~4 for coverglass).
    secondary_yield : float
        Secondary-electron emission coefficient of the surface material.
    """
    t = exp.t
    # Gaussian plume profile evaluated at the standoff distance.
    sigma = np.sqrt(exp.R_r * exp.R_z)
    n_surf = exp.n_e * np.exp(-0.5 * (standoff / np.maximum(sigma, 1e-12))**2)
    T_eV = exp.T_eV
    m_ion = exp.material.m_atom

    phi = np.array([floating_potential(te, m_ion, secondary_yield)
                    for te in T_eV])
    J_i = np.array([thermal_current_density(n, te, m_ion, E_CHARGE, 0.0)
                    for n, te in zip(n_surf, T_eV)])
    J_e = np.array([thermal_current_density(n, te, M_ELECTRON, -E_CHARGE, p)
                    for n, te, p in zip(n_surf, T_eV, phi)])

    # Net deposited charge per unit area (ions positive, electrons negative).
    J_net = J_i - J_e
    Q = np.concatenate(([0.0], np.cumsum(0.5 * (J_net[1:] + J_net[:-1])
                                         * np.diff(t))))
    C_area = EPS_0 * dielectric_eps_r / dielectric_thickness   # F/m^2
    V_diel_raw = Q / C_area

    # A dielectric cannot hold more than its breakdown field: once
    # |V| / d exceeds E_bd the material breaks down and the stored charge is
    # released. The bare capacitor model has no such limit and, for a surface
    # engulfed by the plume, will happily report megavolts across 127 um of
    # Kapton. Clip at the breakdown voltage and flag it -- beyond this point
    # the quantity of interest is not "how many volts" but "it discharged".
    E_bd = UPSET_THRESHOLDS["esd_breakdown_field_V_per_m"][0]
    V_bd = E_bd * dielectric_thickness
    saturated = bool(np.max(np.abs(V_diel_raw)) > V_bd)
    V_diel = np.clip(V_diel_raw, -V_bd, V_bd)

    # Differential potential between the charged dielectric and the grounded
    # structure: the more negative of the floating potential (plasma-coupled
    # conductor) and the capacitively stored dielectric potential.
    V_diff = np.abs(V_diel - phi)
    peak = float(np.max(V_diff)) if V_diff.size else 0.0
    thr = UPSET_THRESHOLDS["esd_dielectric_V"][0]

    return ChargingResult(
        t=t, n_e_at_surface=n_surf, T_eV=T_eV, phi_float=phi,
        J_ion=J_i, J_electron=J_e, Q_surface=Q, V_dielectric=V_diel,
        peak_differential_V=peak, esd_risk=bool(peak > thr),
        diagnostics={
            "standoff": standoff, "area": area,
            "C_area": C_area, "J_net": J_net,
            "Q_total": Q * area,
            "V_dielectric_unclipped": V_diel_raw,
            "V_breakdown": V_bd,
            "breakdown_saturated": saturated,
            "breakdown_field": float(np.max(np.abs(V_diel))
                                     / dielectric_thickness),
            "breakdown_risk": saturated,
        },
    )


# ---------------------------------------------------------------------------
# EMP coupling to cabling
# ---------------------------------------------------------------------------

@dataclass
class CouplingResult:
    """Induced voltages from the radiated EMP."""
    V_loop_peak: float          # V
    V_monopole_peak: float      # V
    dBdt_peak: float            # T/s
    E_peak: float               # V/m
    power_density: float        # W/m^2
    received_power_dBm: float
    shielded_V: float           # V after enclosure attenuation
    upset_risk: bool
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        thr = UPSET_THRESHOLDS["cmos_upset_V"][0]
        return "\n".join([
            "EMP coupling:",
            f"  peak |E|                  {self.E_peak:10.3e} V/m",
            f"  peak dB/dt                {self.dBdt_peak:10.3e} T/s",
            f"  loop-induced voltage      {self.V_loop_peak:10.3e} V",
            f"  monopole voltage          {self.V_monopole_peak:10.3e} V",
            f"  after shielding           {self.shielded_V:10.3e} V",
            f"  received power            {self.received_power_dBm:10.1f} dBm",
            f"  CMOS upset ({thr:.1f} V)        "
            f"{'AT RISK' if self.upset_risk else 'below threshold'}",
        ])


def couple_to_structure(emp: EMPResult,
                        loop_area: float = 1.0e-2,
                        wire_length: float = 0.10,
                        shielding_dB: float = 40.0,
                        antenna_gain_dBi: float = 0.0) -> CouplingResult:
    """Induced voltages and received power from an EMP waveform.

    Parameters
    ----------
    emp : EMPResult
    loop_area : float
        Effective area of the worst-case circuit loop [m^2].  1e-2 m^2 is a
        10 cm x 10 cm harness loop.
    wire_length : float
        Length of an exposed conductor acting as a short monopole [m]; the
        effective height of a short monopole is h_eff = L/2.
    shielding_dB : float
        Enclosure shielding effectiveness [dB].  Applied flat here; real
        enclosures roll off with frequency and through apertures, which is the
        physical basis for the concern that a broadband pulse partly bypasses
        the shield.
    antenna_gain_dBi : float
        Gain of any receiving antenna, for the received-power estimate.
    """
    E = emp.E_t
    B = emp.B_t
    t = emp.t
    dBdt = np.gradient(B, t)

    V_loop = float(np.max(np.abs(dBdt)) * loop_area)
    V_mono = float(np.max(np.abs(E)) * wire_length / 2.0)
    E_pk = float(np.max(np.abs(E)))
    S = E_pk**2 / (2.0 * 376.730_313_412)          # W/m^2, peak Poynting
    lam = C_LIGHT / max(emp.f[np.argmax(emp.psd)], 1.0)
    A_eff = lam**2 / (4.0 * np.pi) * 10 ** (antenna_gain_dBi / 10.0)
    P = S * A_eff
    P_dBm = 10.0 * np.log10(max(P, 1e-30) * 1e3)

    atten = 10 ** (-shielding_dB / 20.0)
    V_sh = max(V_loop, V_mono) * atten

    return CouplingResult(
        V_loop_peak=V_loop, V_monopole_peak=V_mono,
        dBdt_peak=float(np.max(np.abs(dBdt))), E_peak=E_pk,
        power_density=float(S), received_power_dBm=float(P_dBm),
        shielded_V=float(V_sh),
        upset_risk=bool(V_sh > UPSET_THRESHOLDS["cmos_upset_V"][0]),
        diagnostics={
            "loop_area": loop_area, "wire_length": wire_length,
            "shielding_dB": shielding_dB, "A_eff": A_eff,
            "lambda_peak": lam, "dBdt": dBdt,
            "rf_frontend_damage": bool(
                P_dBm > UPSET_THRESHOLDS["rf_frontend_damage_dBm"][0]),
        },
    )


# ---------------------------------------------------------------------------
# Olympus-style anomaly assessment
# ---------------------------------------------------------------------------

def anomaly_assessment(charging: ChargingResult,
                       coupling: CouplingResult) -> dict:
    """Roll the two pathways up into a qualitative anomaly verdict.

    Deliberately conservative and explicit: it reports *which* pathway is
    implicated rather than a single opaque score, because the Olympus, Landsat
    5 and ADEOS-II investigations all turned on distinguishing an ESD-triggered
    upset from a directly radiated one.
    """
    paths = []
    if charging.esd_risk:
        paths.append("surface ESD (differential charging exceeds threshold)")
    if charging.diagnostics.get("breakdown_risk"):
        paths.append("dielectric bulk breakdown")
    if coupling.upset_risk:
        paths.append("radiated EMP coupling into cabling")
    if coupling.diagnostics.get("rf_frontend_damage"):
        paths.append("RF front-end overdrive")
    return {
        "pathways": paths,
        "any_risk": bool(paths),
        "peak_differential_V": charging.peak_differential_V,
        "shielded_induced_V": coupling.shielded_V,
        "verdict": ("credible anomaly mechanism identified"
                    if paths else
                    "no threshold exceeded for the assumed configuration"),
    }


__all__ = [
    "UPSET_THRESHOLDS", "ChargingResult", "CouplingResult",
    "floating_potential", "thermal_current_density", "simulate_charging",
    "couple_to_structure", "anomaly_assessment",
]
