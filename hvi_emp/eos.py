"""Shock equation of state: Hugoniot, impedance matching, release, waste heat.

Physics
-------
**Jump conditions.**  Across a steady shock into material at rest (up0 = 0,
P0 ~ 0):

    rho0 Us = rho (Us - up)                (mass)
    P       = rho0 Us up                   (momentum)
    E - E0  = 1/2 P (V0 - V) = 1/2 up^2    (energy)

closed by the empirical linear relation Us = c0 + s up.

**Off-Hugoniot states.**  Mie-Grueneisen referenced to the *cold (0 K) curve*:

    P(V, E) = P_c(V) + (Gamma(V)/V) (E - E_c(V)),    Gamma(V) rho = Gamma0 rho0

The cold curve is the Vinet ("universal binding energy") form calibrated
directly from the shock parameters,

    K0 = rho0 c0^2,     K0' = 4 s - 1,

which has the crucial property that it is valid in *expansion* as well as
compression and saturates at a finite cohesive energy
E_c(inf) = 4 K0 V0 / (K0' - 1)^2.  For aluminium this predicts 1.03e7 J/kg
against a measured sublimation energy of 1.21e7 J/kg -- a 15% check on the
whole construction with no free parameters.  A Hugoniot-referenced
Mie-Grueneisen (the more common textbook choice) is *unusable* here: it puts
P_c = 0 for V > V0 and therefore predicts unbounded adiabatic cooling.

**Waste heat.**  Because Gamma/V = Gamma0/V0 is constant for the assumed
Gamma rho = const scaling, the thermal energy DE = E - E_c obeys, along an
isentrope,

    d(DE)/dV = -P + P_c = -(Gamma0/V0) DE      =>
    DE(V) = DE(V_H) exp[-Gamma0 (V - V_H)/V0]

an exact closed form.  The release is terminated at

    V_end = min(V_zero_pressure, V_spinodal)

where V_spinodal = argmin P_c(V).  Physically: if the release adiabat can
reach P = 0 while still on the condensed branch, it does so and the leftover
DE is the classical waste heat; if the thermal pressure is too large to be
cancelled anywhere on the condensed branch, the material passes the
mechanical stability limit, fragments into vapour, and the remaining thermal
energy is handed to the plume-expansion stage rather than being converted to
further PdV work inside the target.

Validation of the resulting thresholds against Ahrens & O'Keefe is in
`validation.py`; agreement is within ~35-50% on shock pressure, which is the
honest accuracy of a linear-Us-Up + Vinet + constant-Gamma-rho model.

References
----------
Zel'dovich & Raizer, *Physics of Shock Waves...*, Dover (2002), Chs. I, XI.
Melosh, *Impact Cratering: A Geologic Process*, OUP (1989), Chs. 4-5.
Vinet, Ferrante, Rose & Smith, J. Geophys. Res. 92, 9319 (1987).
Ahrens & O'Keefe, The Moon 4, 214 (1972).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .materials import Material


# ---------------------------------------------------------------------------
# Vinet cold curve
# ---------------------------------------------------------------------------

def vinet_params(mat: Material) -> tuple[float, float, float]:
    """(K0 [Pa], K0' [-], eta = 1.5 (K0'-1)) from the linear Us-Up fit."""
    K0 = mat.rho0 * mat.c0**2
    K0p = 4.0 * mat.s - 1.0
    return K0, K0p, 1.5 * (K0p - 1.0)


def cold_pressure(mat: Material, V: float) -> float:
    """Vinet cold-curve pressure [Pa] at specific volume V [m^3/kg]."""
    if V <= 0.0:
        raise ValueError(
            f"{mat.name}: cold curve evaluated at non-positive specific "
            f"volume V = {V:.3e}. This means the Hugoniot has been pushed "
            "past its compression limit -- see "
            "`eos.compression_limit_velocity`.")
    K0, K0p, eta = vinet_params(mat)
    x = (V / mat.V0) ** (1.0 / 3.0)
    return 3.0 * K0 * x**-2 * (1.0 - x) * np.exp(eta * (1.0 - x))


def cold_energy(mat: Material, V: float) -> float:
    """Vinet cold-curve specific internal energy [J/kg], E_c(V0) = 0."""
    if V <= 0.0:
        raise ValueError(
            f"{mat.name}: cold curve evaluated at non-positive specific "
            f"volume V = {V:.3e}. See `eos.compression_limit_velocity`.")
    K0, K0p, eta = vinet_params(mat)
    x = (V / mat.V0) ** (1.0 / 3.0)
    pref = 4.0 * K0 * mat.V0 / (K0p - 1.0) ** 2
    return pref * (1.0 - (1.0 - eta * (1.0 - x)) * np.exp(eta * (1.0 - x)))


def cohesive_energy_vinet(mat: Material) -> float:
    """Asymptotic Vinet binding energy [J/kg] -- a no-free-parameter check
    against the tabulated sublimation energy."""
    K0, K0p, _ = vinet_params(mat)
    return 4.0 * K0 * mat.V0 / (K0p - 1.0) ** 2


@lru_cache(maxsize=64)
def spinodal_volume(mat: Material) -> float:
    """Specific volume [m^3/kg] at the cold-curve tensile minimum.

    Beyond this the condensed phase is mechanically unstable: the material
    cannot support further isentropic expansion as a single condensed phase.
    """
    res = minimize_scalar(lambda V: cold_pressure(mat, V),
                          bounds=(mat.V0 * 1.001, mat.V0 * 6.0),
                          method="bounded",
                          options={"xatol": mat.V0 * 1e-6})
    return float(res.x)


# ---------------------------------------------------------------------------
# Hugoniot
# ---------------------------------------------------------------------------

def compression_limit_velocity(mat: Material) -> float:
    """Particle velocity [m/s] at which the linear Us-Up fit becomes unphysical.

    The compression ratio implied by the jump conditions is
    ``V/V0 = 1 - up/Us`` with ``Us = c0 + s up``.  When ``up = Us`` this gives
    V = 0 -- infinite density -- which happens at

        up_limit = c0 / (1 - s)      for s < 1

    and never for ``s >= 1``.  Most metals have s ~ 1.2-1.6 and are safe, but a
    few (titanium, s = 0.767; beryllium; some alloys) have s < 1 and their
    linear fits break down at accessible impact speeds.  Titanium reaches the
    limit at 22.4 km/s particle velocity, i.e. around 45 km/s impact speed onto
    a light target.

    This is a failure of the *linear fit*, not of the physics: real Hugoniots
    curve and never reach infinite compression.  Above this velocity a tabular
    EOS (SESAME/ANEOS) is mandatory, not merely advisable.

    Returns
    -------
    float
        The limiting particle velocity, or ``inf`` if ``s >= 1``.
    """
    if mat.s >= 1.0:
        return np.inf
    return mat.c0 / (1.0 - mat.s)


def hugoniot_state(mat: Material, up: float, warn: bool = True) -> dict:
    """Full shocked state for particle velocity `up` [m/s].

    Returns dict with ``up, Us, P, rho, V, E`` in SI (E = specific internal
    energy jump = up^2/2).
    """
    if up <= 0:
        return {"up": 0.0, "Us": mat.c0, "P": 0.0, "rho": mat.rho0,
                "V": mat.V0, "E": 0.0}
    Us = mat.hugoniot_Us(up, warn=warn)
    if Us <= up * (1.0 + 1e-9):
        lim = compression_limit_velocity(mat)
        raise ValueError(
            f"{mat.name}: particle velocity {up/1e3:.2f} km/s reaches or "
            f"exceeds the compression limit of the linear Us-Up fit "
            f"({lim/1e3:.2f} km/s, since s = {mat.s:.3f} < 1). The fit implies "
            "infinite density, which is unphysical. Use a tabular EOS "
            "(SESAME/ANEOS) for this material at this speed.")
    P = mat.rho0 * Us * up
    rho = mat.rho0 * Us / (Us - up)
    return {"up": up, "Us": Us, "P": P, "rho": rho, "V": 1.0 / rho,
            "E": 0.5 * up * up}


def particle_velocity_from_pressure(mat: Material, P: float) -> float:
    """Invert P = rho0 (c0 + s up) up for up [m/s]."""
    if P <= 0:
        return 0.0
    a, b = mat.rho0 * mat.s, mat.rho0 * mat.c0
    return float((-b + np.sqrt(b * b + 4.0 * a * P)) / (2.0 * a))


def hugoniot_P_of_V(mat: Material, V: float) -> float:
    """Hugoniot pressure [Pa] at specific volume V (0 in expansion)."""
    eta = 1.0 - V / mat.V0
    if eta <= 0.0:
        return 0.0
    denom = 1.0 - mat.s * eta
    if denom <= 0.0:
        return np.inf
    up = mat.c0 * eta / denom
    return mat.rho0 * (mat.c0 + mat.s * up) * up


# ---------------------------------------------------------------------------
# Impedance matching (planar impact approximation)
# ---------------------------------------------------------------------------

@dataclass
class ImpactState:
    """Result of impedance matching a projectile onto a target."""
    v_impact: float
    up_target: float
    up_projectile: float
    P: float
    target: dict
    projectile: dict

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ImpactState(v={self.v_impact/1e3:.2f} km/s, "
                f"P={self.P/1e9:.1f} GPa, up_t={self.up_target/1e3:.2f} km/s)")


def impedance_match(projectile: Material, target: Material,
                    v_impact: float, warn: bool = True) -> ImpactState:
    """Planar impact approximation for a normal impact.

    Continuity of pressure and velocity at the interface gives a quadratic in
    the target particle velocity u:

        (rho0t st - rho0p sp) u^2
        + (rho0t c0t + 2 rho0p sp v + rho0p c0p) u
        - (rho0p sp v^2 + rho0p c0p v) = 0
    """
    if v_impact <= 0:
        raise ValueError("v_impact must be positive")
    rp, rt = projectile.rho0, target.rho0
    cp, ct = projectile.c0, target.c0
    sp, st = projectile.s, target.s
    v = v_impact

    a = rt * st - rp * sp
    b = rt * ct + 2.0 * rp * sp * v + rp * cp
    c = -(rp * sp * v * v + rp * cp * v)

    if abs(a) < 1e-12 * max(abs(b), 1.0):
        u = -c / b
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0:
            raise RuntimeError("Impedance match has no real solution.")
        roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
        phys = [r for r in roots if 0.0 < r < v]
        if not phys:
            raise RuntimeError(f"No physical root in (0, v); got {roots}.")
        u = min(phys)

    tgt = hugoniot_state(target, u, warn=warn)
    prj = hugoniot_state(projectile, v - u, warn=warn)
    P = 0.5 * (tgt["P"] + prj["P"])
    return ImpactState(v, u, v - u, P, tgt, prj)


# ---------------------------------------------------------------------------
# Release and waste heat
# ---------------------------------------------------------------------------

def release_state(mat: Material, up: float, warn: bool = True) -> dict:
    """Isentropic release from the Hugoniot state at `up`.

    Returns
    -------
    dict with
        ``E_thermal_shock`` : DE at the shocked volume [J/kg]
        ``E_residual``      : DE at the release endpoint [J/kg] (waste heat)
        ``V_end``           : release endpoint specific volume [m^3/kg]
        ``vaporised``       : True if the adiabat passed the spinodal
        ``P_shock``, ``V_shock``, ``E_shock``
    """
    st = hugoniot_state(mat, up, warn=warn)
    V_H, E_H = st["V"], st["E"]
    dE_H = E_H - cold_energy(mat, V_H)
    g0V0 = mat.gamma0 / mat.V0

    def dE(V):
        return dE_H * np.exp(-g0V0 * (V - V_H))

    def P_of_V(V):
        return cold_pressure(mat, V) + g0V0 * dE(V)

    V_sp = spinodal_volume(mat)

    # Does the adiabat cross P = 0 while still on the condensed branch?
    V_end, vaporised = V_sp, True
    if V_H < mat.V0:
        try:
            if P_of_V(mat.V0) <= 0.0:
                V_end = brentq(P_of_V, V_H, mat.V0, xtol=mat.V0 * 1e-9)
                vaporised = False
            elif P_of_V(V_sp) <= 0.0:
                V_end = brentq(P_of_V, mat.V0, V_sp, xtol=mat.V0 * 1e-9)
                vaporised = False
        except ValueError:
            pass

    return {
        "E_thermal_shock": dE_H,
        "E_residual": max(dE(V_end), 0.0),
        "V_end": V_end,
        "V_spinodal": V_sp,
        "vaporised": vaporised,
        "P_shock": st["P"],
        "V_shock": V_H,
        "E_shock": E_H,
        "up": up,
    }


def residual_energy(mat: Material, up: float, warn: bool = True) -> float:
    """Waste heat [J/kg] left in the material after shock and release."""
    return release_state(mat, up, warn=warn)["E_residual"]


def residual_energy_from_pressure(mat: Material, P: float) -> float:
    """Waste heat [J/kg] for a given peak shock pressure [Pa]."""
    return residual_energy(mat, particle_velocity_from_pressure(mat, P),
                           warn=False)


def max_admissible_pressure(mat: Material) -> float:
    """Highest shock pressure [Pa] the linear Us-Up fit can represent.

    Finite only for ``s < 1`` materials, where the fit reaches infinite
    compression at ``up = c0/(1-s)`` (see `compression_limit_velocity`).
    """
    lim = compression_limit_velocity(mat)
    if not np.isfinite(lim):
        return np.inf
    up = 0.999 * lim
    return mat.rho0 * (mat.c0 + mat.s * up) * up


def pressure_for_residual_energy(mat: Material, E_target: float,
                                 P_lo: float = 1e6, P_hi: float = 5e13
                                 ) -> float:
    """Shock pressure [Pa] whose release leaves waste heat `E_target`.

    Returns ``inf`` if the target energy is not reachable within the validity
    of the linear Us-Up fit -- which for ``s < 1`` materials can happen well
    below the nominal search ceiling, because the fit runs into its
    compression limit first.
    """
    if E_target <= 0:
        return 0.0

    P_hi = min(P_hi, max_admissible_pressure(mat))
    if P_hi <= P_lo:
        return np.inf

    def f(P):
        return residual_energy_from_pressure(mat, P) - E_target

    if f(P_lo) > 0:
        return P_lo
    if f(P_hi) < 0:
        return np.inf
    return float(brentq(f, P_lo, P_hi, rtol=1e-8, maxiter=300))


# ---------------------------------------------------------------------------
# Phase-change energy budget
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def phase_thresholds(mat: Material) -> dict:
    """Specific-energy thresholds [J/kg] and the shock pressures reaching them.

    ``E_melt``  : cold solid -> fully molten at Tm
    ``E_boil``  : -> liquid at Tv (incipient vaporisation on release)
    ``E_vap``   : -> fully vaporised (complete vaporisation)
    ``E_subl``  : cohesive energy per unit mass (free-atom reference)
    plus the corresponding ``P_*`` shock pressures.
    """
    E_melt = mat.cv_solid * (mat.T_melt - 298.0) + mat.L_fusion
    E_boil = E_melt + mat.cv_solid * (mat.T_vap - mat.T_melt)
    E_vap = E_boil + mat.L_vap
    return {
        "E_melt": E_melt,
        "E_boil": E_boil,
        "E_vap": E_vap,
        "E_subl": mat.E_sublimation,
        "P_melt": pressure_for_residual_energy(mat, E_melt),
        "P_boil": pressure_for_residual_energy(mat, E_boil),
        "P_vap": pressure_for_residual_energy(mat, E_vap),
        "P_subl": pressure_for_residual_energy(mat, mat.E_sublimation),
    }


def vapour_fraction(mat: Material, E_residual: float) -> float:
    """Mass fraction vaporised on release, from the waste heat [0, 1].

    Lever rule in the two-phase region: below E_boil nothing vaporises, above
    E_vap everything does, and in between the vapour fraction is the excess
    energy divided by the latent heat.
    """
    th = phase_thresholds(mat)
    if E_residual <= th["E_boil"]:
        return 0.0
    if E_residual >= th["E_vap"]:
        return 1.0
    return float((E_residual - th["E_boil"]) / mat.L_vap)


def melt_fraction(mat: Material, E_residual: float) -> float:
    """Mass fraction melted on release [0, 1] (0 below solidus)."""
    th = phase_thresholds(mat)
    E_solidus = mat.cv_solid * (mat.T_melt - 298.0)
    if E_residual <= E_solidus:
        return 0.0
    if E_residual >= th["E_melt"]:
        return 1.0
    return float((E_residual - E_solidus) / mat.L_fusion)


__all__ = [
    "ImpactState", "vinet_params", "cold_pressure", "cold_energy",
    "compression_limit_velocity", "max_admissible_pressure",
    "cohesive_energy_vinet", "spinodal_volume", "hugoniot_state",
    "hugoniot_P_of_V", "particle_velocity_from_pressure", "impedance_match",
    "release_state", "residual_energy", "residual_energy_from_pressure",
    "pressure_for_residual_energy", "phase_thresholds", "vapour_fraction",
    "melt_fraction",
]
