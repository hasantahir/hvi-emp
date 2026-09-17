"""Radiative energy loss from the expanding plume.

Why this module exists, and what it concludes
---------------------------------------------
Radiative cooling was a listed gap in the Stage-2 energy equation and a
candidate explanation for the framework's one open disagreement with
experiment (the charge-yield velocity exponent).  It is implemented here so
the question can be answered with a number rather than an assumption.

**The answer is that it is negligible for impact plumes.**  For a 1 pg Fe
impact on Al at 50 km/s the radiative cooling time exceeds the expansion time
by a factor ~54 at plume formation, and the ratio grows by six more orders of
magnitude as the plume thins.  Adiabatic expansion wins everywhere.  Keeping
the term costs almost nothing and makes the model correct at parameters where
it *would* matter (denser, hotter or slower-expanding plasmas), so it is on by
default.

Processes included
------------------
**Bremsstrahlung (free-free).**  NRL Plasma Formulary:

    P_ff = 1.69e-32 n_e sqrt(T_e) sum_Z Z^2 n_Z     [W/cm^3, n in cm^-3, T in eV]

which in SI, with a single mean charge state, is

    Lambda_ff = 1.69e-38 Zbar^3 n_h^2 sqrt(T_eV)    [W/m^3]

**Radiative recombination (free-bound).**  Each capture radiates roughly the
binding energy plus the electron's kinetic energy:

    Lambda_fb = alpha_rad(T, Z) n_e n_i (chi + 3/2 k T_e)

using the same `ionization.alpha_radiative` rate the freeze-out calculation
uses, so the two cannot disagree.

**Escape probability.**  A plume that is optically thick does not radiate from
its interior.  The grey free-free (Kramers) opacity gives an optical depth
across the plume, and the escape factor 1/(1 + tau) interpolates between the
optically thin limit (radiate everything) and the thick limit (radiate only
from a surface layer).  This is the standard first-order treatment; it is not
a radiation-transport solution.

What is NOT included
--------------------
**Line radiation**, which for a partially-ionised *metal* vapour near 1 eV is
normally the dominant loss channel -- potentially 10-100x the continuum terms
here.  Implementing it needs level structure and oscillator strengths for
every ion stage of every material, which this framework does not carry.

The omission is one-sided: including line radiation could only *increase*
cooling.  So the numbers here are a **lower bound on radiative losses**, and
the conclusion "radiation loses to adiabatic expansion by ~50x" survives even
if the true loss is an order of magnitude larger than computed.  It would not
survive a factor of 100, which is the honest limit of this statement.

References
----------
NRL Plasma Formulary (2019), "Radiation" section.
Zel'dovich & Raizer, *Physics of Shock Waves...*, Ch. II and V.
Rybicki & Lightman, *Radiative Processes in Astrophysics*, Ch. 5 (free-free),
Ch. 10 (Kramers opacity).
"""

from __future__ import annotations

import numpy as np

from .constants import EV, K_B
from .ionization import alpha_radiative

#: NRL bremsstrahlung coefficient, converted to SI (n in m^-3, T in eV,
#: result in W/m^3) for a single mean charge state.
BREMSSTRAHLUNG_SI = 1.69e-38

#: Kramers free-free opacity coefficient [cm^2 g^-1 in cgs before scaling].
#: kappa = KRAMERS * (Z^2/A) * rho * T^-3.5, rho in g/cm^3, T in K.
KRAMERS_FF = 0.64e23


def bremsstrahlung(n_h: float, T_eV: float, Zbar: float) -> float:
    """Free-free radiated power density [W/m^3]."""
    if Zbar <= 0 or n_h <= 0 or T_eV <= 0:
        return 0.0
    return BREMSSTRAHLUNG_SI * Zbar**3 * n_h**2 * np.sqrt(T_eV)


def recombination_radiation(n_h: float, T_eV: float, Zbar: float,
                            chi_eV: float = 6.0) -> float:
    """Free-bound radiated power density [W/m^3].

    ``chi_eV`` is the effective binding energy released per capture; 6 eV is
    a reasonable mean first ionisation potential for the metals in the
    material library.  The result scales linearly with it.
    """
    if Zbar <= 0 or n_h <= 0 or T_eV <= 0:
        return 0.0
    n_e = Zbar * n_h
    alpha = alpha_radiative(T_eV, max(Zbar, 0.1))
    e_per_event = chi_eV * EV + 1.5 * T_eV * EV
    return alpha * n_e * n_h * e_per_event


def optical_depth(n_h: float, T_eV: float, Zbar: float, length: float,
                  A: float = 27.0) -> float:
    """Grey free-free (Kramers) optical depth across ``length`` metres.

    Used only to decide how much of the radiated power escapes; it is not a
    spectrum.
    """
    if n_h <= 0 or T_eV <= 0 or length <= 0:
        return 0.0
    T_K = max(T_eV * EV / K_B, 1.0)
    # mass density in g/cm^3 from heavy-particle density
    rho_cgs = n_h * A * 1.66053906660e-27 * 1e3 / 1e6
    kappa = KRAMERS_FF * (max(Zbar, 1e-3) ** 2 / A) * rho_cgs * T_K ** -3.5
    return float(kappa * rho_cgs * (length * 100.0))       # cm^2/g * g/cm^3 * cm


def escape_factor(tau: float) -> float:
    """Fraction of radiated power that leaves the plume.

    1/(1 + tau): unity when thin, ~1/tau when thick. Standard first-order
    escape probability, not a transport solution.
    """
    return 1.0 / (1.0 + max(tau, 0.0))


def cooling_rate(n_h: float, T_eV: float, Zbar: float,
                 length: float = 0.0, A: float = 27.0,
                 chi_eV: float = 6.0, thin: bool = False) -> dict:
    """Total radiative loss [W/m^3] and its breakdown.

    Parameters
    ----------
    length : float
        Characteristic plume size for the optical-depth estimate. Zero (or
        ``thin=True``) forces the optically-thin limit.

    Returns
    -------
    dict with ``total``, ``brems``, ``recomb``, ``tau``, ``escape``.
    """
    ff = bremsstrahlung(n_h, T_eV, Zbar)
    fb = recombination_radiation(n_h, T_eV, Zbar, chi_eV)
    tau = 0.0 if thin else optical_depth(n_h, T_eV, Zbar, length, A)
    f = escape_factor(tau)
    return {"total": (ff + fb) * f, "brems": ff, "recomb": fb,
            "tau": tau, "escape": f}


def cooling_time(n_h: float, T_eV: float, Zbar: float, **kw) -> float:
    """Internal energy divided by the radiative loss rate [s].

    Compare against the expansion time R/v: this module exists because that
    comparison comes out at ~50 in favour of expansion, and it is better to
    be able to show that than to assert it.
    """
    L = cooling_rate(n_h, T_eV, Zbar, **kw)["total"]
    if L <= 0:
        return np.inf
    u_density = 1.5 * K_B * (T_eV * EV / K_B) * (1.0 + Zbar) * n_h
    return float(u_density / L)


__all__ = ["bremsstrahlung", "recombination_radiation", "optical_depth",
           "escape_factor", "cooling_rate", "cooling_time",
           "BREMSSTRAHLUNG_SI", "KRAMERS_FF"]
