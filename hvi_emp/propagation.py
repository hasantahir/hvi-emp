"""Propagation of the EMP out of the plume and through the debris cloud.

Two distinct questions:

1.  **Can the wave escape the plasma?**  A cold, unmagnetised plasma has
    refractive index ``N^2 = 1 - omega_pe^2/(omega(omega + i nu))``.  Below the
    plasma frequency N is imaginary and the wave is evanescent.  Since the
    plume density falls outward, radiation generated at the peak density
    escapes only if its frequency exceeds omega_pe everywhere along the path
    -- which is why an antenna at frequency f is effectively looking at the
    shell where omega_pe = 2 pi f.  Kodis (1965) is the reference treatment of
    propagation and scattering in exactly this configuration.

2.  **Does the ejecta dust attenuate it?**  The impact throws out a cloud of
    sub-micron condensate.  Tishkovets et al. (2011) give the framework for
    scattering by ensembles of small particles.  The size parameter
    ``x = 2 pi a / lambda`` decides everything: at optical wavelengths
    x ~ 1 and the cloud is opaque (this is the impact flash), while at RF
    wavelengths x ~ 1e-5 and Rayleigh scattering (~ x^4) is utterly
    negligible.  `dust_optical_depth` makes that quantitative, so the
    (frequently asserted) worry that the debris cloud shields the EMP can be
    checked rather than assumed.

References
----------
Kodis, R.D., "Propagation and scattering in plasmas" (1965).
Tishkovets, Petrova & Mishchenko, JQSRT 112, 2095 (2011).
Ginzburg, *Propagation of Electromagnetic Waves in Plasma*, Gordon & Breach
(1961), Ch. 4.
Bohren & Huffman, *Absorption and Scattering of Light by Small Particles*,
Wiley (1983), Ch. 5 (Rayleigh limit).
"""

from __future__ import annotations

import numpy as np

from .constants import C_LIGHT, EPS_0, E_CHARGE, M_ELECTRON
from .ionization import plasma_frequency


# ---------------------------------------------------------------------------
# Plasma propagation
# ---------------------------------------------------------------------------

def refractive_index(omega: np.ndarray | float, n_e: float,
                     nu: float = 0.0) -> np.ndarray | complex:
    """Complex refractive index of a cold, collisional, unmagnetised plasma.

        N^2 = 1 - omega_pe^2 / (omega (omega + i nu))
    """
    w = np.asarray(omega, dtype=complex)
    wpe = plasma_frequency(n_e)
    with np.errstate(divide="ignore", invalid="ignore"):
        N2 = 1.0 - wpe**2 / (w * (w + 1j * nu))
    return np.sqrt(N2)


def cutoff_frequency(n_e: float) -> float:
    """Ordinary-mode cutoff frequency [Hz] -- waves below this cannot escape."""
    return plasma_frequency(n_e) / (2.0 * np.pi)


def transmission(omega: np.ndarray | float, n_e_profile: np.ndarray,
                 dr: np.ndarray | float, nu_profile: np.ndarray | float = 0.0
                 ) -> np.ndarray:
    """Amplitude transmission through a layered plasma column.

    Uses the WKB (geometric-optics) attenuation
    ``exp(-integral Im(N) omega/c dr)``, valid when the density scale length is
    long compared with the wavelength.  Below cutoff this correctly produces
    exponential evanescence; it does *not* capture the sharp reflection
    resonance right at the cutoff layer, for which a full-wave solution is
    needed.
    """
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    ne = np.atleast_1d(np.asarray(n_e_profile, dtype=float))
    nu = np.broadcast_to(np.atleast_1d(np.asarray(nu_profile, dtype=float)),
                         ne.shape)
    dr = np.broadcast_to(np.atleast_1d(np.asarray(dr, dtype=float)), ne.shape)

    out = np.empty_like(w)
    for i, wi in enumerate(w):
        N = np.array([refractive_index(wi, n, v) for n, v in zip(ne, nu)])
        kappa = np.imag(N) * wi / C_LIGHT
        out[i] = float(np.exp(-np.sum(np.maximum(kappa, 0.0) * dr)))
    return out if out.size > 1 else out[0]


def escape_frequency(exp_result, index: int | None = None) -> float:
    """Minimum frequency [Hz] able to escape the plume at a given epoch.

    Equals the peak plasma frequency at that time.  Anything below it is
    trapped and either damped or re-absorbed.
    """
    i = -1 if index is None else index
    return float(exp_result.omega_pe[i] / (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Dust / ejecta cloud
# ---------------------------------------------------------------------------

def size_parameter(radius: float, wavelength: float) -> float:
    """x = 2 pi a / lambda."""
    return 2.0 * np.pi * radius / wavelength


def rayleigh_cross_section(radius: float, wavelength: float,
                           m: complex = 1.5 + 0.01j) -> float:
    """Rayleigh scattering cross-section [m^2] of a small sphere.

        C_sca = (8/3) pi k^4 a^6 |(m^2-1)/(m^2+2)|^2
    """
    k = 2.0 * np.pi / wavelength
    f = (m**2 - 1.0) / (m**2 + 2.0)
    return float((8.0 / 3.0) * np.pi * k**4 * radius**6 * abs(f) ** 2)


def rayleigh_absorption_cross_section(radius: float, wavelength: float,
                                      m: complex = 1.5 + 0.01j) -> float:
    """Rayleigh absorption cross-section [m^2]: C_abs = 4 pi k a^3 Im(f)."""
    k = 2.0 * np.pi / wavelength
    f = (m**2 - 1.0) / (m**2 + 2.0)
    return float(4.0 * np.pi * k * radius**3 * np.imag(f))


def dust_optical_depth(n_dust: float, radius: float, path: float,
                       wavelength: float, m: complex = 1.5 + 0.01j) -> dict:
    """Optical depth of an ejecta dust cloud, and whether it matters.

    Parameters
    ----------
    n_dust : float
        Dust number density [m^-3].
    radius : float
        Grain radius [m].
    path : float
        Path length through the cloud [m].
    wavelength : float
        Free-space wavelength [m].

    Notes
    -----
    Returns both the Rayleigh result and the geometric-optics limit, plus the
    size parameter, so the regime is explicit.  For a typical impact this
    shows tau ~ 1 in the visible (the impact flash is optically thick) and
    tau ~ 1e-20 at 1 GHz -- i.e. the debris cloud is completely transparent to
    the EMP, and any observed RF attenuation must come from the *plasma*, not
    the dust.
    """
    x = size_parameter(radius, wavelength)
    C_sca = rayleigh_cross_section(radius, wavelength, m)
    C_abs = rayleigh_absorption_cross_section(radius, wavelength, m)
    C_geo = np.pi * radius**2
    if x > 1.0:
        # Outside the Rayleigh regime, cap the extinction at ~2 x geometric
        # (the extinction-paradox limit) rather than extrapolating x^4.
        C_ext = min(C_sca + C_abs, 2.0 * C_geo)
        regime = "Mie / geometric"
    else:
        C_ext = C_sca + C_abs
        regime = "Rayleigh"
    tau = n_dust * C_ext * path
    return {
        "size_parameter": x, "regime": regime,
        "C_scattering": C_sca, "C_absorption": C_abs,
        "C_geometric": C_geo, "C_extinction": C_ext,
        "optical_depth": float(tau),
        "transmission": float(np.exp(-tau)),
        "significant": bool(tau > 0.01),
    }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def density_for_frequency(f_Hz: float) -> float:
    """Electron density [m^-3] with plasma frequency f_Hz (cutoff density)."""
    w = 2.0 * np.pi * f_Hz
    return w**2 * EPS_0 * M_ELECTRON / E_CHARGE**2


__all__ = [
    "refractive_index", "cutoff_frequency", "transmission",
    "escape_frequency", "size_parameter", "rayleigh_cross_section",
    "rayleigh_absorption_cross_section", "dust_optical_depth",
    "density_for_frequency",
]
