"""Physical constants (SI, CODATA 2018) used throughout HVI-EMP.

Everything in this package is SI internally. Convenience conversions are
provided at the bottom; user-facing helpers accept km/s, eV, etc.
"""

from __future__ import annotations

import math

# --- fundamental ---------------------------------------------------------
C_LIGHT = 2.997_924_58e8         # m/s          exact
MU_0 = 1.256_637_062_12e-6       # H/m
EPS_0 = 1.0 / (MU_0 * C_LIGHT**2)  # F/m
H_PLANCK = 6.626_070_15e-34      # J s          exact
HBAR = H_PLANCK / (2.0 * math.pi)
K_B = 1.380_649e-23              # J/K          exact
E_CHARGE = 1.602_176_634e-19     # C            exact
M_ELECTRON = 9.109_383_7015e-31  # kg
M_PROTON = 1.672_621_923_69e-27  # kg
AMU = 1.660_539_066_60e-27       # kg
N_AVOGADRO = 6.022_140_76e23     # 1/mol        exact
R_GAS = N_AVOGADRO * K_B         # J/(mol K)
SIGMA_SB = 5.670_374_419e-8      # W/(m^2 K^4)
A_BOHR = 5.291_772_109_03e-11    # m
SIGMA_THOMSON = 6.652_458_7321e-29  # m^2

# --- unit conversions ----------------------------------------------------
EV = E_CHARGE                    # 1 eV in J
EV_PER_K = K_B / E_CHARGE        # T[eV] = T[K] * EV_PER_K
K_PER_EV = E_CHARGE / K_B        # 11604.518 K per eV
GPA = 1.0e9
KM_S = 1.0e3

# Thermal de Broglie prefactor for the Saha equation:
#   (2 pi m_e k T / h^2)^{3/2}
SAHA_PREFACTOR = (2.0 * math.pi * M_ELECTRON * K_B / H_PLANCK**2) ** 1.5


def ev_to_kelvin(T_eV: float) -> float:
    """Convert temperature in eV to kelvin."""
    return T_eV * K_PER_EV


def kelvin_to_ev(T_K: float) -> float:
    """Convert temperature in kelvin to eV."""
    return T_K * EV_PER_K


__all__ = [
    "C_LIGHT", "MU_0", "EPS_0", "H_PLANCK", "HBAR", "K_B", "E_CHARGE",
    "M_ELECTRON", "M_PROTON", "AMU", "N_AVOGADRO", "R_GAS", "SIGMA_SB",
    "A_BOHR", "SIGMA_THOMSON", "EV", "EV_PER_K", "K_PER_EV", "GPA", "KM_S",
    "SAHA_PREFACTOR", "ev_to_kelvin", "kelvin_to_ev",
]
