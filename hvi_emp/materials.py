"""Material database: shock EOS parameters, thermophysics and atomic data.

Sources
-------
Us-Up Hugoniot parameters (c0, s) and Gruneisen Gamma_0:
    Marsh, S.P. (ed.), *LASL Shock Hugoniot Data*, UC Press (1980).
    Steinberg, D.J., *Equation of State and Strength Properties of Selected
    Materials*, LLNL UCRL-MA-106439 (1996).
Thermophysical data (Tm, Tv, latent heats, cp):
    CRC Handbook of Chemistry and Physics, 97th ed.
First ionisation potentials and statistical weights:
    NIST Atomic Spectra Database (levels), ground-term degeneracies.
Work functions:
    CRC Handbook; Michaelson, J. Appl. Phys. 48, 4729 (1977).

The linear Us = c0 + s*up fit is used only up to the validity limit noted in
`us_up_valid_to`; beyond that a warning is raised because real materials
stiffen (electronic contributions, shell ionisation) and a tabular EOS
(SESAME / ANEOS) should be substituted.  See `docs/THEORY.md` Sec. 2.1.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from dataclasses import dataclass, field

from .constants import AMU, EV


@dataclass(frozen=True)
class Material:
    """Shock-EOS + thermophysical + atomic description of one material.

    Attributes
    ----------
    name : str
    rho0 : float
        Ambient density [kg/m^3].
    c0 : float
        Bulk sound speed, intercept of the linear Us-Up fit [m/s].
    s : float
        Slope of the linear Us-Up fit [-].
    gamma0 : float
        Gruneisen parameter at rho0 [-].  Gamma(V) = gamma0 * V/V0 assumed.
    cv_solid : float
        Solid specific heat at constant volume [J/(kg K)] (Dulong-Petit-ish).
    T_melt, T_vap : float
        Melt and 1-atm boiling temperature [K].
    L_fusion, L_vap : float
        Latent heat of fusion / vaporisation [J/kg].
    E_cohesive : float
        Cohesive (sublimation) energy per atom [J].  Used as the energy floor
        for complete vaporisation, more appropriate than L_vap alone.
    A : float
        Mean atomic mass [amu].
    Z : int
        Atomic number.
    E_ion : tuple[float, ...]
        Successive ionisation potentials [J], I, II, III ...
    g_ion : tuple[float, ...]
        Ground-state statistical weights g_0, g_1, g_2, ... [-].
    work_function : float
        Surface work function [J] (used in the secondary-emission / surface
        charging model).
    us_up_valid_to : float
        Particle velocity [m/s] beyond which the linear fit is extrapolated.
    """

    name: str
    rho0: float
    c0: float
    s: float
    gamma0: float
    cv_solid: float
    T_melt: float
    T_vap: float
    L_fusion: float
    L_vap: float
    E_cohesive: float
    A: float
    Z: int
    E_ion: tuple
    g_ion: tuple
    work_function: float
    us_up_valid_to: float = 1.5e4
    #: (name_a, name_b, mass fraction of a) when this Material was built by
    #: `mix_materials`; None for a pure element. The ionisation layer uses it
    #: to interpolate tables in composition instead of quantising it.
    mix_of: tuple | None = None

    # -- derived ----------------------------------------------------------
    @property
    def m_atom(self) -> float:
        """Mean atomic mass [kg]."""
        return self.A * AMU

    @property
    def V0(self) -> float:
        """Ambient specific volume [m^3/kg]."""
        return 1.0 / self.rho0

    @property
    def E_sublimation(self) -> float:
        """Specific energy to take cold solid -> free neutral atoms [J/kg]."""
        return self.E_cohesive / self.m_atom

    def hugoniot_Us(self, up: float, warn: bool = True) -> float:
        """Shock velocity from the linear Us-Up fit [m/s]."""
        if warn and up > self.us_up_valid_to:
            warnings.warn(
                f"{self.name}: particle velocity {up/1e3:.1f} km/s exceeds the "
                f"validated linear Us-Up range ({self.us_up_valid_to/1e3:.1f} "
                "km/s). Consider a tabular EOS (SESAME/ANEOS).",
                RuntimeWarning,
                stacklevel=2,
            )
        return self.c0 + self.s * up

    def gruneisen(self, V: float) -> float:
        """Gruneisen parameter at specific volume V, Gamma*rho = const."""
        return self.gamma0 * V / self.V0


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

_LIB: dict[str, Material] = {}

#: Directory holding one YAML file per material. Overridable with
#: ``$HVI_EMP_MATERIALS`` so a project can carry its own library without
#: editing the package, which is what a cluster user actually needs.
MATERIALS_DIR = Path(os.environ.get(
    "HVI_EMP_MATERIALS",
    Path(__file__).resolve().parent / "conf" / "materials"))


def _add(m: Material) -> Material:
    _LIB[m.name.lower()] = m
    return m


def material_from_spec(spec) -> Material:
    """Build a `Material` from a validated `schema.MaterialSpec`."""
    d = spec.model_dump()
    # `bonding` classifies the solid for schema cross-checks (see
    # schema._check_material_fields); it is not a physical parameter of the
    # EOS, so it does not become a Material field.
    d.pop("bonding", None)
    return Material(**{k: (tuple(v) if isinstance(v, (list, tuple)) else v)
                       for k, v in d.items()})


def load_material_file(path) -> Material:
    """Read, validate and construct one material from a YAML file.

    Validation is not optional here. A material is ~17 numbers that no
    reviewer can eyeball, several of which (the ionisation ladder, the
    melt/boil ordering) break the physics silently rather than loudly when
    wrong. `schema.validate_material` states the reason with the failure.
    """
    import yaml

    from .schema import ConfigError, validate_material

    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except Exception as exc:                                # noqa: BLE001
        raise ConfigError(f"{path}: not readable as YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping of fields, got "
                          f"{type(raw).__name__}")
    raw.setdefault("name", path.stem)
    try:
        return material_from_spec(validate_material(raw))
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def load_material_library(directory=None, replace: bool = False) -> dict:
    """Load every ``*.yaml`` in `directory` into the library.

    Parameters
    ----------
    replace : bool
        Clear the existing library first. Default is to merge, so a project
        directory can add materials without losing the built-ins.

    Returns the names loaded, in sorted order.
    """
    directory = Path(directory) if directory is not None else MATERIALS_DIR
    if not directory.is_dir():
        raise FileNotFoundError(
            f"material directory {directory} does not exist. Set "
            f"$HVI_EMP_MATERIALS or pass an explicit path.")
    if replace:
        _LIB.clear()
    loaded = []
    for path in sorted(directory.glob("*.yaml")) + \
            sorted(directory.glob("*.yml")):
        _add(load_material_file(path))
        loaded.append(path.stem)
    if not loaded:
        raise FileNotFoundError(f"no *.yaml material files in {directory}")
    return sorted(loaded)


# The library is the YAML tree: one file per material, validated on load.
# It used to be eight `Material(...)` literals in this module, which meant
# adding a material required editing library code, and a typo'd number was
# indistinguishable from a measured one. The literals were the single source
# of truth; now the files are, and these module-level names are bound from
# them so existing imports keep working.
load_material_library()

ALUMINIUM = _LIB["al"]
IRON = _LIB["fe"]
TUNGSTEN = _LIB["w"]
COPPER = _LIB["cu"]
GLASS = _LIB["sio2"]
KAPTON = _LIB["kapton"]
OLIVINE = _LIB["olivine"]
DOLOMITE = _LIB["dolomite"]


def mix_materials(mat_a: Material, w_a: float,
                  mat_b: Material, w_b: float,
                  round_to: int | None = None) -> Material:
    """Number-weighted synthetic material for a two-component vapour.

    A hypervelocity impact plume is a mixture of projectile and target vapour,
    and the mixture ratio varies continuously with impact speed.  Simply
    picking whichever component happens to dominate by mass produces an
    unphysical *discontinuity* in the predicted plume temperature and charge
    state at the crossover, because the mean atomic mass jumps.  This function
    instead builds a synthetic single-component material whose mean atomic
    mass is the true mixture mean

        1/m_bar = x_a/m_a + x_b/m_b        (x = mass fractions)

    and whose ionisation potentials, degeneracies and thermophysical
    properties are weighted by *number* fraction.

    Limitation: a genuine two-species Saha solve would track each element's
    ionisation ladder separately and let them share one electron reservoir.
    The single effective ladder used here is accurate when the two first
    ionisation potentials are within a couple of eV (true for most
    metal-on-metal pairs) and degrades when they are not; the composition is
    reported in the material name so the approximation is visible.

    Parameters
    ----------
    mat_a, mat_b : Material
    w_a, w_b : float
        Mass fractions (need not be normalised).
    round_to : int or None
        Quantise the composition to this many decimal places.  **Default None
        = exact**, which is what physics requires.

        An earlier version defaulted to 1 (10% granularity) to keep the number
        of cached ionisation tables small.  That was a mistake: it put visible
        steps in the mean atomic mass, and through it a *non-monotonic* plume
        temperature and charge state as a function of impact speed -- a
        sawtooth in exactly the velocity range this framework is used for.
        Composition is now exact and the table cost is handled where it
        belongs, by interpolating tables in composition
        (`ionization.MixtureTable`).
    """
    tot = w_a + w_b
    if tot <= 0:
        return mat_a
    w_a, w_b = w_a / tot, w_b / tot
    if w_a >= 1.0 - 1e-12:
        return mat_a
    if w_b >= 1.0 - 1e-12:
        return mat_b

    # number fractions
    na, nb = w_a / mat_a.m_atom, w_b / mat_b.m_atom
    xa, xb = na / (na + nb), nb / (na + nb)
    m_bar = 1.0 / (w_a / mat_a.m_atom + w_b / mat_b.m_atom)

    def nw(attr):
        return xa * getattr(mat_a, attr) + xb * getattr(mat_b, attr)

    n_stage = min(len(mat_a.E_ion), len(mat_b.E_ion))
    E_ion = tuple(xa * mat_a.E_ion[j] + xb * mat_b.E_ion[j]
                  for j in range(n_stage))
    g_ion = tuple(xa * mat_a.g_ion[j] + xb * mat_b.g_ion[j]
                  for j in range(n_stage + 1))

    if round_to is not None:
        wa_r = round(w_a, round_to)
        w_a, w_b = wa_r, 1.0 - wa_r
        if w_a <= 0.0:
            return mat_b
        if w_a >= 1.0:
            return mat_a
        na, nb = w_a / mat_a.m_atom, w_b / mat_b.m_atom
        xa, xb = na / (na + nb), nb / (na + nb)
        m_bar = 1.0 / (w_a / mat_a.m_atom + w_b / mat_b.m_atom)
        E_ion = tuple(xa * mat_a.E_ion[j] + xb * mat_b.E_ion[j]
                      for j in range(n_stage))
        g_ion = tuple(xa * mat_a.g_ion[j] + xb * mat_b.g_ion[j]
                      for j in range(n_stage + 1))
    name = f"{mat_a.name}{w_a:.4f}+{mat_b.name}{w_b:.4f}"
    return Material(
        name=name,
        rho0=1.0 / (w_a / mat_a.rho0 + w_b / mat_b.rho0),
        c0=nw("c0"), s=nw("s"), gamma0=nw("gamma0"),
        cv_solid=w_a * mat_a.cv_solid + w_b * mat_b.cv_solid,
        T_melt=nw("T_melt"), T_vap=nw("T_vap"),
        L_fusion=w_a * mat_a.L_fusion + w_b * mat_b.L_fusion,
        L_vap=w_a * mat_a.L_vap + w_b * mat_b.L_vap,
        E_cohesive=xa * mat_a.E_cohesive + xb * mat_b.E_cohesive,
        A=m_bar / AMU, Z=int(round(nw("Z"))),
        E_ion=E_ion, g_ion=g_ion,
        work_function=nw("work_function"),
        us_up_valid_to=min(mat_a.us_up_valid_to, mat_b.us_up_valid_to),
        mix_of=(mat_a.name, mat_b.name, float(w_a)),
    )


def get_material(name: str) -> Material:
    """Look up a material by (case-insensitive) name."""
    key = name.lower()
    if key not in _LIB:
        raise KeyError(
            f"Unknown material {name!r}. Available: {sorted(_LIB)}"
        )
    return _LIB[key]


def list_materials() -> list[str]:
    """Names of all materials in the built-in library."""
    return sorted(m.name for m in _LIB.values())


__all__ = [
    "MATERIALS_DIR", "load_material_file", "load_material_library",
    "material_from_spec",
    "Material", "get_material", "list_materials", "mix_materials",
    "ALUMINIUM", "IRON", "TUNGSTEN", "COPPER", "GLASS", "KAPTON",
    "OLIVINE", "DOLOMITE",
]
