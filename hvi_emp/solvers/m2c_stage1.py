"""Stage 1 (and part of Stage 2) on M2C: compressible multi-material flow.

M2C (Zhao, Ma, Islam, Narkhede & Wang, *Comput. Phys. Commun.*, 2026;
https://github.com/kevinwgy/m2c, GPLv3) is a 3-D finite-volume solver for
compressible multi-material flow with level-set interface tracking, FIVER
interfacial fluxes, phase transition, and a **multi-species non-ideal Saha
ionisation solver coupled to the flow**.

Why this bridge exists
----------------------
It closes three gaps at once, which is unusual enough to be worth stating.

1. **It is the hydrocode we can actually get.** The iSALE bridge needs an
   application and a wait; M2C is `git clone`, GPLv3, on GitHub. iSALE2D is
   also axisymmetric, so it cannot do the one thing below.

2. **Oblique incidence.** `hvi_emp`'s reduced chain turns `angle_deg` into
   `v cos(theta)` and is otherwise identical -- the plume stays axisymmetric
   about the surface normal at every angle and never becomes downrange
   directed (see `docs/THEORY.md`). M2C is 3-D Cartesian, so an oblique
   impact is a geometry change rather than a modelling impossibility.

3. **The two open physics tasks.** M2C ships Tillotson and
   ANEOS-Birch-Murnaghan-Debye equations of state -- the tabular EOS the
   linear Us-Up fit needs above ~20 km/s -- and a non-ideal Saha solver with
   Ebeling/Griem continuum lowering and temperature-dependent partition
   functions built from NIST level data. Those are exactly the framework's
   two outstanding physics items.

What it does *not* do: charge separation and EMP radiation. M2C stops at the
plasma state. That remains Stage 3 (`emp.py`, `picongpu_stage3`), so the two
codes compose rather than compete -- M2C upstream, `hvi_emp` downstream.

Units: this is the sharp edge
-----------------------------
**M2C input is in mm, g, s, K, A -- not SI.** Every deck begins with that
line, and getting it wrong produces a run that completes and is wrong by
powers of a thousand. The factors in `TO_M2C` are verified against the
constants in M2C's own shipped hypervelocity-impact deck
(`Tests/HVImpactShafquatIslam/2D_axisym_HVI/input.st`), which quotes
Planck = 6.62607004e-25, electron mass = 9.10938356e-28 and Boltzmann =
1.38064852e-14 in those units. `tests/test_m2c.py` pins all three, plus the
deck's tantalum density of 16.65e-3 g/mm^3.

Provenance of the input grammar
-------------------------------
Most of the block structure and keyword spellings come from that shipped
deck. The geometric entities do not -- the shipped deck uses a rod, and
`hvi_emp` models a sphere -- so those were read directly out of M2C's parser,
`IoData.cpp`, which is the code that consumes them:

* `Sphere` / `Parallelepiped` / `Spheroid` / `CylinderAndCone` /
  `CylinderWithSphericalCaps` are the entity names registered under
  `GeometricEntities` (`MultiInitialConditionsData::setup`).
* `Sphere` takes `Center_x`, `Center_y`, `Center_z`, `Radius`
  (`SphereData::getAssigner`).
* `CylinderAndCone` and `CylinderWithSphericalCaps` take `Axis_*`,
  `BaseCenter_*`, `CylinderRadius`, `CylinderHeight`, plus
  `ConeOpeningAngleInDegrees` / `ConeHeight` and `FrontSphericalCap` /
  `BackSphericalCap` respectively.

`check_grammar()` re-checks every keyword this module emits against a local
checkout, which catches an upstream rename rather than a mistake of ours.
Cheap, so worth running once after cloning:

    python -m hvi_emp.solvers.m2c_stage1 --check $M2C_HOME
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np

if __name__ == "__main__" and __package__ is None:      # pragma: no cover
    raise SystemExit(
        "\nThis file is a module of the `hvi_emp` package, not a script.\n"
        "Run it as a module from the project root:\n\n"
        "    python -m hvi_emp.solvers.m2c_stage1 --check $M2C_HOME\n")

from ..constants import K_B, N_AVOGADRO
from ..materials import Material, get_material

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

#: SI -> M2C (mm, g, s, K, A). Every factor here is checked against the
#: constants M2C's own hypervelocity-impact deck quotes; see the module
#: docstring. Do not "simplify" these -- pressure and charge really are 1.0,
#: which looks like a mistake and is not: kg/(m s^2) and g/(mm s^2) are the
#: same number, as are C and A s.
TO_M2C = {
    "length": 1.0e3,            # m -> mm
    "density": 1.0e-6,          # kg/m^3 -> g/mm^3
    "velocity": 1.0e3,          # m/s -> mm/s
    "pressure": 1.0,            # kg/(m s^2) -> g/(mm s^2)
    "energy_per_mass": 1.0e6,   # J/kg = m^2/s^2 -> mm^2/s^2
    "specific_heat": 1.0e6,     # J/(kg K) -> mm^2/(s^2 K)
    "mass": 1.0e3,              # kg -> g
    "action": 1.0e9,            # J s = kg m^2/s -> g mm^2/s
    "energy": 1.0e9,            # J = kg m^2/s^2 -> g mm^2/s^2
    "energy_per_K": 1.0e9,      # J/K -> g mm^2/(s^2 K)
    "number_density": 1.0e-9,   # 1/m^3 -> 1/mm^3
    "charge": 1.0,              # C = A s
    "time": 1.0,
    "temperature": 1.0,
}

#: Physical constants in M2C units, for the `under Ionization` block. M2C
#: reads these from the input file rather than hard-coding them, precisely
#: because the unit system is the user's choice.
M2C_CONSTANTS = {
    "PlanckConstant": 6.626_070_15e-34 * TO_M2C["action"],
    "ElectronCharge": 1.602_176_634e-19 * TO_M2C["charge"],
    "ElectronMass": 9.109_383_7015e-31 * TO_M2C["mass"],
    "BoltzmannConstant": 1.380_649e-23 * TO_M2C["energy_per_K"],
}


#: Cost per cell-step per core, microseconds.
#:
#: The non-ideal figure is **MEASURED**, not assumed: 13.12 s/step for
#: 353,864 cells on 32 ranks (Fe->Al, 50 km/s, 2-D axisym, NonIdealSaha with
#: Ebeling depression, MaxIts=200), from steady-state steps 2-5 of a real
#: run on a 64-core Xeon.
#:
#: The previous value here was an invented "~2 us, typical for explicit
#: multi-material finite volume". It was optimistic by a factor of ~590,
#: turning a 9-day run into a predicted 3 hours. The error was not the
#: hydro: it is the Saha solver, which runs per cell per step and dominates
#: everything else. That is why the rate is split by `ionisation` rather
#: than being one number.
#:
#: The hydro-only figure is still an ASSUMPTION -- no run with ionisation
#: off has been timed. Treat it as a lower bound.
CELL_STEP_US_NONIDEAL = 1187.0     # measured, non-ideal Saha
CELL_STEP_US_HYDRO = 2.0           # ASSUMED, ionisation off/ideal

#: Mesh grading, matching what `_mesh_block` writes.
_FINE_ZONE_RADII = 4.0             # fine out to 4 projectile radii
_COARSENING = 8.0                  # dx_coarse = 8 * dx_fine

#: Correction for the smooth ramp between the fine and coarse zones, which
#: a two-zone count misses.
#:
#: CALIBRATED AGAINST ONE MESH: the default Fe->Al deck, for which M2C
#: reported 994 x 356 = 353,864 cells against a two-zone count of 650 x 325
#: = 211,250. One data point, so treat the cell count as good to tens of
#: percent, not percent -- but that is two orders better than the uniform
#: assumption it replaces.
MESH_GRADING_FACTOR = 1.68

#: How every generated command names the M2C binary.
#:
#: ``${VAR:?msg}`` rather than ``$VAR``: with M2C_HOME unset, ``$M2C_HOME/m2c``
#: expands to the literal ``/m2c`` and mpirun reports
#:
#:     prterun was unable to launch ... Executable: /m2c
#:
#: which reads as a permissions problem on a file that was never there. The
#: ``:?`` form makes the shell stop first and say which variable is unset.
M2C_HOME_GUARD = ('"${M2C_HOME:?set it to the directory holding the m2c '
                  'binary, e.g. ~/src/m2c/build}"')


def si_to_m2c(value, kind: str):
    """Convert an SI quantity to M2C's mm-g-s-K-A system."""
    if kind not in TO_M2C:
        raise KeyError(f"unknown unit kind {kind!r}; known: {sorted(TO_M2C)}")
    f = TO_M2C[kind]
    return np.asarray(value, float) * f if np.ndim(value) else float(value) * f


def m2c_to_si(value, kind: str):
    """Inverse of `si_to_m2c`, for reading M2C output back."""
    if kind not in TO_M2C:
        raise KeyError(f"unknown unit kind {kind!r}; known: {sorted(TO_M2C)}")
    f = TO_M2C[kind]
    return np.asarray(value, float) / f if np.ndim(value) else float(value) / f


# ---------------------------------------------------------------------------
# Ambient gases
# ---------------------------------------------------------------------------

#: The material library is a *condensed-matter* library -- Mie-Grueneisen
#: shock EOS, latent heats, cohesive energies. None of that applies to a
#: chamber gas, so ambient gases live here as ideal (zero-`PressureConstant`)
#: stiffened gases instead of being forced into `Material`.
#:
#: `Z` and `A` are per-atom values for the Saha solver; N2 dissociates well
#: before it ionises, so nitrogen is entered atomically.
AMBIENT_GASES = {
    "vacuum": dict(label="near-vacuum ideal gas", gamma=5.0 / 3.0,
                   M=28.0e-3, Z=7, A=14.007, ionises=False),
    "ar":  dict(label="argon", gamma=5.0 / 3.0,
                M=39.948e-3, Z=18, A=39.948, ionises=True),
    "he":  dict(label="helium", gamma=5.0 / 3.0,
                M=4.0026e-3, Z=2, A=4.0026, ionises=True),
    "n2":  dict(label="nitrogen (treated atomically)", gamma=7.0 / 5.0,
                M=28.014e-3, Z=7, A=14.007, ionises=True),
    "air": dict(label="air (treated as atomic nitrogen)", gamma=7.0 / 5.0,
                M=28.96e-3, Z=7, A=14.4, ionises=True),
}


def _gas(name: str | None) -> dict:
    key = (name or "vacuum").lower()
    if key not in AMBIENT_GASES:
        raise KeyError(f"unknown ambient gas {name!r}; "
                       f"known: {sorted(AMBIENT_GASES)}")
    return AMBIENT_GASES[key]


def _gas_density(gas: dict, pressure: float, T: float = 300.0) -> float:
    """Ideal-gas density [kg/m^3] at `pressure` [Pa] and `T` [K]."""
    return pressure * gas["M"] / (N_AVOGADRO * K_B * T)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Mesh defaults for the axisymmetric case: 50 cells per projectile radius
#: over 24 radii, matching the shipped deck's resolution. 5.8M cells.
DEFAULTS_2D = {"cells_per_radius": 50.0, "domain_radii": 24.0}

#: Mesh defaults for the 3-D case. Cubing the axisymmetric numbers gives
#: about 461M cells and 86 GB (graded), so 3-D trades resolution and domain
#: for feasibility: 64M cells, ~12 GB, which fits a large workstation. An
#: oblique run at 2-D resolution is a cluster allocation, not a default.
DEFAULTS_3D = {"cells_per_radius": 20.0, "domain_radii": 10.0}

#: Above this, `estimate_resources` is telling you the run will not fit on a
#: workstation. Roughly the RAM of a large single node.
MEMORY_WARN_GB = 256.0

#: Ionisation ladder depth for an ambient gas. Materials get theirs from the
#: number of stages in their YAML; a gas has no `Material` record, so it
#: takes this. Four stages is generous for a chamber gas at impact
#: temperatures, where Z-bar is order 1.
AMBIENT_MAX_CHARGE = 4


@dataclass
class M2CConfig:
    """An M2C hypervelocity-impact problem, in SI. Converted on write.

    Parameters
    ----------
    projectile, target : Material
    diameter : float
        Projectile diameter [m].
    velocity : float
        Impact speed [m/s].
    angle_deg : float
        Incidence measured from the surface normal. **Non-zero forces 3-D
        Cartesian**, because an oblique impact is not axisymmetric -- which
        is the whole reason to run M2C rather than the reduced chain.
    ambient : str or None
        Key into `AMBIENT_GASES`. ``None`` means near-vacuum (a very tenuous
        ideal gas: a finite-volume code cannot have a true void). Naming a
        real gas reproduces a chamber experiment, where Islam et al. (2023)
        find the *ambient gas itself* ionises from compression ahead of the
        projectile -- a plasma source `hvi_emp` does not model at all.
    ambient_pressure : float
        Ambient pressure [Pa]. Default 1e-4 Pa is a decent laboratory vacuum.
    cells_per_radius : float or None
        Resolution in the refined zone. ``None`` picks a default that
        depends on dimensionality, because the same numbers that are
        comfortable in 2-D are ruinous in 3-D: 50 cells per radius over 24
        radii is 5.8M cells axisymmetric and 13.8 *billion* in 3-D. The 3-D
        defaults (`DEFAULTS_3D`) are chosen to fit a large workstation;
        raise them deliberately, on a machine that can take it.
    domain_radii : float or None
        Domain half-width in projectile radii. See `cells_per_radius`.
    t_end : float or None
        End time [s]. ``None`` derives it from the time for the shock to
        cross several projectile diameters.
    ionisation : {"non-ideal", "ideal", "off"}
        ``non-ideal`` adds continuum lowering, which is the physics behind
        the framework's open pressure-ionisation discrepancy.
    depression_model : {"Ebeling", "Griem", "None"}
    projectile_shape : {"sphere", "rod"}
        ``sphere`` is what `hvi_emp` models; ``rod`` matches the geometry of
        M2C's own shipped hypervelocity-impact test, which is the right
        choice when the point is to reproduce their result.
    """
    projectile: Material
    target: Material
    diameter: float
    velocity: float
    angle_deg: float = 0.0
    ambient: str | None = None
    ambient_pressure: float = 1.0e-4
    cells_per_radius: float | None = None
    domain_radii: float | None = None
    t_end: float | None = None
    n_outputs: int = 60
    ionisation: str = "non-ideal"
    depression_model: str = "Ebeling"
    projectile_shape: str = "sphere"
    cfl: float = 0.1
    atomic_data_dir: str = "AtomicData"

    #: How M2C evaluates the partition function: OnTheFly | (Cubic)Spline.
    #:
    #: **OnTheFly is required by the atomic data this bridge generates**, and
    #: is not a compromise. `write_atomic_data()` writes one level per charge
    #: state at zero excitation energy, so U_r = g_r exactly -- a constant,
    #: independent of temperature. There is nothing to interpolate.
    #:
    #: `CubicSplineInterpolation` does not merely waste effort on that data,
    #: it aborts. `InitializeInterpolationForCharge` takes the first NON-ZERO
    #: excitation energy as `factor = -E[r][i]/kb`; with a single level at
    #: E = 0 no such entry exists, factor stays 0, and then
    #:
    #:     expmin = exp(0/Tmin) = 1;  expmax = exp(0/Tmax) = 1
    #:     assert(expmin < expmax);          // 1 < 1 is false
    #:
    #: fails on every rank (AtomicIonizationData.cpp:218). Switch back to
    #: interpolation only after replacing E_* and g_* with real NIST level
    #: lists, at which point it is the faster choice.
    partition_function: str = "OnTheFly"
    notes: list = field(default_factory=list)

    def __post_init__(self):
        if self.ionisation not in ("non-ideal", "ideal", "off"):
            raise ValueError("ionisation must be 'non-ideal', 'ideal' or "
                             f"'off', got {self.ionisation!r}")
        if self.depression_model not in ("Ebeling", "Griem", "None"):
            raise ValueError("depression_model must be 'Ebeling', 'Griem' or "
                             f"'None', got {self.depression_model!r}")
        if self.projectile_shape not in ("sphere", "rod"):
            raise ValueError("projectile_shape must be 'sphere' or 'rod', got "
                             f"{self.projectile_shape!r}")
        if self.diameter <= 0.0 or self.velocity <= 0.0:
            raise ValueError("diameter and velocity must be positive")
        if not 0.0 <= self.angle_deg < 90.0:
            raise ValueError(
                f"angle_deg must be in [0, 90), got {self.angle_deg}")
        gas = _gas(self.ambient)        # validate early, not at write time
        # Resolution defaults depend on dimensionality: the axisymmetric
        # numbers are 2400 cells per axis, which is 5.8M cells in 2-D and
        # about 461M in 3-D. Silently generating the latter would be worse
        # than useless, so 3-D gets its own defaults.
        res = DEFAULTS_3D if self.is_3d else DEFAULTS_2D
        if self.cells_per_radius is None:
            self.cells_per_radius = res["cells_per_radius"]
        if self.domain_radii is None:
            self.domain_radii = res["domain_radii"]
        if self.cells_per_radius < 4:
            raise ValueError("cells_per_radius below 4 cannot resolve the "
                             "projectile at all")
        if self.domain_radii < 2:
            raise ValueError("domain_radii below 2 puts the boundary inside "
                             "the crater")
        if self.t_end is None:
            # Long enough for the shock to cross ~6 projectile diameters at
            # the target's bulk sound speed, so the release wave returns and
            # the vapour has started to expand.
            self.t_end = 6.0 * self.diameter / max(self.target.c0, 1.0)
        if self.is_3d:
            self.notes.append(
                f"angle_deg = {self.angle_deg:g} is non-zero, so the mesh is "
                f"3-D Cartesian rather than 2-D axisymmetric. This is the "
                f"case the reduced hvi_emp chain structurally cannot "
                f"represent, and it costs roughly "
                f"{self.cell_count() / 1e6:.0f}M cells at "
                f"{self.cells_per_radius:g} cells per projectile radius.")
        mem = self.estimate_resources()["memory_GB"]
        if mem > MEMORY_WARN_GB:
            self.notes.append(
                f"Estimated {mem:.0f} GB of mesh, above the "
                f"{MEMORY_WARN_GB:.0f} GB single-node guide. Either run it "
                f"distributed across "
                f"nodes, or cut cells_per_radius / domain_radii -- memory "
                f"goes as the "
                f"{'cube' if self.is_3d else 'square'} of both.")
        if gas["ionises"] and self.ionisation != "off":
            self.notes.append(
                f"The ambient {gas['label']} is included in the ionisation "
                f"block. Islam et al. (2023) find the ambient gas contributes "
                f"free electrons in its own right; hvi_emp's Stage 1 has no "
                f"such term, so the two will not agree -- and that difference "
                f"is the point of running with a chamber gas at all.")

    # -- derived ----------------------------------------------------------
    @property
    def is_3d(self) -> bool:
        """Oblique incidence is not axisymmetric, so it needs three axes."""
        return self.angle_deg > 0.0

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def mass(self) -> float:
        """Projectile mass [kg], for cross-checking against a scenario."""
        return self.projectile.rho0 * np.pi * self.diameter ** 3 / 6.0

    def ambient_density(self) -> float:
        """Ambient density [kg/m^3] from the ideal gas law at 300 K."""
        return _gas_density(_gas(self.ambient), self.ambient_pressure)

    def cell_size(self) -> float:
        """Fine-zone cell size [m]."""
        return self.radius / self.cells_per_radius

    def cell_count(self) -> float:
        """Cell count for the GRADED mesh this deck actually writes.

        The previous version assumed a uniform fine mesh across the whole
        domain and over-counted by 16x: it predicted 5.76e6 cells where M2C
        reported 353,864 for the same deck. Combined with a rate that was
        590x optimistic, the two errors partly cancelled and hid each other
        -- which is worse than either alone, because the total looked
        plausible.

        The mesh is fine (`dx_fine`) out to `_FINE_ZONE_RADII` projectile
        radii and coarse (8 x dx_fine, matching `_mesh_block`) beyond, with
        a smooth ramp between. This counts the two zones and applies
        `MESH_GRADING_FACTOR` for the ramp.
        """
        cpr = self.cells_per_radius
        fine_half = _FINE_ZONE_RADII * cpr           # cells, one side
        coarse_half = ((self.domain_radii - _FINE_ZONE_RADII) * cpr
                       / _COARSENING)                # cells, one side

        # Axial spans the full domain; radial is a half-domain (symmetry
        # axis at y=0), which is why they differ.
        n_axial = 2.0 * (fine_half + coarse_half)
        n_radial = fine_half + coarse_half
        if self.is_3d:
            return n_axial * n_axial * n_axial * MESH_GRADING_FACTOR
        return n_axial * n_radial * MESH_GRADING_FACTOR

    def estimate_resources(self, cores: int = 64) -> dict:
        """Cells, memory and a wall-clock guess.

        The wall-clock figure is an order-of-magnitude estimate from the cell
        count and an explicit CFL-limited step -- not a measurement, and
        labelled as such wherever it is printed.
        """
        cells = self.cell_count()
        dx = self.cell_size()
        speed = max(self.target.c0 + self.velocity, 1.0)
        dt = self.cfl * dx / speed
        n_steps = self.t_end / dt
        rate = (CELL_STEP_US_NONIDEAL if self.ionisation == "non-ideal"
                else CELL_STEP_US_HYDRO)
        seconds = cells * n_steps * rate * 1e-6 / max(cores, 1)
        return {
            "cells": cells,
            "cell_size_m": dx,
            "dt_s": dt,
            "n_steps": n_steps,
            "memory_GB": cells * 200.0 / 1024 ** 3,   # ~200 B/cell
            "core_hours": seconds * cores / 3600.0,
            "wall_hours_estimate": seconds / 3600.0,
            "dimensionality": "3-D Cartesian" if self.is_3d else "2-D axisym",
            "cell_step_us": rate,
            # True only for the non-ideal path, where the rate came from a
            # timed run. Say which, so nobody plans a week around a guess.
            "rate_is_measured": self.ionisation == "non-ideal",
            "cost_dominated_by": ("non-ideal Saha (per cell, per step)"
                                  if self.ionisation == "non-ideal"
                                  else "hydro"),
        }


def config_from_scenario(scenario, **overrides) -> M2CConfig:
    """Build an `M2CConfig` from an `hvi_emp` scenario or a plain mapping.

    Accepts anything exposing `projectile`/`target`/`diameter`/`velocity`/
    `angle_deg` as attributes or keys, so the pydantic schema, the dataclass
    fallback and a bare dict from YAML all work unchanged.
    """
    def pick(name, default=None):
        if isinstance(scenario, dict):
            return scenario.get(name, default)
        return getattr(scenario, name, default)

    def as_material(v):
        return v if isinstance(v, Material) else get_material(str(v))

    diameter = pick("diameter")
    if diameter is None:
        diameter = pick("projectile_diameter", pick("size"))
    if diameter is None:
        raise KeyError(
            "scenario has no projectile diameter (looked for 'diameter', "
            "'projectile_diameter', 'size')")
    velocity = pick("velocity", pick("impact_velocity"))
    if velocity is None:
        raise KeyError("scenario has no impact velocity")

    kw = dict(
        projectile=as_material(pick("projectile", "al")),
        target=as_material(pick("target", "al")),
        diameter=float(diameter),
        velocity=float(velocity),
        angle_deg=float(pick("angle_deg", 0.0) or 0.0),
    )
    kw.update(overrides)
    return M2CConfig(**kw)


# ---------------------------------------------------------------------------
# Grammar provenance
# ---------------------------------------------------------------------------

#: The complete vocabulary this module emits, every entry traced to M2C's
#: shipped HVI deck or to `IoData.cpp` (the geometric entities -- see the
#: module docstring).
#:
#: This is deliberately exhaustive rather than a sample: `check_grammar()`
#: only protects what is listed, and `tests/test_m2c.py` asserts that the
#: generated deck emits nothing outside this tuple. Adding a keyword to the
#: writer without adding it here fails that test, which is the point.
VERIFIED_KEYWORDS = (
    # -- geometric entities, from IoData.cpp --------------------------------
    "GeometricEntities", "Sphere", "Center_x", "Center_y", "Center_z",
    "Radius", "CylinderAndCone", "CylinderWithSphericalCaps",
    "FrontSphericalCap", "BackSphericalCap", "Axis_x", "Axis_y", "Axis_z",
    "BaseCenter_x", "BaseCenter_y", "BaseCenter_z", "CylinderRadius",
    "CylinderHeight", "ConeOpeningAngleInDegrees", "ConeHeight",
    "InitialState", "MaterialID",
    # -- top-level blocks ---------------------------------------------------
    "Mesh", "Equations", "Ionization", "InitialCondition",
    "BoundaryConditions", "Space", "MultiPhase", "Time", "Output",
    # -- mesh ---------------------------------------------------------------
    "Type", "X0", "Xmax", "Y0", "Ymax", "Z0", "Zmax", "NumberOfCellsZ",
    "ControlPointX", "ControlPointY", "ControlPointZ", "Coordinate",
    "CellWidth", "BoundaryConditionX0", "BoundaryConditionXmax",
    "BoundaryConditionY0", "BoundaryConditionYmax", "BoundaryConditionZ0",
    "BoundaryConditionZmax", "Inlet",
    # -- equations of state -------------------------------------------------
    "Material", "EquationOfState", "ExtendedMieGruneisenModel",
    "ReferenceDensity", "BulkSpeedOfSound", "HugoniotSlope", "ReferenceGamma",
    "SpecificHeatAtConstantVolume", "ReferenceSpecificInternalEnergy",
    "ReferenceTemperature", "VolumetricStrainBreak", "StiffenedGasModel",
    "SpecificHeatRatio", "PressureConstant", "DensityCutOff",
    "PressureCutOff", "DensityUpperLimit", "DensityPrescribedAtFailure",
    # -- ionisation ---------------------------------------------------------
    "NonIdealSahaEquation", "IdealSahaEquation", "DepressionModel",
    "PartitionFunctionEvaluation", "MaxIts", "ConvergenceTolerance",
    "Element", "MolarFraction", "MolarMass", "AtomicNumber",
    "MaxChargeNumber", "IonizationEnergyFile", "ExcitationEnergyFilesPrefix",
    "ExcitationEnergyFilesSuffix", "DegeneracyFilesPrefix",
    "DegeneracyFilesSuffix", "PlanckConstant", "ElectronCharge",
    "ElectronMass", "BoltzmannConstant",
    # -- state variables ----------------------------------------------------
    "Density", "Pressure", "Temperature", "Velocity", "VelocityX",
    "VelocityY", "VelocityZ",
    # -- numerics -----------------------------------------------------------
    "NavierStokes", "Flux", "LocalLaxFriedrichs", "Reconstruction",
    "VariableType", "ConservativeCharacteristic", "Limiter",
    "GeneralizedMinMod", "GeneralizedMinModCoefficient", "LevelSet",
    "Solver", "Bandwidth", "Reinitialization", "Frequency",
    "ReconstructionAtInterface", "PhaseChange", "RiemannNormal",
    "LevelSetCorrectionFrequency", "ConstantReconstructionDepth",
    "Explicit", "RungeKutta2", "MaxTime", "CFL",
    # -- output -------------------------------------------------------------
    "Prefix", "Solution", "TimeInterval", "LevelSet0", "LevelSet1",
    "MeanCharge", "ElectronDensity", "Probes", "Node", "X", "Y", "Z",
    "IonizationResult", "VerboseScreenOutput",
)

#: Kept as an explicit, currently-empty record. Every keyword this module
#: emits has been traced to either the shipped deck or `IoData.cpp`; if a
#: future addition cannot be, it belongs here and `check_grammar()` labels it
#: as never-verified rather than as an upstream rename.
INFERRED_KEYWORDS: tuple = ()

#: Enum VALUES the deck assigns, as opposed to the keys above.
#:
#: These were missing from the check entirely, which is how a deck could
#: report "133 of 133 keywords found" and still abort at startup: the key
#: `PartitionFunctionEvaluation` exists, but the value it was given
#: (`CubicSplineInterpolation`) is only valid when there is real excitation
#: data behind it. `check_grammar()` now greps for each as a quoted string,
#: the form `ClassToken` registers.
VERIFIED_VALUES: tuple = (
    # PartitionFunctionEvaluation -- ON_THE_FLY=0, CUBIC_SPLINE=1, LINEAR=2
    "OnTheFly", "CubicSplineInterpolation", "LinearInterpolation",
    # Ionization model types
    "NonIdealSahaEquation", "IdealSahaEquation",
    # DepressionModel
    "Griem", "Ebeling",
)


# ---------------------------------------------------------------------------
# Input-deck generation
# ---------------------------------------------------------------------------

def _mie_gruneisen_block(mat: Material, mat_id: int, indent="  ") -> str:
    """An `ExtendedMieGruneisen` material block.

    The mapping to `hvi_emp.Material` is exact and needs no fitting, because
    both codes use the same linear Us-Up Mie-Grueneisen form:

        ReferenceDensity              <- rho0
        BulkSpeedOfSound              <- c0
        HugoniotSlope                 <- s
        ReferenceGamma                <- gamma0
        SpecificHeatAtConstantVolume  <- cv_solid

    That correspondence is why this bridge is thin: the two codes already
    agree on the EOS, so only the units differ.
    """
    rho = si_to_m2c(mat.rho0, "density")
    # Mie-Grueneisen gives c^2 < 0 once 1 - s*eta <= 0, i.e. at
    # rho = rho0 * s/(s-1); stay just below that.
    rho_max = rho * (1.0 + 1.0 / max(mat.s - 1.0, 0.05)) * 0.95
    return f"""{indent}under Material[{mat_id}] {{ // {mat.name}
{indent}  EquationOfState = ExtendedMieGruneisen;
{indent}  under ExtendedMieGruneisenModel {{
{indent}    ReferenceDensity = {rho:.6e}; // g/mm3
{indent}    BulkSpeedOfSound = {si_to_m2c(mat.c0, 'velocity'):.6e}; // mm/s
{indent}    HugoniotSlope = {mat.s:.6f};
{indent}    ReferenceGamma = {mat.gamma0:.6f};
{indent}    SpecificHeatAtConstantVolume = \
{si_to_m2c(mat.cv_solid, 'specific_heat'):.6e}; // mm2/(s2 K)
{indent}    ReferenceSpecificInternalEnergy = 0.0;
{indent}    ReferenceTemperature = 300.0;
{indent}    VolumetricStrainBreak = -4.0;
{indent}  }}
{indent}  DensityCutOff = {rho * 1.0e-6:.6e};
{indent}  PressureCutOff = 1.0;
{indent}  DensityUpperLimit = {rho_max:.6e};
{indent}  DensityPrescribedAtFailure = {rho:.6e};
{indent}}}
"""


def _stiffened_gas_block(gas: dict, rho: float, mat_id: int = 0,
                         indent="  ") -> str:
    """The ambient gas as an ideal (zero-`PressureConstant`) stiffened gas."""
    cv = N_AVOGADRO * K_B / (gas["M"] * (gas["gamma"] - 1.0))   # J/(kg K)
    rho_m2c = si_to_m2c(rho, "density")
    return f"""{indent}under Material[{mat_id}] {{ // ambient: {gas['label']}
{indent}  EquationOfState = StiffenedGas;
{indent}  under StiffenedGasModel {{
{indent}    SpecificHeatRatio = {gas['gamma']:.7f};
{indent}    PressureConstant = 0.0;
{indent}    SpecificHeatAtConstantVolume = \
{si_to_m2c(cv, 'specific_heat'):.6e}; // mm2/(s2 K)
{indent}  }}
{indent}  DensityCutOff = {rho_m2c * 1.0e-4:.6e};
{indent}  PressureCutOff = 1.0e-10;
{indent}  DensityPrescribedAtFailure = {rho_m2c:.6e};
{indent}}}
"""


def _ionisation_block(cfg: M2CConfig, species: list) -> str:
    """The `under Ionization` block.

    `species` is a list of ``(material_id, label, A, Z, max_charge)``. One
    `Element` per material: exact for the elemental metals, a simplification
    for the library's compounds (SiO2, Kapton, Olivine, Dolomite each carry a
    single mean Z and A). M2C itself supports a proper multi-element mixture
    -- its shipped deck uses five elements for soda-lime glass -- so a
    compound study should hand-edit this block, and the generated deck says
    so.

    `max_charge` is set to the number of ionisation stages the material's own
    YAML defines, so M2C solves the *same* ladder `hvi_emp.ionization` does
    and the two are directly comparable. It also keeps the deck from
    demanding atomic data files that M2C may not ship for the higher stages.
    If a run reports Z-bar approaching `MaxChargeNumber`, that is the ladder
    truncating, not physics -- extend the material YAML and regenerate.
    """
    if cfg.ionisation == "off":
        return ""
    kind = ("NonIdealSahaEquation" if cfg.ionisation == "non-ideal"
            else "IdealSahaEquation")
    lines = ["under Ionization {", "",
             "  // Constants are given in the input unit system (mm, g, s,",
             "  // K, A) -- M2C does not hard-code them."]
    for key, val in M2C_CONSTANTS.items():
        lines.append(f"  {key} = {val:.9e};")
    lines += ["",
              "  // One Element per material. Exact for elemental metals; a",
              "  // mean-Z simplification for compounds -- hand-edit into a",
              "  // proper multi-element mixture for those.",
              "  // MaxChargeNumber matches the ionisation ladder in the",
              "  // hvi_emp material YAML, so the two codes solve the same",
              "  // problem. Z-bar approaching it means truncation, not",
              "  // physics."]
    for mid, label, A, Z, max_charge in species:
        z = int(round(Z))
        zmax = max(1, min(int(max_charge), z))
        lines += [
            f"  under Material[{mid}] {{ // {label}",
            f"    Type = {kind};",
        ]
        if cfg.ionisation == "non-ideal":
            lines.append(f"    DepressionModel = {cfg.depression_model};")
        lines += [
            f"    PartitionFunctionEvaluation = {cfg.partition_function};",
            "    MaxIts = 200;",
            "    ConvergenceTolerance = 1.0e-5;",
            "",
            f"    under Element[0] {{ // Z = {z}, ladder truncated at {zmax}",
            "      MolarFraction = 1.0;",
            f"      MolarMass = {A:.4f}; // g/mol",
            f"      AtomicNumber = {z};",
            f"      MaxChargeNumber = {zmax};",
            f'      IonizationEnergyFile = "{cfg.atomic_data_dir}/I_{z}.txt";',
            f'      ExcitationEnergyFilesPrefix = '
            f'"{cfg.atomic_data_dir}/E_{z}_";',
            '      ExcitationEnergyFilesSuffix = ".txt";',
            f'      DegeneracyFilesPrefix = "{cfg.atomic_data_dir}/g_{z}_";',
            '      DegeneracyFilesSuffix = ".txt";',
            "    }",
            "  }",
            "",
        ]
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_atomic_data(directory: str, materials, max_charge=None) -> list:
    """Write the ``AtomicData/`` files M2C's Saha solver reads. Returns paths.

    **M2C does not ship these files.** The repository has no ``AtomicData``
    directory, so a deck that references one dies at startup with::

        *** Error: Cannot open ionization energy file AtomicData/I_13.txt.

    M2C reads three sets per element (``AtomicIonizationData::Setup``):

    ==================  =====================  =========================
    file                contents               missing behaviour
    ==================  =====================  =========================
    ``I_<Z>.txt``       ionisation energies    **fatal** -- exit_mpi()
    ``E_<Z>_<r>.txt``   excitation energies    warning; truncates rmax
    ``g_<Z>_<r>.txt``   degeneracies           warning; truncates rmax
    ==================  =====================  =========================

    The warnings are not harmless: ``rmax`` collapses to
    ``min(len(I), len(E), len(g))``, so supplying ``I`` alone sets rmax to 0
    and switches ionisation off silently. All three are written here.

    **What is written is the ground-state approximation**: one level per
    charge state, at zero excitation energy, with the ground-state
    statistical weight. The partition function is then ``U_r = g_r`` exactly,
    which is the closure ``hvi_emp.ionization`` already uses -- so the two
    codes solve the *same* ladder and their Z-bar is directly comparable,
    which is the whole point of running M2C beside the reduced chain.

    It is an approximation. Excited states raise U_r and therefore Z-bar at
    high temperature. To do better, replace ``E_<Z>_<r>.txt`` and
    ``g_<Z>_<r>.txt`` with full level lists from NIST ASD; M2C reads up to
    10,000 levels per charge state and the file format is one number per
    line, so nothing here needs to change.

    Energies are written in M2C's mm-g-s-K-A system, where 1 J = 1e9 units,
    consistent with the ``BoltzmannConstant`` the deck emits.
    """
    import os as _os

    if isinstance(materials, (str, Material)):
        materials = [materials]
    _os.makedirs(directory, exist_ok=True)

    written = []
    seen = set()
    for mat in materials:
        m = mat if isinstance(mat, Material) else get_material(str(mat))
        z = int(round(m.Z))
        if z in seen:
            continue
        seen.add(z)

        e_ion = list(m.E_ion)                      # joules, ascending stage
        g_ion = list(m.g_ion)
        n = len(e_ion) if max_charge is None else min(len(e_ion),
                                                      int(max_charge))
        if n <= 0:
            raise ValueError(f"{m.name}: no ionisation ladder to write")

        # NUMBERS ONLY -- no comments, no header, no units line.
        #
        # M2C reads these with:
        #     for(i=0;i<MaxCount;i++){ file>>tmp;
        #                              if(file.eof()) break;
        #                              X.push_back(tmp); }
        # Extracting '#' into a double sets failbit, NOT eofbit, so the
        # loop never breaks: it appends the failed value (0 since C++11)
        # MaxCount times. A commented I_13.txt would load as 1000 zero
        # ionisation energies and the run would complete with nonsense
        # instead of failing. Provenance goes in README.txt, which M2C
        # never opens.
        p = _os.path.join(directory, f"I_{z}.txt")
        with open(p, "w") as fh:
            for e in e_ion[:n]:
                fh.write(f"{si_to_m2c(e, 'energy'):.9e}\n")
        written.append(p)

        # E_<Z>_<r>.txt and g_<Z>_<r>.txt, one charge state per file.
        for r in range(n):
            pe = _os.path.join(directory, f"E_{z}_{r}.txt")
            with open(pe, "w") as fh:
                fh.write("0.0\n")                  # ground state only
            written.append(pe)

            pg = _os.path.join(directory, f"g_{z}_{r}.txt")
            g = g_ion[r] if r < len(g_ion) else 1.0
            with open(pg, "w") as fh:
                fh.write(f"{float(g):.6f}\n")
            written.append(pg)

    readme = _os.path.join(directory, "README.txt")
    with open(readme, "w") as fh:
        fh.write(
            "Atomic data for M2C's Saha solver\n"
            "=================================\n"
            "Written by hvi_emp.solvers.m2c_stage1.write_atomic_data().\n\n"
            "M2C ships no AtomicData directory. These are generated from\n"
            "the hvi_emp material YAML (sourced from NIST ASD), so both\n"
            "codes solve the same ionisation ladder and their Z-bar is\n"
            "directly comparable.\n\n"
            "  I_<Z>.txt      successive ionisation energies\n"
            "  E_<Z>_<r>.txt  excitation energies, charge state r\n"
            "  g_<Z>_<r>.txt  degeneracies, charge state r\n\n"
            "Energies are in M2C's mm-g-s-K-A system, where 1 J = 1e9,\n"
            "matching the BoltzmannConstant emitted in input.st.\n\n"
            "APPROXIMATION: one level per charge state at zero excitation\n"
            "energy, so the partition function is exactly the ground-state\n"
            "statistical weight. Excited states raise it, and so raise\n"
            "Z-bar at high temperature. For better, drop in full NIST ASD\n"
            "level lists -- M2C reads up to 10,000 levels per charge\n"
            "state, same one-number-per-line format.\n\n"
            "Do NOT put comments in the data files. M2C reads them with\n"
            "`file >> double` and breaks only on eof, so a '#' sets\n"
            "failbit and the reader appends zeros up to its limit rather\n"
            "than reporting an error.\n")
    written.append(readme)
    return written


def _mesh_block(cfg: M2CConfig) -> str:
    """Graded Cartesian or cylindrical mesh, sized from the projectile."""
    R = si_to_m2c(cfg.radius, "length")
    half = cfg.domain_radii * R
    dx_fine = si_to_m2c(cfg.cell_size(), "length")
    dx_coarse = 8.0 * dx_fine
    fine = 4.0 * R              # keep the fine zone around the impact site

    if cfg.is_3d:
        head = f"""under Mesh {{
  Type = ThreeDimensional;
  X0   = {-half:.6f};
  Xmax = {half:.6f};
  Y0   = {-half:.6f};
  Ymax = {half:.6f};
  Z0   = {-half:.6f};
  Zmax = {half:.6f};

  BoundaryConditionX0   = Inlet;
  BoundaryConditionXmax = Inlet;
  BoundaryConditionY0   = Inlet;
  BoundaryConditionYmax = Inlet;
  BoundaryConditionZ0   = Inlet;
  BoundaryConditionZmax = Inlet;
"""
        axes = ("X", "Y", "Z")
        spans = {a: (-half, half) for a in axes}
    else:
        # Cylindrical: X is the symmetry axis, Y the radius (Y0 = 0), one
        # cell in Z. This is the shipped 2D_axisym_HVI layout.
        head = f"""under Mesh {{
  Type = Cylindrical;
  X0   = {-half:.6f};
  Xmax = {half:.6f};
  Y0   = 0.0;
  Ymax = {half:.6f};
  Z0   = {-0.5 * dx_fine:.6e};
  Zmax = {0.5 * dx_fine:.6e};

  BoundaryConditionX0   = Inlet;
  BoundaryConditionXmax = Inlet;
  BoundaryConditionY0   = Symmetry;
  BoundaryConditionYmax = Inlet;
  BoundaryConditionZ0   = Symmetry;
  BoundaryConditionZmax = Symmetry;
"""
        axes = ("X", "Y")
        spans = {"X": (-half, half), "Y": (0.0, half)}

    body = ["\n"]
    for ax in axes:
        lo, hi = spans[ax]
        pts = [(lo, dx_coarse), (max(-fine, lo), dx_fine),
               (min(fine, hi), dx_fine), (hi, dx_coarse)]
        seen, clean = [], []
        for c, w in pts:
            c = min(max(c, lo), hi)
            if any(abs(c - s) < 1.0e-12 for s in seen):
                continue
            seen.append(c)
            clean.append((c, w))
        clean.sort()
        for i, (c, w) in enumerate(clean):
            body.append(f"  under ControlPoint{ax}[{i}] "
                        f"{{Coordinate = {c:.6f}; CellWidth = {w:.6e};}}\n")
        body.append("\n")
    if not cfg.is_3d:
        body.append("  NumberOfCellsZ = 1;\n")
    return head + "".join(body) + "}\n"


def _projectile_entity(cfg: M2CConfig) -> str:
    """The projectile as a geometric entity with its initial state."""
    R = si_to_m2c(cfg.radius, "length")
    v = si_to_m2c(cfg.velocity, "velocity")
    th = np.radians(cfg.angle_deg)
    vx, vy = v * np.cos(th), v * np.sin(th)
    rho = si_to_m2c(cfg.projectile.rho0, "density")
    p0 = cfg.ambient_pressure
    x0 = -1.25 * R              # start just clear of the target surface

    state = f"""      under InitialState {{
        MaterialID = 2;
        Density = {rho:.6e};
        VelocityX = {vx:.6e};
        VelocityY = {vy:.6e};
        VelocityZ = 0.0;
        Pressure = {p0:.6e};
      }}"""

    if cfg.projectile_shape == "sphere":
        return f"""    // Sphere keywords: SphereData::getAssigner, IoData.cpp
    under Sphere[0] {{ // projectile
      Center_x = {x0:.6f};
      Center_y = 0.0;
      Center_z = 0.0;
      Radius   = {R:.6f};
{state}
    }}
"""
    return f"""    // Keywords: CylinderSphereData::getAssigner, IoData.cpp
    under CylinderWithSphericalCaps[0] {{ // projectile (rod)
      Axis_x = 1.0;
      Axis_y = 0.0;
      Axis_z = 0.0;
      BaseCenter_x = {x0 - 2.0 * R:.6f};
      BaseCenter_y = 0.0;
      BaseCenter_z = 0.0;
      CylinderRadius = {R:.6f};
      CylinderHeight = {2.0 * R:.6f};
      FrontSphericalCap = On;
      BackSphericalCap = On;
{state}
    }}
"""


def _initial_condition_block(cfg: M2CConfig) -> str:
    """Projectile and target as geometric entities.

    The projectile travels along +X and the target is a slab normal to it.
    For an oblique impact the *velocity* is tilted rather than the target,
    which keeps the target a simple slab and puts the asymmetry where it
    physically belongs -- in the momentum, not the mesh.
    """
    R = si_to_m2c(cfg.radius, "length")
    rho_t = si_to_m2c(cfg.target.rho0, "density")
    p0 = cfg.ambient_pressure
    target_R = 0.5 * cfg.domain_radii * R
    target_h = 8.0 * R

    return f"""under InitialCondition {{

  under GeometricEntities {{

{_projectile_entity(cfg)}
    under CylinderAndCone[1] {{ // target slab
      Axis_x = 1.0;
      Axis_y = 0.0;
      Axis_z = 0.0;
      BaseCenter_x = 0.0;
      BaseCenter_y = 0.0;
      BaseCenter_z = 0.0;
      CylinderRadius = {target_R:.6f};
      CylinderHeight = {target_h:.6f};
      ConeOpeningAngleInDegrees = 0.0;
      ConeHeight = 0.0;
      under InitialState {{
        MaterialID = 1;
        Density = {rho_t:.6e};
        VelocityX = 0.0;
        VelocityY = 0.0;
        VelocityZ = 0.0;
        Pressure = {p0:.6e};
      }}
    }}
  }}
}}
"""


def _probe_nodes(cfg: M2CConfig) -> str:
    """Probe points along and across the impact axis.

    Placed in projectile radii so the set is scale-free, and concentrated
    near the impact site where the plasma forms.
    """
    R = si_to_m2c(cfg.radius, "length")
    lines, k = [], 0
    for x in (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0):
        for y in (0.0, 1.0, 2.0, 4.0):
            lines.append(f"    under Node[{k}] {{X = {x * R:.6f}; "
                         f"Y = {y * R:.6f}; Z = 0.0;}}")
            k += 1
    return "\n".join(lines) + "\n"


def _level_set_block(cfg: M2CConfig, index: int, mat_id: int) -> str:
    axis_bc = "LinearExtrapolation" if cfg.is_3d else "ZeroNeumann"
    return f"""  under LevelSet[{index}] {{
    MaterialID = {mat_id};
    Solver = FiniteDifference;
    Bandwidth = 15;
    BoundaryConditionX0   = LinearExtrapolation;
    BoundaryConditionXmax = LinearExtrapolation;
    BoundaryConditionY0   = {axis_bc};
    BoundaryConditionYmax = LinearExtrapolation;
    BoundaryConditionZ0   = {axis_bc};
    BoundaryConditionZmax = {axis_bc};
    under Reinitialization {{
      Frequency = 2;
      MaxIts = 200;
    }}
  }}
"""


def write_input(cfg: M2CConfig, path: str) -> str:
    """Write a complete M2C `input.st`. Returns the text as well."""
    gas = _gas(cfg.ambient)
    rho_amb = cfg.ambient_density()
    R = si_to_m2c(cfg.radius, "length")

    # Material IDs follow the shipped deck: 0 ambient, 1 target, 2 projectile.
    equations = ("under Equations {\n"
                 + _stiffened_gas_block(gas, rho_amb, 0)
                 + _mie_gruneisen_block(cfg.target, 1)
                 + _mie_gruneisen_block(cfg.projectile, 2)
                 + "}\n")

    # MaxChargeNumber = the number of stages the material YAML defines, so
    # M2C solves the same ionisation ladder hvi_emp.ionization does.
    species = [(1, cfg.target.name, cfg.target.A, cfg.target.Z,
                len(cfg.target.E_ion)),
               (2, cfg.projectile.name, cfg.projectile.A, cfg.projectile.Z,
                len(cfg.projectile.E_ion))]
    if gas["ionises"]:
        species.insert(0, (0, gas["label"], gas["A"], gas["Z"],
                           AMBIENT_MAX_CHARGE))

    out_dt = cfg.t_end / max(cfg.n_outputs, 1)
    ion_on = "On" if cfg.ionisation != "off" else "Off"

    text = f"""//Units: mm, g, s, K, A
//
// Generated by hvi_emp.solvers.m2c_stage1. Edit the scenario, not this file,
// so the run stays reproducible.
//
// {cfg.projectile.name} -> {cfg.target.name}
//   diameter  {cfg.diameter * 1e3:.4g} mm   (mass {cfg.mass:.4e} kg)
//   velocity  {cfg.velocity / 1e3:.4g} km/s
//   incidence {cfg.angle_deg:g} deg from the surface normal
//   mesh      {'3-D Cartesian' if cfg.is_3d else '2-D axisymmetric'}
//   ambient   {gas['label']} at {cfg.ambient_pressure:g} Pa \
({rho_amb:.3e} kg/m3)
//
// Material IDs: 0 = ambient, 1 = target, 2 = projectile.

{_mesh_block(cfg)}
{equations}
{_ionisation_block(cfg, species)}
{_initial_condition_block(cfg)}
under BoundaryConditions {{
  under Inlet {{
    Density = {si_to_m2c(rho_amb, 'density'):.6e};
    VelocityX = 0.0;
    VelocityY = 0.0;
    VelocityZ = 0.0;
    Pressure = {cfg.ambient_pressure:.6e};
  }}
}}

under Space {{
  under NavierStokes {{
    Flux = LocalLaxFriedrichs;
    under Reconstruction {{
      Type = Linear;
      VariableType = ConservativeCharacteristic;
      Limiter = GeneralizedMinMod;
      GeneralizedMinModCoefficient = 1.1;
    }}
  }}

{_level_set_block(cfg, 0, 1)}
{_level_set_block(cfg, 1, 2)}}}

under MultiPhase {{
  Flux = Numerical;
  ReconstructionAtInterface = Constant;
  PhaseChange = RiemannSolution;
  RiemannNormal = Average;
  LevelSetCorrectionFrequency = 100;
  ConstantReconstructionDepth = {0.3 * R:.6e};
}}

under Time {{
  Type = Explicit;
  MaxTime = {cfg.t_end:.6e};
  CFL    = {cfg.cfl:.4f};
  under Explicit {{
    Type = RungeKutta2;
  }}
}}

under Output {{
  Prefix = "results/";
  Solution = "solution";
  TimeInterval = {out_dt:.6e};
  Density = On;
  Velocity = On;
  Pressure = On;
  Temperature = On;
  MaterialID = On;
  LevelSet0 = On;
  LevelSet1 = On;
  // These two are what hvi_emp Stage 3 consumes: the mean charge and the
  // electron number density are exactly the plume state emp.py needs.
  MeanCharge = {ion_on};
  ElectronDensity = {ion_on};

  under Probes {{
    Frequency = 10;
    Pressure = "pressure_probes.txt";
    Density = "density_probes.txt";
    Temperature = "temperature_probes.txt";
    IonizationResult = "ionization_probes.txt";

{_probe_nodes(cfg)}  }}
  VerboseScreenOutput = Low;
}}
"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return text


# ---------------------------------------------------------------------------
# Grammar verification against a real checkout
# ---------------------------------------------------------------------------

def check_grammar(m2c_root: str) -> dict:
    """Check every keyword this module emits against an M2C source tree.

    The answer to "does M2C actually accept this deck?" is a grep against the
    code that parses it, not a confident assertion. This scans `IoData.h` /
    `IoData.cpp` and any shipped `*.st` for each keyword and reports what is
    missing -- which, once the keywords are right, means it is watching for
    an upstream rename.

    Returns ``{"found", "missing", "sources", "inferred_ok",
    "inferred_missing"}``. A keyword in `missing` means the generated deck
    will be rejected or, worse, silently ignored -- fix the spelling first.
    """
    if not os.path.isdir(m2c_root):
        raise FileNotFoundError(f"not a directory: {m2c_root}")
    sources, blob = [], []
    for dirpath, dirnames, filenames in os.walk(m2c_root):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "build", "AtomicData")]
        for fn in filenames:
            # Every header and source, not just IoData.*: the assigner that
            # registers the ionisation keywords and their enum tokens does
            # not live in IoData.cpp, so the narrower scan could not see it.
            if fn.endswith((".h", ".cpp", ".st")):
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, "r", errors="replace") as fh:
                        blob.append(fh.read())
                except OSError:
                    continue
                sources.append(p)
    if not sources:
        raise FileNotFoundError(
            f"found no IoData.h/.cpp or *.st under {m2c_root} -- is this an "
            f"M2C checkout? Expected e.g. "
            f"{os.path.join(m2c_root, 'IoData.h')}")
    text = "\n".join(blob)
    found, missing = [], []
    for kw in tuple(VERIFIED_KEYWORDS) + tuple(INFERRED_KEYWORDS):
        (found if re.search(rf"\b{re.escape(kw)}\b", text)
         else missing).append(kw)

    # Enum VALUES, not just keys. Checking keys alone gave false
    # confidence: a deck reported "133 of 133 keywords found" and then
    # aborted, because `PartitionFunctionEvaluation` was accepted while
    # the value assigned to it was never examined at all.
    #
    # Matched as a quoted string literal, which is how ClassToken
    # registers them -- a bare-word search would also hit the C++ enum
    # identifier and comments, and so could pass on a token the parser
    # does not accept.
    values_found, values_missing = [], []
    for val in VERIFIED_VALUES:
        (values_found if f'"{val}"' in text
         else values_missing).append(val)

    return {
        "found": found,
        "missing": missing,
        "values_found": values_found,
        "values_missing": values_missing,
        "sources": sources,
        "inferred_ok": [k for k in INFERRED_KEYWORDS if k in found],
        "inferred_missing": [k for k in INFERRED_KEYWORDS if k in missing],
    }


def find_m2c(roots=None) -> str | None:
    """Locate an M2C checkout or build.

    ``$M2C_HOME`` is authoritative when set: pointing it somewhere wrong is
    reported rather than papered over by silently finding a different copy,
    which is the rule the other bridges follow and for the same reason.
    """
    def classify(root):
        """(state, text) where state is 'built' | 'source' | None."""
        if not root:
            return None, None
        if not os.path.isdir(root):
            # A common slip is to point M2C_HOME at the executable itself.
            # Returning None for that says "not found", which sends people
            # looking for a build problem they do not have.
            if os.path.isfile(root) and os.path.basename(root) == "m2c":
                return None, (f"{root} is the executable -- M2C_HOME must be "
                              f"the directory holding it: "
                              f"{os.path.dirname(root)}")
            return None, None
        exe = os.path.join(root, "m2c")
        if os.path.isfile(exe) and os.access(exe, os.X_OK):
            return "built", f"{root} (built)"
        if (os.path.isfile(os.path.join(root, "Main.cpp"))
                or os.path.isfile(os.path.join(root, "IoData.h"))):
            return "source", f"{root} (source only, not compiled)"
        return None, None

    if roots is None:
        env = os.environ.get("M2C_HOME")
        if env:
            return classify(env)[1]
        roots = [os.path.join(os.getcwd(), "m2c"),
                 os.path.expanduser("~/m2c"),
                 os.path.expanduser("~/src/m2c"),
                 os.path.expanduser("~/src/m2c/build")]

    # Prefer a built tree over a source-only one *wherever* it sits in the
    # list. Returning the first hit reported a compiled M2C as "not
    # compiled" whenever the source root was searched before its own build/
    # subdirectory -- which is the normal layout, so this was the common
    # case, not an edge case.
    first_source = None
    for r in roots:
        state, text = classify(r)
        if state == "built":
            return text
        if state == "source" and first_source is None:
            first_source = text
    return first_source


# ---------------------------------------------------------------------------
# Reading output back
# ---------------------------------------------------------------------------

def read_probes(path: str) -> dict:
    """Read one M2C probe file into arrays.

    M2C writes probe files as whitespace-separated columns with a header
    line. Which columns appear depends on what the Output block asked for, so
    the header is parsed rather than assumed -- a bridge that assumes a column
    order fails silently the moment the deck changes.

    Returns ``{"t", "values", "columns", "raw"}`` in M2C units.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    header, rows = None, []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#") or s[0].isalpha():
                if header is None:
                    header = s.lstrip("# ").split()
                continue
            try:
                rows.append([float(x) for x in s.split()])
            except ValueError:
                continue
    if not rows:
        return {"t": np.zeros(0), "values": np.zeros((0, 0)),
                "columns": header or [], "raw": np.zeros((0, 0))}
    width = min(len(r) for r in rows)
    a = np.array([r[:width] for r in rows], dtype=float)
    return {"t": a[:, 0], "values": a[:, 1:], "columns": header or [],
            "raw": a}


def read_ionization_probes(path: str) -> dict:
    """Read `ionization_probes.txt` and convert the densities to SI.

    M2C reports the electron number density in its own units (1/mm^3), so it
    is divided by 1e-9 to reach 1/m^3 -- the same conversion, and the same
    trap, as everywhere else in this module.
    """
    pr = read_probes(path)
    pr["n_e"] = (m2c_to_si(pr["values"], "number_density")
                 if pr["values"].size else pr["values"])
    return pr


def to_plume_state(probes: dict, material: Material) -> dict:
    """Reduce M2C probe output to the handful of numbers Stage 3 needs.

    This is the join between the two codes: M2C computes the plasma, and
    `hvi_emp` takes the peak electron density and when it occurs into the EMP
    stage. Kept deliberately small -- if the interface is one dict of
    scalars, either side can be replaced without touching the other.
    """
    n_e = np.asarray(probes.get("n_e", np.zeros(0)), dtype=float)
    if n_e.size == 0 or not np.isfinite(n_e).any():
        raise ValueError(
            "no finite electron-density data in these probes; check that the "
            "deck had `ElectronDensity = On` and that the run got past t = 0")
    i_t = int(np.unravel_index(int(np.nanargmax(n_e)), n_e.shape)[0])
    t = np.asarray(probes.get("t", np.zeros(0)), dtype=float)
    return {
        "n_e_peak_m3": float(np.nanmax(n_e)),
        "t_peak_s": float(t[i_t]) if t.size > i_t else float("nan"),
        "n_samples": int(n_e.shape[0]),
        "material": material.name,
        "source": "M2C",
    }


# ---------------------------------------------------------------------------
# Problem directory
# ---------------------------------------------------------------------------

def build_command(cores: int = 64) -> str:
    """The command that runs a written problem directory."""
    return (f"mpirun -np {cores} "
            # :? so an unset M2C_HOME aborts with a readable message
            # instead of silently launching the nonexistent "/m2c".
            f"{M2C_HOME_GUARD}/m2c input.st")


def write_problem_directory(cfg: M2CConfig, directory: str,
                            cores: int = 64) -> dict:
    """Write `input.st` plus a README. Returns paths, command and estimate."""
    os.makedirs(directory, exist_ok=True)
    paths = {"input": os.path.join(directory, "input.st"),
             "readme": os.path.join(directory, "README.txt")}
    write_input(cfg, paths["input"])

    # M2C does not create its own output directory. The deck sets
    # `Prefix = "results/"`, and without the directory the run initialises
    # fully -- mesh, level sets, both Saha solvers, all 24 probes -- and
    # only then dies with "Cannot open file 'results/density_probes.txt'".
    # Minutes of setup wasted on a missing mkdir.
    os.makedirs(os.path.join(directory, "results"), exist_ok=True)

    # Write the atomic data the deck references. M2C ships none, so without
    # this the run dies at startup on the first Saha setup -- and telling
    # the user to "symlink M2C's AtomicData directory" was wrong advice,
    # because no such directory exists in the repository.
    if cfg.ionisation != "off":
        atomic_dir = os.path.join(directory, cfg.atomic_data_dir)
        paths["atomic_data"] = atomic_dir
        paths["atomic_files"] = write_atomic_data(
            atomic_dir, [cfg.target, cfg.projectile])
    est = cfg.estimate_resources(cores)
    notes = "\n".join(f"  ! {n}" for n in cfg.notes) or "  (none)"
    depression = (f" ({cfg.depression_model} continuum lowering)"
                  if cfg.ionisation == "non-ideal" else "")

    with open(paths["readme"], "w") as fh:
        fh.write(f"""M2C hypervelocity-impact problem
================================
Generated by hvi_emp.solvers.m2c_stage1

{cfg.projectile.name} -> {cfg.target.name}
  diameter        {cfg.diameter * 1e3:.4g} mm  (mass {cfg.mass:.4e} kg)
  velocity        {cfg.velocity / 1e3:.4g} km/s
  incidence       {cfg.angle_deg:g} deg from normal
  mesh            {est['dimensionality']}
  ambient         {_gas(cfg.ambient)['label']} at {cfg.ambient_pressure:g} Pa
  ionisation      {cfg.ionisation}{depression}

Install M2C (once)
------------------
  git clone https://github.com/kevinwgy/m2c.git $HOME/src/m2c
  cd $HOME/src/m2c && mkdir -p build && cd build
  cmake .. && make -j
  export M2C_HOME=$HOME/src/m2c/build

M2C needs PETSc and MPI. It is GPLv3 -- unlike iSALE there is no application
process and no wait.

Verify the input grammar (cheap, worth doing once)
--------------------------------------------------
  python -m hvi_emp.solvers.m2c_stage1 --check $HOME/src/m2c

Every keyword in this deck was traced to M2C's shipped HVI input file or to
IoData.cpp, its parser. The check re-greps the parser and reports anything it
no longer contains, which catches an upstream rename before it becomes a run
that silently ignores your projectile.

The Ionization blocks reference atomic data files under {cfg.atomic_data_dir}/,
which has been generated alongside this deck -- M2C ships none, so a run
without it aborts with "Cannot open ionization energy file". The values come
from the hvi_emp material YAML (NIST ASD), so M2C and the reduced chain solve
the same ionisation ladder. See {cfg.atomic_data_dir}/README.txt for the
ground-state approximation used and how to replace it with full NIST levels.

Run
---
  {build_command(cores)}

Estimated cost -- ORDER OF MAGNITUDE ONLY. Derived from the cell count and an
explicit CFL step, not measured:
  cells           ~{est['cells']:.3e}
  fine cell size  {est['cell_size_m'] * 1e3:.4g} mm
  time step       {est['dt_s']:.3e} s
  steps           ~{est['n_steps']:.3e}
  memory          ~{est['memory_GB']:.1f} GB
  wall time       ~{est['wall_hours_estimate']:.1f} h on {cores} cores

Notes
-----
{notes}

Reading the output back
-----------------------
  from hvi_emp.solvers.m2c_stage1 import read_ionization_probes, \\
      to_plume_state
  pr = read_ionization_probes("results/ionization_probes.txt")
  print(to_plume_state(pr, target_material))

M2C stops at the plasma state. Charge separation and the radiated field
remain hvi_emp Stage 3 (emp.py) or a PIC run (picongpu_stage3) -- see
docs/EMP_UNCERTAINTY.md for why that last step is the uncertain one.
""")
    return {"paths": paths, "command": build_command(cores),
            "estimate": est, "notes": list(cfg.notes)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv=None) -> int:      # pragma: no cover - thin CLI wrapper
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = ("usage:\n"
             "  python -m hvi_emp.solvers.m2c_stage1 --check <M2C_ROOT>\n"
             "  python -m hvi_emp.solvers.m2c_stage1 --find")
    if not argv or argv[0] not in ("--check", "--find"):
        print(usage)
        return 0
    if argv[0] == "--find":
        got = find_m2c()
        print(got or "M2C not found. Set $M2C_HOME or clone it:\n"
                     "  git clone https://github.com/kevinwgy/m2c.git")
        return 0 if got else 1
    root = argv[1] if len(argv) > 1 else os.environ.get("M2C_HOME", "")
    if not root:
        print("--check needs a path, or $M2C_HOME set")
        return 2
    try:
        res = check_grammar(root)
    except FileNotFoundError as exc:
        print(exc)
        return 2
    print(f"scanned {len(res['sources'])} file(s) under {root}")
    print(f"  {len(res['found'])} of "
          f"{len(VERIFIED_KEYWORDS) + len(INFERRED_KEYWORDS)} keywords found")
    if res["missing"]:
        print("  MISSING -- fix these before running a generated deck:")
        for kw in res["missing"]:
            tag = (" (never verified)" if kw in INFERRED_KEYWORDS
                   else " (was verified -- M2C's input format may have moved)")
            print(f"    {kw}{tag}")
        return 1
    print("  all present; this checkout accepts the generated grammar")
    return 0


if __name__ == "__main__":        # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "M2CConfig", "config_from_scenario", "TO_M2C", "M2C_CONSTANTS",
    "AMBIENT_GASES", "si_to_m2c", "m2c_to_si", "write_input",
    "write_problem_directory", "build_command", "find_m2c", "check_grammar",
    "VERIFIED_KEYWORDS", "INFERRED_KEYWORDS", "read_probes",
    "read_ionization_probes", "to_plume_state",
]
