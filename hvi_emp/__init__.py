"""HVI-EMP -- hypervelocity impact plasma and electromagnetic pulse framework.

A physically-traceable, four-stage reduced-order model of what happens when a
piece of orbital debris or a meteoroid strikes a spacecraft:

    Stage 1  `impact`      shock, release, vaporisation, ionisation inventory
    Stage 1b `jetting`     shaped-charge jetting: the fast, sparse material
                           the bulk shock model misses (Jean & Rollins 1970)
    Stage 2  `expansion`   plume expansion into vacuum, freeze-out, transitions
    Stage 3  `emp`         charge separation, coherent oscillation, radiation
    Stage 4  `coupling`    surface charging, ESD, induced currents

Supporting modules: `eos` (Hugoniot / Vinet cold curve / release),
`ionization` (multi-stage Saha, continuum lowering, rate coefficients),
`propagation` (plasma cutoff, dust scattering), `validation` (comparisons with
published data), `pipeline` (end-to-end driver).

Quick start
-----------
>>> from hvi_emp import run_scenario
>>> sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3)
>>> print(sc.summary())

Everything is SI internally.  Every stage exposes its assumptions as named
parameters; see `docs/THEORY.md` for the derivations and the validity limits.
"""

from .constants import *                     # noqa: F401,F403
from .materials import (ALUMINIUM, COPPER, DOLOMITE, GLASS, IRON, KAPTON,
                        OLIVINE, TUNGSTEN, Material, get_material,
                        list_materials)
from .eos import (ImpactState, hugoniot_state, impedance_match,
                  phase_thresholds, release_state, residual_energy,
                  vapour_fraction)
from .ionization import (IonisationState, IonisationTable,
                         collision_frequency_ei, coupling_parameter,
                         debye_length, get_table, plasma_frequency,
                         recombination_time, saha_solve)
from .impact import (ImpactResult, Projectile, available_empirical_laws,
                     empirical_charge_yield, simulate_impact)
from .jetting import (JetState, critical_angle, downrange_threshold,
                      jet_velocity)
from .expansion import (ExpansionResult, simulate_expansion,
                        stopping_distance, torr_to_number_density)
from .emp import (ChargeSeparation, EMPResult, dipole_field,
                  emission_at_frequency, resonant_density, simulate_emp)
from .propagation import (cutoff_frequency, dust_optical_depth,
                          refractive_index, transmission)
from .coupling import (ChargingResult, CouplingResult, UPSET_THRESHOLDS,
                       anomaly_assessment, couple_to_structure,
                       floating_potential, simulate_charging)
from .pipeline import Scenario, run_scenario, velocity_sweep

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # materials
    "Material", "get_material", "list_materials",
    "ALUMINIUM", "IRON", "TUNGSTEN", "COPPER", "GLASS", "KAPTON",
    "OLIVINE", "DOLOMITE",
    # eos
    "ImpactState", "hugoniot_state", "impedance_match", "release_state",
    "residual_energy", "phase_thresholds", "vapour_fraction",
    # ionisation
    "IonisationState", "IonisationTable", "saha_solve", "get_table",
    "plasma_frequency", "debye_length", "collision_frequency_ei",
    "recombination_time", "coupling_parameter",
    # stages
    "Projectile", "ImpactResult", "simulate_impact",
    "JetState", "critical_angle", "jet_velocity", "downrange_threshold",
    "empirical_charge_yield", "available_empirical_laws",
    "ExpansionResult", "simulate_expansion", "stopping_distance",
    "torr_to_number_density",
    "ChargeSeparation", "EMPResult", "simulate_emp", "dipole_field",
    "emission_at_frequency", "resonant_density",
    "refractive_index", "cutoff_frequency", "transmission",
    "dust_optical_depth",
    "ChargingResult", "CouplingResult", "simulate_charging",
    "couple_to_structure", "floating_potential", "anomaly_assessment",
    "UPSET_THRESHOLDS",
    # pipeline
    "Scenario", "run_scenario", "velocity_sweep",
]
