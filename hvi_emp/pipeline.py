"""End-to-end driver: impact -> plasma -> expansion -> EMP -> spacecraft.

`run_scenario` wires the four stages together with sane defaults and returns a
single `Scenario` object holding every intermediate result, so any stage can be
inspected or re-run with different parameters without repeating the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .coupling import (ChargingResult, CouplingResult, anomaly_assessment,
                       couple_to_structure, simulate_charging)
from .emp import EMPResult, emission_at_frequency, simulate_emp
from .expansion import ExpansionResult, simulate_expansion
from .impact import ImpactResult, Projectile, empirical_charge_yield, simulate_impact
from .materials import Material, get_material
from .parallel import parallel_map

#: Ceiling on the automatically-chosen expansion window [s].  Long enough for
#: a plume to become tenuous enough to radiate at a few hundred MHz, short
#: enough that a scenario still runs in seconds.
MAX_AUTO_T_END = 1.0e-3


@dataclass
class Scenario:
    """Complete impact-to-anomaly result set."""
    impact: ImpactResult
    expansion: ExpansionResult | None = None
    emp: EMPResult | None = None
    charging: ChargingResult | None = None
    coupling: CouplingResult | None = None
    anomaly: dict = field(default_factory=dict)
    bands: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        parts = [self.impact.summary()]
        if self.expansion is not None:
            parts += ["", self.expansion.summary()]
        if self.emp is not None:
            parts += ["", self.emp.summary()]
        if self.bands:
            parts += ["", "Narrowband response (resonant-shell model):"]
            for f, d in sorted(self.bands.items()):
                if d.get("reached"):
                    parts.append(
                        f"  {f/1e6:8.1f} MHz  E = {d['E_peak']:.3e} V/m  "
                        f"(n_e = {d['n_e']:.3e} m^-3, t = {d['t']:.3e} s, "
                        f"kr = {d['kr']:.2f})")
                else:
                    parts.append(f"  {f/1e6:8.1f} MHz  resonant density never "
                                 "reached")
        if self.charging is not None:
            parts += ["", self.charging.summary()]
        if self.coupling is not None:
            parts += ["", self.coupling.summary()]
        if self.anomaly:
            parts += ["", f"Anomaly verdict: {self.anomaly['verdict']}"]
            for p in self.anomaly["pathways"]:
                parts.append(f"  - {p}")
        if self.warnings:
            parts += ["", "Validity warnings:"]
            parts += [f"  ! {w}" for w in self.warnings]
        return "\n".join(parts)


def run_scenario(projectile_material: str | Material,
                 target_material: str | Material,
                 mass: float,
                 velocity: float,
                 angle_deg: float = 0.0,
                 *,
                 n_decay: float = 2.0,
                 core_scale: float = 1.0,
                 plume_expansion_factor: float = 3.0,
                 t_end: float | None = None,
                 expansion_model: str = "1T",
                 aspect0: float = 0.5,
                 r_sensor: float = 0.30,
                 theta_deg: float = 90.0,
                 f_max: float = 5.0e9,
                 separation_factor: float = 1.0,
                 closure: str = "debye",
                 bands: tuple = (315e6, 916e6),
                 standoff: float = 0.10,
                 loop_area: float = 1.0e-2,
                 wire_length: float = 0.10,
                 shielding_dB: float = 40.0,
                 verbose: bool = False) -> Scenario:
    """Run the full chain for one impact.

    Parameters
    ----------
    projectile_material, target_material : str | Material
        Names from `materials.list_materials()` or Material objects.
    mass : float
        Projectile mass [kg].
    velocity : float
        Impact speed [m/s].
    angle_deg : float
        Incidence angle from the surface normal [deg].
    expansion_model : {"1T", "2T"}
        Stage-2 closure. ``"1T"`` forces T_e = T_i and applies freeze-out as a
        post-process; ``"2T"`` evolves T_e and T_i separately with
        non-equilibrium ionisation.

    aspect0 : float
        Initial plume R_z/R_r. Calibrated against the measured cosine
        angular law, not derived; 1.0 gives an isotropic plume, which the
        measurement excludes. See `expansion.simulate_expansion`.

        ``"2T"`` is the more defensible physics -- electrons and ions
        demonstrably decouple during the freeze-out window -- and it brings
        the charge-yield exponent from 6.70 to 3.65 against 3.48 measured,
        which is the framework's largest open discrepancy. It is **not** the
        default only because it is ~5x slower and changes every downstream
        number; see `docs/PHYSICS_AUDIT.md`. Prefer it for quantitative work.

    Returns
    -------
    Scenario
    """
    if expansion_model not in ("1T", "2T"):
        raise ValueError(f"expansion_model must be '1T' or '2T', "
                         f"got {expansion_model!r}")
    if expansion_model == "2T":
        from .expansion_2t import simulate_expansion_2t as _expand
    else:
        _expand = simulate_expansion
    pm = (get_material(projectile_material)
          if isinstance(projectile_material, str) else projectile_material)
    tm = (get_material(target_material)
          if isinstance(target_material, str) else target_material)

    proj = Projectile(pm, mass=mass, velocity=velocity, angle_deg=angle_deg)
    imp = simulate_impact(proj, tm, n_decay=n_decay, core_scale=core_scale,
                          plume_expansion_factor=plume_expansion_factor,
                          warn=False)
    sc = Scenario(impact=imp)

    # --- validity flags ---------------------------------------------------
    if proj.v_normal > min(pm.us_up_valid_to, tm.us_up_valid_to) * 2.0:
        sc.warnings.append(
            f"impact speed {velocity/1e3:.0f} km/s is far beyond the validated "
            "linear Us-Up range; substitute a tabular (SESAME/ANEOS) EOS for "
            "quantitative work")
    if angle_deg > 60.0:
        sc.warnings.append(
            f"incidence {angle_deg:.0f} deg from normal: the normal-component "
            "approximation degrades beyond ~60 deg")
    if imp.m_plasma <= 0:
        sc.warnings.append(
            "no superheated plasma produced -- below the complete-vaporisation "
            "threshold for this material pair; only weakly-ionised two-phase "
            "vapour is present and no EMP is predicted")
        return sc

    # --- Stage 2 ----------------------------------------------------------
    sc.expansion = _expand(imp, t_end=t_end, aspect0=aspect0)

    # If the caller did not pin t_end, make sure the integration window is
    # long enough for the plume to reach the resonant density of the lowest
    # requested antenna band.  Without this the narrowband comparison -- the
    # headline result -- silently reports "resonant density never reached",
    # because the auto-scaled default window stops while the plume is still
    # orders of magnitude too dense.  Asymptotically the plume is ballistic,
    # so n_e ~ t^-3 and the required window follows in closed form.
    if t_end is None and bands:
        from .emp import resonant_density
        n_needed = min(resonant_density(f) for f in bands)
        n_final = float(sc.expansion.n_e[-1])
        if n_final > n_needed > 0:
            t_now = float(sc.expansion.t[-1])
            t_req = t_now * (n_final / n_needed) ** (1.0 / 3.0) * 3.0
            t_req = min(t_req, MAX_AUTO_T_END)
            if t_req > t_now:
                sc.expansion = _expand(imp, t_end=t_req,
                                       aspect0=aspect0)
                if float(sc.expansion.n_e[-1]) > n_needed:
                    sc.warnings.append(
                        f"integration window capped at {MAX_AUTO_T_END:.1e} s; "
                        f"the plume never reaches the resonant density for "
                        f"{min(bands)/1e6:.0f} MHz, so that band reports no "
                        "emission. Pass an explicit t_end to extend it.")
    if sc.expansion.Gamma_coupling[0] > 1.0:
        sc.warnings.append(
            f"initial coupling parameter Gamma = "
            f"{sc.expansion.Gamma_coupling[0]:.2f} > 1: the plasma starts "
            "strongly coupled, where ideal Saha and the Spitzer collision "
            "rate are both approximations")
    if not sc.expansion.transition:
        sc.warnings.append(
            "no collisional->collisionless transition inside the integration "
            "window; increase t_end")
        return sc

    # --- Stage 3 ----------------------------------------------------------
    sc.emp = simulate_emp(sc.expansion, r_sensor=r_sensor,
                          theta_deg=theta_deg, f_max=f_max,
                          separation_factor=separation_factor,
                          closure=closure)
    sc.bands = {f: emission_at_frequency(
        sc.expansion, f, r_sensor=r_sensor, theta_deg=theta_deg,
        separation_factor=separation_factor, closure=closure) for f in bands}

    # --- Stage 4 ----------------------------------------------------------
    sc.charging = simulate_charging(sc.expansion, standoff=standoff)
    sc.coupling = couple_to_structure(sc.emp, loop_area=loop_area,
                                      wire_length=wire_length,
                                      shielding_dB=shielding_dB)
    sc.anomaly = anomaly_assessment(sc.charging, sc.coupling)

    if verbose:
        print(sc.summary())
    return sc


def _sweep_point(arg):
    """One scenario, reduced to the scalars a sweep reports.

    Module-level so it can be pickled to a worker process, and returning
    plain floats rather than the `Scenario` so that only ~50 bytes cross the
    process boundary instead of the whole result set (every array, every
    diagnostic). For a 40-point 2T sweep that is the difference between a
    few kB and a few hundred MB of pickling.
    """
    pm, tm, mass, vv, kw = arg
    sc = run_scenario(pm, tm, mass, vv, **kw)
    d = sc.bands.get(916e6, {}) if sc.bands else {}
    return (float(sc.impact.Q_free),
            float(sc.expansion.Q_final) if sc.expansion is not None else 0.0,
            float(sc.impact.plasma.T_eV),
            float(sc.impact.plasma.Zbar),
            float(sc.impact.vaporisation_efficiency),
            float(d.get("E_peak", 0.0)) if d.get("reached") else 0.0)


def velocity_sweep(projectile_material: str | Material,
                   target_material: str | Material,
                   mass: float,
                   velocities: np.ndarray,
                   workers: int | None = None,
                   **kw) -> dict:
    """Charge yield and EMP amplitude versus impact speed.

    Returns arrays suitable for the classic log-log Q(v) plot, together with
    the empirical scaling law for comparison.

    Parameters
    ----------
    workers
        Scenarios are independent, so the sweep is run across processes.
        ``None`` uses every available core, ``1`` forces the serial path.
        This is where a many-core machine pays: a single expansion is an
        inherently serial ODE, but forty of them are forty independent
        serial ODEs.

    Notes
    -----
    A scenario that raises is recorded as zero rather than aborting the
    sweep -- non-convergence at one velocity should not cost the other
    thirty-nine -- but the count is returned under ``"n_failed"`` so the
    failure is visible instead of silently looking like a zero yield.
    """
    v = np.asarray(velocities, dtype=float)
    args = [(projectile_material, target_material, mass, float(vv), kw)
            for vv in v]
    res = parallel_map(_sweep_point, args, workers=workers, on_error="skip")

    cols = np.zeros((6, len(v)))
    for i, r in enumerate(res):
        if r is not None:
            cols[:, i] = r
    Q1, Qf, Te, Zb, mvap, E916 = cols
    return {"v": v, "Q_stage1": Q1, "Q_frozen": Qf, "T_eV": Te, "Zbar": Zb,
            "vapour_efficiency": mvap, "E_916MHz": E916,
            "n_failed": res.n_failed,
            "Q_empirical": np.array(
                [empirical_charge_yield(mass, vv) for vv in v])}


__all__ = ["Scenario", "run_scenario", "velocity_sweep", "MAX_AUTO_T_END"]
