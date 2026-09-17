"""Command-line interface.

Run a scenario, a velocity sweep, or the validation suite without writing any
Python:

    python -m hvi_emp run     --projectile Fe --target Al --mass 1e-12 --velocity 50e3
    python -m hvi_emp sweep   --projectile Fe --target Al --mass 1e-12 --vmin 20e3 --vmax 72e3
    python -m hvi_emp doctor          <- start here on a new machine
    python -m hvi_emp validate
    python -m hvi_emp materials
    python -m hvi_emp lammps  --material W --velocity 9e3 --out in.impact

Results can be written to JSON with `--json out.json` for downstream analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _jsonable(obj):
    """Recursively convert numpy/dataclass content to JSON-safe types."""
    import dataclasses
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name))
                for f in dataclasses.fields(obj)}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _scenario_digest(sc) -> dict:
    """Compact, JSON-safe summary of a Scenario (no full time series)."""
    d = {
        "impact": {
            "projectile_material": sc.impact.projectile.material.name,
            "target_material": sc.impact.target.name,
            "mass_kg": sc.impact.projectile.mass,
            "velocity_m_s": sc.impact.projectile.velocity,
            "angle_deg": sc.impact.projectile.angle_deg,
            "radius_m": sc.impact.projectile.radius,
            "kinetic_energy_J": sc.impact.projectile.kinetic_energy,
            "P_shock_Pa": sc.impact.P_ic,
            "m_vapour_kg": sc.impact.m_vapour,
            "m_plasma_kg": sc.impact.m_plasma,
            "m_melt_target_kg": sc.impact.m_melt_target,
            "T_e_eV": sc.impact.plasma.T_eV,
            "Zbar": sc.impact.plasma.Zbar,
            "n_e_initial_m3": sc.impact.plasma.n_e,
            "Q_formation_C": sc.impact.Q_free,
        },
        "warnings": list(sc.warnings),
    }
    if sc.expansion is not None:
        d["expansion"] = {
            "Q_frozen_C": sc.expansion.Q_final,
            "Zbar_frozen": float(sc.expansion.Zbar[-1]),
            "v_z_asymptotic_m_s": float(sc.expansion.v_z[-1]),
            "R_final_m": float(sc.expansion.R_r[-1]),
            "freeze": _jsonable({k: v for k, v in sc.expansion.freeze.items()
                                 if not isinstance(v, np.ndarray)}),
            "transition": _jsonable(sc.expansion.transition),
        }
    if sc.emp is not None:
        d["emp"] = {
            "r_sensor_m": sc.emp.r_sensor,
            "theta_deg": sc.emp.theta_deg,
            "peak_E_V_per_m": sc.emp.peak_field,
            "peak_B_T": sc.emp.peak_field / 2.99792458e8,
            "energy_radiated_J": sc.emp.energy_radiated,
            "spectral_peak_Hz": float(sc.emp.f[int(np.argmax(sc.emp.psd))]),
            "band_100_500MHz_V_per_m": sc.emp.band_field(1e8, 5e8),
            "band_0p5_2GHz_V_per_m": sc.emp.band_field(5e8, 2e9),
        }
    if sc.bands:
        d["narrowband"] = {f"{f/1e6:.0f}MHz": _jsonable(v)
                           for f, v in sc.bands.items()}
    if sc.charging is not None:
        d["charging"] = {
            "peak_differential_V": sc.charging.peak_differential_V,
            "esd_risk": sc.charging.esd_risk,
            "peak_floating_potential_V": float(np.min(sc.charging.phi_float)),
        }
    if sc.coupling is not None:
        d["coupling"] = {
            "V_loop_peak": sc.coupling.V_loop_peak,
            "V_monopole_peak": sc.coupling.V_monopole_peak,
            "shielded_V": sc.coupling.shielded_V,
            "received_power_dBm": sc.coupling.received_power_dBm,
            "upset_risk": sc.coupling.upset_risk,
        }
    if sc.anomaly:
        d["anomaly"] = _jsonable(sc.anomaly)
    return d


# ---------------------------------------------------------------------------
# sub-commands
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    from .pipeline import run_scenario
    sc = run_scenario(
        args.projectile, args.target, mass=args.mass, velocity=args.velocity,
        angle_deg=args.angle, n_decay=args.n_decay,
        core_scale=args.core_scale,
        plume_expansion_factor=args.plume_expansion_factor,
        t_end=args.t_end, r_sensor=args.r_sensor, theta_deg=args.theta,
        separation_factor=args.separation_factor, closure=args.closure,
        bands=tuple(args.bands), standoff=args.standoff,
        loop_area=args.loop_area, wire_length=args.wire_length,
        shielding_dB=args.shielding_db)
    print(sc.summary())
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(_scenario_digest(sc), fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


def cmd_sweep(args) -> int:
    from .pipeline import velocity_sweep
    v = np.linspace(args.vmin, args.vmax, args.n)
    res = velocity_sweep(args.projectile, args.target, mass=args.mass,
                         velocities=v, t_end=args.t_end,
                         closure=args.closure)
    print(f"{'v [km/s]':>10}{'m_vap/m_p':>12}{'T_e [eV]':>11}{'Zbar':>9}"
          f"{'Q_frozen [C]':>15}{'E@916MHz [V/m]':>17}")
    for i in range(len(v)):
        print(f"{v[i]/1e3:>10.2f}{res['vapour_efficiency'][i]:>12.3f}"
              f"{res['T_eV'][i]:>11.3f}{res['Zbar'][i]:>9.4f}"
              f"{res['Q_frozen'][i]:>15.4e}{res['E_916MHz'][i]:>17.4e}")
    good = (res["Q_frozen"] > 0) & (v >= 0.6 * args.vmax)
    if good.sum() >= 3:
        beta = np.polyfit(np.log(v[good]), np.log(res["Q_frozen"][good]), 1)[0]
        print(f"\nasymptotic charge-yield exponent beta = {beta:.2f} "
              "(measured 3.4-3.5; see docs/THEORY.md Sec. 6 for why this is "
              "range- and definition-dependent)")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(_jsonable(res), fh, indent=2)
        print(f"wrote {args.json}")
    return 0


def cmd_validate(args) -> int:
    from .validation import run_all
    res = run_all(verbose=True)
    return 0 if res["n_pass"] == res["n_total"] else 1


def cmd_lammps(args) -> int:
    """Generate (and optionally run) a LAMMPS MD impact deck."""
    from .solvers import lammps_diagnostics
    from .solvers.lammps_stage1 import (LammpsImpactConfig,
                                        generate_input_deck, run_impact)

    cfg = LammpsImpactConfig(material=args.material, velocity=args.velocity,
                             projectile_radius=args.radius,
                             damping_walls=args.damping_walls)
    print(f"# {args.material} sphere r = {args.radius*1e9:.2f} nm at "
          f"{args.velocity/1e3:.1f} km/s")
    print(f"# {cfg.estimated_atom_count():,} atoms, "
          f"{cfg.ke_per_atom_eV:.2f} eV/atom, cells {cfg.resolved_cells()}")

    deck = generate_input_deck(cfg)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(deck)
        print(f"# wrote {args.out}\n"
              f"#   run it:  mpirun -np <N> lmp -in {args.out}")
    elif not args.run:
        print(deck)

    if args.run:
        diag = lammps_diagnostics()
        if not diag["ok"]:
            print(f"\nERROR: cannot run LAMMPS in-process.\n"
                  f"  reason: {diag['reason']}\n"
                  f"{diag['hint'].rstrip()}\n"
                  "The generated deck is unaffected and can be run with the "
                  "standalone `lmp` executable.")
            return 1
        print(f"# {diag['reason']}")
        print(run_impact(cfg).summary())
    return 0


def cmd_doctor(args) -> int:
    """Report what this machine can run."""
    from .doctor import report
    return report(skip_self_test=args.quick)


def cmd_materials(args) -> int:
    from .eos import phase_thresholds
    from .materials import get_material, list_materials
    print(f"{'name':<10}{'rho0':>9}{'c0':>8}{'s':>7}{'Gamma0':>8}"
          f"{'A':>8}{'chi_I [eV]':>12}{'P_melt':>9}{'P_vap':>9}   (GPa)")
    for n in list_materials():
        m = get_material(n)
        th = phase_thresholds(m)
        print(f"{m.name:<10}{m.rho0:>9.0f}{m.c0:>8.0f}{m.s:>7.3f}"
              f"{m.gamma0:>8.2f}{m.A:>8.2f}"
              f"{m.E_ion[0]/1.602176634e-19:>12.3f}"
              f"{th['P_melt']/1e9:>9.0f}{th['P_vap']/1e9:>9.0f}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hvi-emp",
        description="Hypervelocity impact plasma and EMP simulation.")
    p.add_argument("--quiet", action="store_true",
                   help="suppress physics-validity RuntimeWarnings")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(q):
        q.add_argument("--projectile", default="Fe", help="projectile material")
        q.add_argument("--target", default="Al", help="target material")
        q.add_argument("--mass", type=float, default=1e-12,
                       help="projectile mass [kg]")
        q.add_argument("--t-end", dest="t_end", type=float, default=None,
                       help="expansion integration end time [s]")
        q.add_argument("--closure", default="debye",
                       choices=["debye", "fletcher"],
                       help="charge-separation closure")
        q.add_argument("--json", default=None, help="write results to JSON")

    r = sub.add_parser("run", help="run one full scenario")
    add_common(r)
    r.add_argument("--velocity", type=float, default=50e3,
                   help="impact speed [m/s]")
    r.add_argument("--angle", type=float, default=0.0,
                   help="incidence angle from normal [deg]")
    r.add_argument("--n-decay", dest="n_decay", type=float, default=2.0)
    r.add_argument("--core-scale", dest="core_scale", type=float, default=1.0)
    r.add_argument("--plume-expansion-factor",
                   dest="plume_expansion_factor", type=float, default=3.0)
    r.add_argument("--separation-factor", dest="separation_factor",
                   type=float, default=1.0)
    r.add_argument("--r-sensor", dest="r_sensor", type=float, default=0.30,
                   help="sensor standoff [m]")
    r.add_argument("--theta", type=float, default=90.0,
                   help="angle from dipole axis [deg]")
    r.add_argument("--bands", type=float, nargs="*",
                   default=[315e6, 916e6], help="antenna frequencies [Hz]")
    r.add_argument("--standoff", type=float, default=0.10,
                   help="charged-surface standoff [m]")
    r.add_argument("--loop-area", dest="loop_area", type=float, default=1e-2)
    r.add_argument("--wire-length", dest="wire_length", type=float,
                   default=0.10)
    r.add_argument("--shielding-db", dest="shielding_db", type=float,
                   default=40.0)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="charge yield vs impact speed")
    add_common(s)
    s.add_argument("--vmin", type=float, default=20e3)
    s.add_argument("--vmax", type=float, default=72e3)
    s.add_argument("-n", type=int, default=15, help="number of speeds")
    s.set_defaults(func=cmd_sweep)

    v = sub.add_parser("validate", help="run the published-data comparisons")
    v.set_defaults(func=cmd_validate)

    lm = sub.add_parser("lammps",
                        help="LAMMPS MD impact deck (Fraile et al. protocol)")
    lm.add_argument("--material", default="W")
    lm.add_argument("--velocity", type=float, default=9e3,
                    help="impact speed [m/s]")
    lm.add_argument("--radius", type=float, default=1.3e-9,
                    help="projectile radius [m]")
    lm.add_argument("--damping-walls", dest="damping_walls",
                    action="store_true",
                    help="add the paper's fix-viscous edge damping")
    lm.add_argument("--out", default=None, help="write the deck to this file")
    lm.add_argument("--run", action="store_true",
                    help="run in-process (needs: pip install lammps mpich)")
    lm.set_defaults(func=cmd_lammps)

    dr = sub.add_parser("doctor",
                        help="check this machine: hardware, solvers, self-test")
    dr.add_argument("--quick", action="store_true",
                    help="skip the live self-test")
    dr.set_defaults(func=cmd_doctor)

    m = sub.add_parser("materials", help="list the material library")
    m.set_defaults(func=cmd_materials)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "quiet", False):
        warnings.simplefilter("ignore")
    return args.func(args)


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
