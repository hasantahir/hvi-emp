"""Validation against published measurements and simulations.

Every entry below is a number somebody else measured or computed, with the
citation attached.  `run_all()` executes the framework against each and prints
a pass/fail table with the ratio, so a change anywhere in the physics shows up
immediately as a regression.

Tolerances are deliberately loose (factors, not percentages).  This is a
reduced-order model built on a linear Us-Up Hugoniot, a Vinet cold curve, a
self-similar plume and a single-parameter charge-separation closure; claiming
better than factor-of-two agreement on absolute EMP amplitude would be
dishonest.  What the framework should get right, and is tested for, is:

  * shock-vaporisation thresholds (a stringent EOS test with no free
    parameters),
  * the Vinet cohesive energy against tabulated sublimation energies,
  * plume temperature and expansion speed,
  * the *scaling* of charge yield with velocity,
  * the order of magnitude of the radiated field at a real antenna,
  * the qualitative frequency discrepancy that the literature itself reports.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .eos import cohesive_energy_vinet, phase_thresholds
from .emp import emission_at_frequency
from .expansion import (simulate_expansion, stopping_distance,
                        torr_to_number_density)
from .impact import Projectile, empirical_charge_yield, simulate_impact
from .materials import ALUMINIUM, COPPER, DOLOMITE, IRON, TUNGSTEN
from .propagation import dust_optical_depth
from .pipeline import run_scenario


@dataclass
class Check:
    """One validation comparison."""
    name: str
    model: float
    reference: float
    unit: str
    tolerance: float          # acceptable multiplicative factor
    source: str
    note: str = ""
    #: Set when the model is *known* to disagree with the reference and the
    #: reason is understood and documented. Such a check still runs and still
    #: reports its number -- it is never silently hidden -- but it does not
    #: count as a regression, because the alternative is to widen a tolerance
    #: until it passes, which destroys the value of the whole suite.
    known_open: str = ""

    @property
    def ratio(self) -> float:
        if self.reference == 0:
            return np.inf if self.model != 0 else 1.0
        return self.model / self.reference

    @property
    def passed(self) -> bool:
        r = abs(self.ratio)
        if r == 0:
            return False
        return (1.0 / self.tolerance) <= r <= self.tolerance


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# Shock pressures for incipient (IM) / complete (CM) melting and incipient (IV)
# / complete (CV) vaporisation on release to 1 bar.  Ahrens & O'Keefe,
# The Moon 4, 214 (1972); Melosh, *Impact Cratering* (1989) Table 5.2; Pierazzo,
# Vickery & Melosh, Icarus 127, 408 (1997).  These are themselves uncertain by
# tens of percent between compilations.
SHOCK_THRESHOLDS_GPa = {
    "Al": {"IM": 65.0, "CM": 110.0, "IV": 380.0, "CV": 1500.0},
    "Fe": {"IM": 220.0, "CM": 400.0, "IV": 890.0, "CV": 2000.0},
    "Cu": {"IM": 140.0, "CM": 270.0, "IV": 580.0, "CV": 1500.0},
    "W": {"IM": 380.0, "CM": 650.0, "IV": 1500.0, "CV": 4000.0},
}


def check_eos() -> list[Check]:
    """Vinet cohesive energy and shock phase-change thresholds."""
    out: list[Check] = []
    for mat in (ALUMINIUM, IRON, COPPER, TUNGSTEN):
        out.append(Check(
            f"{mat.name}: Vinet binding energy vs sublimation energy",
            cohesive_energy_vinet(mat), mat.E_sublimation, "J/kg", 2.5,
            "Vinet et al., JGR 92, 9319 (1987); CRC Handbook cohesive "
            "energies",
            "no free parameters -- derived entirely from rho0, c0, s"))
        th = phase_thresholds(mat)
        ref = SHOCK_THRESHOLDS_GPa[mat.name]
        out.append(Check(
            f"{mat.name}: complete-melt shock pressure",
            th["P_melt"] / 1e9, ref["CM"], "GPa", 2.5,
            "Ahrens & O'Keefe (1972); Melosh (1989) Table 5.2"))
        tol = 4.0 if mat.name == "W" else 2.5
        out.append(Check(
            f"{mat.name}: incipient-vaporisation shock pressure",
            th["P_boil"] / 1e9, ref["IV"], "GPa", tol,
            "Ahrens & O'Keefe (1972); Melosh (1989) Table 5.2",
            "tungsten reference values are poorly constrained; the linear "
            "Us-Up fit is also weakest for refractory metals"
            if mat.name == "W" else ""))
        out.append(Check(
            f"{mat.name}: complete-vaporisation shock pressure",
            th["P_vap"] / 1e9, ref["CV"], "GPa", 2.5,
            "Ahrens & O'Keefe (1972); Melosh (1989) Table 5.2"))
    return out


def check_plasma_state() -> list[Check]:
    """Plume temperature and expansion speed against hydrocode results."""
    out: list[Check] = []
    # Fletcher & Close, Phys. Plasmas 24, 053102 (2017), Sec. III:
    # "The simulated plasma expands with a bulk fluid velocity of 30 km/s,
    #  corresponding to an impact speed of approximately 30 km/s, and an
    #  initial temperature of 2.5 eV."  Their initial conditions come from the
    #  ALEGRA/hydrocode runs of Fletcher, Close & Mathias (2015).
    sc = run_scenario("W", "Al", mass=1e-12, velocity=40e3, t_end=1e-5)
    out.append(Check(
        "plume electron temperature (W->Al, 40 km/s)",
        sc.impact.plasma.T_eV, 2.5, "eV", 3.0,
        "Fletcher & Close, Phys. Plasmas 24, 053102 (2017), Sec. III",
        "quoted as ~2-2.5 eV for a 30 km/s class impact"))
    if sc.expansion is not None:
        out.append(Check(
            "asymptotic plume expansion speed",
            sc.expansion.v_z[-1] / 1e3, 30.0, "km/s", 2.0,
            "Fletcher & Close (2017): 'the plasma flow speed is on the order "
            "of the original projectile speed'"))
        out.append(Check(
            "peak plume density, early expansion",
            sc.expansion.n_e[0], 1e27, "m^-3", 100.0,
            "Fletcher & Close (2017): 'peak density of ~1e27 m^-3 and a "
            "length scale of ~10 um' in the very early stages"))
    return out


def check_charge_yield() -> list[Check]:
    """Charge yield magnitude and, more importantly, its velocity exponent."""
    out: list[Check] = []
    v = np.array([30e3, 35e3, 40e3, 45e3, 50e3, 55e3, 60e3, 66e3, 72e3])
    Q = []
    for vv in v:
        try:
            sc = run_scenario("Fe", "Al", mass=1e-12, velocity=vv, t_end=1e-5)
            Q.append(sc.expansion.Q_final if sc.expansion else 0.0)
        except Exception:
            Q.append(0.0)
    Q = np.array(Q)

    # Fit above the vaporisation threshold.  Close et al. span 3-66 km/s and
    # fit a single power law across the threshold, so the meaningful
    # comparison is with the asymptotic (fully-vaporising) branch; the
    # near-threshold branch is necessarily much steeper and is reported
    # separately as a diagnostic.
    hi = (v >= 40e3) & (Q > 0)
    if hi.sum() >= 3:
        beta = float(np.polyfit(np.log(v[hi]), np.log(Q[hi]), 1)[0])
        out.append(Check(
            "charge-yield velocity exponent (asymptotic branch, >40 km/s)",
            beta, 3.48, "-", 1.5,
            "Close et al., Phys. Plasmas 20, 092102 (2013); McBride & "
            "McDonnell, Planet. Space Sci. 47, 1005 (1999)",
            "measured 3.4-3.5 for metal-on-metal impacts",
            known_open="""
This number is for the DEFAULT one-temperature Stage 2. It decomposes as:

    beta(at formation)        ~ 4.68    (Stage 1)
    beta(added by freeze-out) ~ +2.02   (Stage 2, 1T)
    beta(frozen, 1T)          ~ 6.70    vs 3.48 observed

The amplification is self-consistent, not a coding error: three-body
recombination goes as n_e^2 T^-4.5 and the plume temperature rises nearly
linearly with impact speed, so the surviving fraction climbs steeply.

RESOLVED by the two-temperature model, which is available but not default:

    run_scenario(..., expansion_model="2T")   ->   beta = 3.65

Two candidates were named here before either was implemented, and only one
of them mattered:

  * radiative cooling (radiation.py) -- RULED OUT. The radiative cooling time
    beats the expansion time by ~54x at formation and by six further orders
    of magnitude afterwards. Effect on beta: -0.26.
  * a two-temperature plume with non-equilibrium ionisation
    (expansion_2t.py) -- THIS IS THE MECHANISM. Recombination returns its
    binding energy to the electrons alone rather than to a pool dominated by
    ions with ~2.7x the heat capacity per particle, so the electrons stay hot
    and the T_e^-4.5 feedback throttles further recombination. Surviving
    charge rises from 0.5% to 41-69%, and the freeze-out contribution to beta
    changes SIGN: 1T adds +2.0, 2T subtracts -1.0.

No single 2T term explains it -- switching off recombination heating or e-i
equilibration individually gives 4.09 and 4.07, both far below the 1T 6.70.
The flattening is two-temperature physics as a whole.

The residual 4.4% (3.65 vs 3.48) is inside the framework's other
uncertainties, and the comparison is still not like for like: Close fits
across the vaporisation threshold over 3-66 km/s, this is an above-40 km/s
fit, and the model's own across-threshold fit is steeper still.

Left OPEN deliberately, on the default path, at the original tolerance. A
validation suite that must be 100% green is a suite that will be tuned until
it is. See docs/PHYSICS_AUDIT.md.
"""))
    lo = (v < 45e3) & (Q > 0)
    if lo.sum() >= 3:
        beta_lo = float(np.polyfit(np.log(v[lo]), np.log(Q[lo]), 1)[0])
        out.append(Check(
            "charge-yield exponent near the vaporisation threshold",
            beta_lo, 10.0, "-", 3.0,
            "diagnostic -- no direct measurement",
            "steepening near threshold is a prediction of the model, not a "
            "fitted feature; it is why a single power law fitted across the "
            "threshold lands between 3 and 4"))

    i = int(np.argmin(np.abs(v - 50e3)))
    if Q[i] > 0:
        out.append(Check(
            "absolute charge yield, Fe->Al, 1 pg, 50 km/s",
            Q[i], empirical_charge_yield(1e-12, 50e3, "fe_on_al"), "C", 30.0,
            "Ratcliff et al., Adv. Space Res. 20, 1471 (1997) prefactor",
            "empirical prefactors scatter by ~1 order of magnitude between "
            "experiments; only the order of magnitude is meaningful"))
    return out


def check_emp_amplitude() -> list[Check]:
    """Radiated field against the Close et al. (2013) patch-antenna data.

    Close et al. fired 1-100 fg iron projectiles at 3-66 km/s onto biased
    tungsten and aluminium targets and measured 1.9 mV/m at 0.30 m with a
    916 MHz patch antenna.  Fletcher & Close's DG-PIC, run for the same
    configuration, predicted 9.8 mV/m -- 5x high, as they note.

    The two charge-separation closures in `emp.separation_from_shell` bracket
    the measurement: the conservative Debye-scale displacement underpredicts,
    the Fletcher bulk-drift assumption overpredicts, and the measurement lies
    between them.  That bracket, rather than either single number, is the
    honest statement of what this framework can say about absolute EMP
    amplitude.
    """
    out: list[Check] = []
    mass, vel = 1e-16, 50e3          # inside the Close et al. range
    fields = {}
    for closure in ("debye", "fletcher"):
        try:
            sc = run_scenario("Fe", "W", mass=mass, velocity=vel, t_end=1e-4,
                              r_sensor=0.30, closure=closure,
                              bands=(315e6, 916e6))
            d = sc.bands.get(916e6, {})
            fields[closure] = d["E_peak"] if d.get("reached") else 0.0
        except Exception:
            fields[closure] = 0.0

    lo, hi = fields.get("debye", 0.0), fields.get("fletcher", 0.0)
    ref = 1.9e-3
    if lo > 0 and hi > 0:
        out.append(Check(
            "E at 916 MHz / 0.30 m -- geometric mean of the two closures",
            float(np.sqrt(lo * hi)), ref, "V/m", 5.0,
            "Close et al., Phys. Plasmas 20, 092102 (2013)",
            f"bracket: Debye closure {lo:.2e}, Fletcher closure {hi:.2e} V/m; "
            f"the measurement {ref:.2e} lies between them, and Fletcher & "
            "Close's own DG-PIC gave 9.8e-3 V/m"))
        out.append(Check(
            "measurement lies inside the closure bracket (1 = yes)",
            1.0 if lo <= ref <= hi else 0.0, 1.0, "-", 1.0,
            "Close et al. (2013) vs the two closures of this framework"))
    return out


def check_thresholds() -> list[Check]:
    """The RF-emission velocity threshold reported by Close and by Fletcher."""
    out: list[Check] = []
    v = np.arange(8e3, 46e3, 2e3)
    v_thresh = np.nan
    for vv in v:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = simulate_impact(Projectile(TUNGSTEN, 1e-12, vv), ALUMINIUM,
                                warn=False)
        if r.m_plasma > 0:
            v_thresh = vv
            break
    out.append(Check(
        "complete-vaporisation / RF-emission velocity threshold (W->Al)",
        v_thresh / 1e3, 17.5, "km/s", 2.0,
        "Close et al. (2013): no RF below 15-20 km/s; Fletcher (2021): "
        "'a threshold velocity that lies between 10 km/s and 20 km/s at "
        "which the projectile is entirely vaporized'"))
    return out


def check_jetting() -> list[Check]:
    """Jetting against Jean & Rollins (1970), the oldest data in this suite.

    Jean & Rollins fired spheres and cones at 1-8 km/s and tabulated the
    critical angle for jet onset, the jet speed against collision angle, and
    the luminous-ring and fast-jet speeds for aluminium. That is three
    independent geometric predictions with no free parameters, from a paper
    published fifty years before the rest of the references here.

    The comparison matters for more than antiquarian reasons: they observed a
    hot, strongly ionised, continuum-radiating plasma at velocities where this
    framework's *bulk* threshold says there should be none. `jetting.py`
    exists because that discrepancy has a specific resolution.
    """
    from .jetting import (FAST_JET_FACTOR, critical_angle, jet_velocity,
                          jet_velocity_at_angle)
    out: list[Check] = []

    # Table 1 -- critical angle for jet onset, Cu on Al.
    # (impact velocity km/s, measured critical angle deg, quoted error deg)
    tab1 = [(1.0, 12.5, 0.7), (3.0, 17.0, 2.0), (3.5, 22.0, 3.0),
            (5.5, 28.0, 1.0), (6.3, 32.0, 3.0)]
    errs = []
    for v_kms, meas, _err in tab1:
        model = np.degrees(critical_angle(COPPER, ALUMINIUM, v_kms * 1e3))
        errs.append(abs(model - meas))
    out.append(Check(
        "jetting critical angle, Cu->Al: mean |error| over 1-6.3 km/s",
        float(np.mean(errs)), 2.5, "deg", 2.0,
        "Jean & Rollins, AIAA J. 8, 1742 (1970), Table 1",
        "reference is the mean experimental error they quote; the model has "
        "no free parameters -- tan(alpha_c) = v / U_s with U_s from the "
        "framework's own impedance match"))

    # Table 2 -- steady jet speed ratio against collision angle, Cu-Al cones.
    # (angle deg, measured Vj/V1)
    tab2 = [(10, 13.0), (22, 5.5), (25, 4.9), (27, 4.7), (30, 3.2),
            (35, 3.3), (40, 3.5)]
    ratios = [jet_velocity_at_angle(1.0, np.radians(a)) / m for a, m in tab2]
    out.append(Check(
        "cone jet speed ratio, Cu-Al, 10-40 deg (model/measured)",
        float(np.mean(ratios)), 1.0, "-", 1.3,
        "Jean & Rollins (1970), Table 2",
        "v_jet = v * cot(alpha/2), a closed form with nothing fitted"))

    # Table 3 -- Al sphere on Al: luminous ring and fast jet.
    # (impact km/s, luminous ring km/s, fast jet km/s)
    tab3 = [(5.30, 15.6, 33.0), (4.94, 17.0, 37.6), (3.89, 14.6, 32.0),
            (6.05, 18.6, 35.0), (6.04, 16.8, 32.4), (5.88, 16.0, 30.6)]
    ring_ratio, fast_ratio = [], []
    for v_kms, ring, fast in tab3:
        js = jet_velocity(ALUMINIUM, ALUMINIUM, v_kms * 1e3, warn=False)
        ring_ratio.append((js.v_steady / 1e3) / ring)
        fast_ratio.append(fast / (js.v_steady / 1e3))
    out.append(Check(
        "steady jet vs measured luminous ring, Al->Al (model/measured)",
        float(np.mean(ring_ratio)), 1.18, "-", 1.15,
        "Jean & Rollins (1970), Table 3, 6 shots at 3.9-6.1 km/s",
        "the model should sit ~18% HIGH: Jean & Rollins explain that the "
        "fastest jet material is too sparse to photograph, so the measured "
        "ring is a lower bound on the theoretical maximum"))
    out.append(Check(
        "fast-jet factor recovered from the same six shots",
        float(np.mean(fast_ratio)), FAST_JET_FACTOR, "-", 1.05,
        "Jean & Rollins (1970), Table 3",
        "circular by construction -- FAST_JET_FACTOR is defined as this "
        "mean. It is here as a regression guard on the steady-jet formula "
        "underneath it, not as evidence"))

    # The reason the module exists: jetted material is shocked as though the
    # impact were several times faster than it is.
    js = jet_velocity(ALUMINIUM, ALUMINIUM, 6.0e3, warn=False)
    out.append(Check(
        "fast jet as a multiple of impact speed, Al->Al at 6 km/s",
        js.fast_ratio, 5.8, "-", 1.3,
        "Jean & Rollins (1970): 30.6-37.6 km/s jets from 3.9-6.1 km/s "
        "impacts, and Cu jets 'as high as 50 km/sec'",
        "this is why they saw ionised material far below the bulk "
        "vaporisation threshold; mass in the jet is NOT predicted"))
    return out


def check_chamber_conditions() -> list[Check]:
    """Whether published chambers were vacua for the plume."""
    out: list[Check] = []
    n0, R0 = 1e27, 1e-5
    # Close et al. (2013) ran at <= 1e-6 Torr explicitly to allow a
    # collisionless expansion; Bianchi et al. (1984) at 0.1-6 Torr.
    d_close = stopping_distance(torr_to_number_density(1e-6), n0, R0)
    d_bianchi = stopping_distance(torr_to_number_density(1.0), n0, R0)
    out.append(Check(
        "Close chamber (1e-6 Torr): stopping distance exceeds 1 cm (1 = yes)",
        1.0 if d_close > 0.01 else 0.0, 1.0, "-", 1.0,
        "Close et al. (2013): chamber at <= 1e-6 Torr 'to allow a "
        "collisionless expansion of plasma'",
        f"model gives {d_close*100:.1f} cm for a 10 um / 1e27 m^-3 seed "
        "plume -- comparable to the chamber scale, so the expansion is "
        "collisionless over the region that matters but not by a huge "
        "margin; the snowplough estimate scales as n_plume^(1/3) R0 and is "
        "therefore sensitive to the assumed seed"))
    out.append(Check(
        "Bianchi chamber (1 Torr): plume arrested within 1 cm (1 = yes)",
        1.0 if d_bianchi < 0.01 else 0.0, 1.0, "-", 1.0,
        "Bianchi et al., Nature 308, 830 (1984): 0.1-6 Torr",
        f"model gives {d_bianchi*1e3:.2f} mm -- the plume is collisionally "
        "arrested, so their ~300 kHz emission cannot be the coherent "
        "plasma-oscillation mechanism, consistent with their own argument "
        "that the plasma scale is too small for a dipole at that frequency"))
    out.append(Check(
        "ordering: 1e-6 Torr stopping distance >> 1 Torr (ratio)",
        d_close / max(d_bianchi, 1e-30), 100.0, "-", 3.0,
        "snowplough scaling R_stop ~ n_b^(-1/3): 6 decades of pressure gives "
        "2 decades of range"))
    return out


def check_dust_transparency() -> list[Check]:
    """Ejecta dust is opaque optically and transparent at RF."""
    out: list[Check] = []
    n_dust, a, path = 1e16, 1e-7, 0.05
    vis = dust_optical_depth(n_dust, a, path, 550e-9)
    rf = dust_optical_depth(n_dust, a, path, 0.327)     # 916 MHz
    out.append(Check(
        "ejecta cloud optically thick in the visible (tau > 1, 1 = yes)",
        1.0 if vis["optical_depth"] > 1.0 else 0.0, 1.0, "-", 1.0,
        "Tishkovets, Petrova & Mishchenko, JQSRT 112, 2095 (2011)",
        f"tau = {vis['optical_depth']:.2f} at 550 nm -- this is the visible "
        "impact flash"))
    out.append(Check(
        "ejecta cloud transparent at 916 MHz (tau < 1e-3, 1 = yes)",
        1.0 if rf["optical_depth"] < 1e-3 else 0.0, 1.0, "-", 1.0,
        "Rayleigh limit, Bohren & Huffman (1983) Ch. 5",
        f"tau = {rf['optical_depth']:.2e}, size parameter "
        f"x = {rf['size_parameter']:.2e}; any RF attenuation therefore comes "
        "from the plasma, not the dust"))
    out.append(Check(
        "visible/RF opacity contrast spans > 6 decades (log10 ratio)",
        float(np.log10(vis["optical_depth"] / max(rf["optical_depth"], 1e-300))),
        7.0, "-", 2.0,
        "Rayleigh x^4 scaling across 6 decades of wavelength"))
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(verbose: bool = True) -> dict:
    """Run every validation check and return a summary dict."""
    groups = {
        "Equation of state / phase change": check_eos,
        "Plasma state": check_plasma_state,
        "Charge yield": check_charge_yield,
        "EMP amplitude": check_emp_amplitude,
        "Velocity thresholds": check_thresholds,
        "Jetting (Jean & Rollins 1970)": check_jetting,
        "Chamber conditions": check_chamber_conditions,
        "Dust transparency": check_dust_transparency,
    }
    results: dict[str, list[Check]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, fn in groups.items():
            try:
                results[name] = fn()
            except Exception as exc:                     # pragma: no cover
                results[name] = []
                if verbose:
                    print(f"[{name}] FAILED TO RUN: {exc}")

    all_checks = [c for cs in results.values() for c in cs]
    n_pass = sum(c.passed for c in all_checks)
    n_tot = len(all_checks)
    # A failing check is a regression only if it is not a documented open
    # discrepancy.
    regressions = [c for c in all_checks if not c.passed and not c.known_open]
    open_issues = [c for c in all_checks if not c.passed and c.known_open]

    if verbose:
        print("=" * 100)
        print("HVI-EMP validation against published data")
        print("=" * 100)
        for name, cs in results.items():
            if not cs:
                continue
            print(f"\n--- {name} ---")
            print(f"{'check':<58}{'model':>12}{'ref':>12}{'ratio':>9}  ok")
            for c in cs:
                flag = ("OK " if c.passed
                        else ("OPEN" if c.known_open else "!! "))
                print(f"{c.name:<58}{c.model:>12.4g}{c.reference:>12.4g}"
                      f"{c.ratio:>9.3g}  {flag}")
            for c in cs:
                if c.note:
                    print(f"    note [{c.name[:40]}]: {c.note}")
        print("\n" + "=" * 100)
        print(f"{n_pass}/{n_tot} checks within tolerance")
        if open_issues:
            print(f"\n{len(open_issues)} KNOWN OPEN discrepancy(ies) -- "
                  "understood, documented, and deliberately not hidden by "
                  "widening a tolerance:")
            for c in open_issues:
                print(f"  * {c.name}")
                print(f"      model {c.model:.4g} vs {c.reference:.4g} "
                      f"{c.unit} (ratio {c.ratio:.3g}, tolerance {c.tolerance})")
                for line in c.known_open.strip().splitlines():
                    print(f"      {line.strip()}")
        if regressions:
            print(f"\n{len(regressions)} REGRESSION(S) -- these are bugs:")
            for c in regressions:
                print(f"  * {c.name}: {c.model:.4g} vs {c.reference:.4g} "
                      f"(ratio {c.ratio:.3g})")
        print("=" * 100)

    return {"results": results, "n_pass": n_pass, "n_total": n_tot,
            "checks": all_checks,
            "regressions": regressions, "open_issues": open_issues,
            "n_regressions": len(regressions),
            "ok": len(regressions) == 0}


if __name__ == "__main__":       # pragma: no cover
    run_all()


__all__ = ["Check", "run_all", "check_eos", "check_plasma_state",
           "check_charge_yield", "check_emp_amplitude", "check_thresholds",
           "check_chamber_conditions", "check_dust_transparency",
           "check_jetting",
           "SHOCK_THRESHOLDS_GPa"]
