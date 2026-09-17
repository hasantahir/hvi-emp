"""Impact jetting: the fast, sparse plasma source the bulk shock model misses.

Why this module exists
----------------------
Jean & Rollins (*AIAA J.* **8**, 1742, 1970) fired spheres at 2-8 km/s and saw
a hot, strongly ionised, continuum-radiating plasma at *every* velocity they
tried. The rest of this framework says that should be impossible: Stage 1 puts
the bulk complete-vaporisation threshold for Al on Al at 37 km/s, and Close
et al. (2013) report no RF emission below 15-20 km/s.

Both are right. The resolution is **jetting**. When two surfaces meet at a
shallow angle the contact point runs outward far faster than either body
moves, and a thin sheet of material is squirted out ahead of it at several
times the impact speed. Jean & Rollins measured Al jets at 30-38 km/s from
4-6 km/s impacts, and Cu jets at up to 50 km/s. Jetted material is therefore
shocked as though the impact were ~6x faster than it is, and it vaporises and
ionises when the bulk does not.

The catch, and the reason this does not overturn the RF threshold, is mass.
Jean & Rollins note that their fast jet is *not luminous* because "too little
material is involved". A jet can make a spectroscopically obvious plasma and
still carry far too little charge to radiate a detectable pulse. **This module
predicts jet velocity, not jet mass.** That is the honest limit of what closed
-form jetting theory gives you, and the mass fraction is exactly what a
resolved multi-material hydrocode — `solvers.m2c_stage1` — is for.

The model
---------
Two results, both with zero free parameters beyond one measured ratio.

**1. Critical angle.** Jetting begins when the shock detaches from the contact
point, i.e. when the contact point stops outrunning the shock (Walsh,
Shreffler & Willig, *J. Appl. Phys.* **24**, 349, 1953). For surfaces meeting
at angle `alpha` with closing speed `v`, the contact point travels at
`v / tan(alpha)`, so the condition is

    tan(alpha_c) = v / U_s

with `U_s` the shock velocity in the jetting material, taken from this
framework's own impedance match and linear Us-Up fit. Against Jean & Rollins'
Table 1 (Cu on Al, 1-6.3 km/s) this lands within **1.9 degrees** on average,
and inside the quoted experimental error at four of their five velocities.

**2. Jet velocity.** In the frame of the contact point the flow is steady and
Bernoulli applies, so material enters and leaves at the same speed. Adding the
contact-point velocity back gives the lab-frame jet speed
`v/tan(alpha) + v/sin(alpha)`, which simplifies to

    v_jet = v * cot(alpha_c / 2)

This reproduces Jean & Rollins' own tabulated jet velocities (their Table 2)
to within ~13% across 10-40 degrees, with no fitting.

**3. The fast jet.** Jean & Rollins distinguish the *steady* jet, which they
see as a luminous ring, from a faster transient produced while the shock is
detaching. The steady-jet formula above overpredicts their measured luminous
ring by a consistent 1.18 +/- 0.07 -- as they explain, the very fastest
material is too sparse to photograph -- while their non-luminous "fast jet"
runs at 1.73 +/- 0.18 times the steady prediction. That single measured ratio,
`FAST_JET_FACTOR`, is the only empirical number in this module, and it is
labelled as such rather than buried.

What this does not do
---------------------
No jet mass, no jet temperature, no charge, and therefore no EMP. The jet is
released material whose thermal state was fixed by the shock it already passed
through, so its final *speed* does not tell you its *temperature*, and this
module does not pretend otherwise. What the speed does license is
`downrange_threshold()`: the jet arrives somewhere, and when it does it is a
hypervelocity impactor in its own right.

Everything here is an upper bound that a sphere never quite realises, and
below `V_MIN_SPHERE` the bound goes vacuous -- `v * cot(alpha_c/2)` tends to
`2 * U_s` rather than to zero, so it will happily claim a 21 km/s jet from a
1 km/s impact. Functions warn or report `saturated` rather than returning that
number as if it meant something.

Treat a large jet/bulk gap as a flag that says "resolve this properly", not as
a new emission threshold. Resolving it properly means a multi-material
hydrocode that tracks the jet as a distinct material with its own mass and
temperature: `solvers.m2c_stage1`, whose level-set interface tracking and
non-ideal Saha solver exist for exactly this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eos import (impedance_match, particle_velocity_from_pressure,
                  phase_thresholds)
from .materials import Material

#: Measured ratio of the transient "fast" jet to the steady shaped-charge jet.
#: From Jean & Rollins (1970) Table 3, six Al-on-Al shots: the mean of their
#: measured fast-jet speed divided by this module's steady-jet prediction is
#: 1.73 with a standard deviation of 0.18. It is one number from one 1970
#: experiment on one material pair; `JetState.v_fast` carries that caveat.
FAST_JET_FACTOR = 1.73

#: Standard deviation of the above, across those six shots.
FAST_JET_FACTOR_SD = 0.18

#: Provenance, quoted wherever `v_fast` is reported.
FAST_JET_SOURCE = ("Jean & Rollins, AIAA J. 8, 1742 (1970), Table 3 "
                   "(6 Al-on-Al shots, 3.9-6.1 km/s)")


@dataclass
class JetState:
    """Jetting geometry and speeds for one projectile/target/velocity."""

    v_impact: float
    #: Angle between the colliding surfaces at which the shock detaches and
    #: jetting begins [rad]. For a sphere the angle sweeps 0 -> 90 degrees
    #: during penetration, so every impact passes through this value.
    alpha_c: float
    #: Shock velocity in the jetting (target) material at the impact
    #: condition [m/s]. This is what sets `alpha_c`.
    U_s: float
    #: Steady shaped-charge jet speed [m/s]. Compare with an observed
    #: luminous ring; it is an upper bound on what is bright enough to see.
    v_steady: float
    #: Transient detachment jet [m/s] = `FAST_JET_FACTOR * v_steady`. This is
    #: the fastest material in the problem and the likeliest plasma source.
    v_fast: float

    @property
    def alpha_c_deg(self) -> float:
        return float(np.degrees(self.alpha_c))

    @property
    def steady_ratio(self) -> float:
        """Steady jet speed as a multiple of the impact speed."""
        return self.v_steady / self.v_impact

    @property
    def fast_ratio(self) -> float:
        """Fast jet speed as a multiple of the impact speed."""
        return self.v_fast / self.v_impact

    def summary(self) -> str:
        return (
            f"impact {self.v_impact/1e3:.2f} km/s -> "
            f"alpha_c {self.alpha_c_deg:.1f} deg, "
            f"steady jet {self.v_steady/1e3:.1f} km/s "
            f"({self.steady_ratio:.1f}x), "
            f"fast jet {self.v_fast/1e3:.1f} km/s "
            f"({self.fast_ratio:.1f}x)  [mass fraction not predicted]")


def critical_angle(projectile: Material, target: Material,
                   v_impact: float, warn: bool = False) -> float:
    """Angle at which the shock detaches and jetting begins [rad].

    `tan(alpha_c) = v / U_s`, with `U_s` the shock velocity in the target
    taken from the framework's impedance match. Validated against Jean &
    Rollins (1970) Table 1 to a mean absolute error of 1.9 degrees.

    The angle *increases* with impact speed, because `U_s = c0 + s*u_p` grows
    more slowly than `v` does. Physically: faster impacts jet over a wider
    range of the sphere's surface, so they jet more readily, not less.
    """
    if v_impact <= 0:
        raise ValueError("v_impact must be positive")
    st = impedance_match(projectile, target, v_impact, warn=warn)
    U_s = target.hugoniot_Us(st.up_target, warn=warn)
    return float(np.arctan(v_impact / U_s))


def jet_velocity(projectile: Material, target: Material, v_impact: float,
                 warn: bool = True) -> JetState:
    """Jetting speeds for a sphere impact.

    A sphere's contact angle sweeps continuously from 0 to 90 degrees as it
    penetrates, so it necessarily passes through `alpha_c`; the peak jet speed
    is the one produced there. A cone at fixed angle jets at a fixed speed,
    which is why Jean & Rollins used cones to map the angle dependence.

    These are **upper bounds**. Jean & Rollins are explicit that a sphere
    never realises the theoretical maximum, because the amount of material
    jetted depends on how long the sphere lingers near the critical angle, and
    the fastest material is too sparse to see. Below `V_MIN_SPHERE` the bound
    becomes vacuous and this warns.
    """
    if v_impact < V_MIN_SPHERE and warn:
        import warnings as _w
        _w.warn(
            f"jet_velocity called at {v_impact/1e3:.2f} km/s, below the "
            f"{V_MIN_SPHERE/1e3:.0f} km/s floor of Jean & Rollins' sphere "
            f"data. The prediction tends to 2*U_s as v -> 0 and becomes a "
            f"vacuous upper bound, not a wrong one -- treat it as 'jetting "
            f"is possible', never as a speed.",
            RuntimeWarning, stacklevel=2)
    a_c = critical_angle(projectile, target, v_impact, warn=False)
    st = impedance_match(projectile, target, v_impact, warn=False)
    U_s = target.hugoniot_Us(st.up_target, warn=False)
    # v/tan(a) + v/sin(a) == v * cot(a/2)
    v_steady = v_impact / np.tan(0.5 * a_c)
    return JetState(v_impact=float(v_impact), alpha_c=a_c, U_s=float(U_s),
                    v_steady=float(v_steady),
                    v_fast=float(FAST_JET_FACTOR * v_steady))


def jet_velocity_at_angle(v_impact: float, alpha: float) -> float:
    """Steady jet speed for a *fixed* collision angle [m/s].

    This is the cone-impact case, and the form Jean & Rollins tabulate in
    their Table 2. `alpha` in radians. Diverges as `alpha -> 0`, which is
    physical only down to `critical_angle`: below that the shock stays
    attached and no jet forms at all.
    """
    if not 0.0 < alpha < np.pi:
        raise ValueError("alpha must be in (0, pi) radians")
    return float(v_impact / np.tan(0.5 * alpha))


#: Jean & Rollins' critical-angle data starts here, so neither does ours.
V_MIN_VALID = 1.0e3

#: Below this the *sphere* jet-speed prediction is extrapolation, and in the
#: dangerous direction. `v_jet = v * cot(alpha_c/2)` tends to `2 * U_s` as
#: `v -> 0`, so it claims a ~21 km/s jet from a 1 km/s impact on aluminium.
#: That is not an artefact of our critical angle -- feeding Jean & Rollins'
#: own *measured* 12.5 degrees at 1 km/s into the same formula still gives
#: 9x -- it is inherent to evaluating a steady-flow result exactly at the
#: angle where the jet is only beginning to form and carries almost no mass.
#: Their sphere measurements (Table 3) span 3.9-6.1 km/s and that is the only
#: range where the sphere prediction has been checked. `jet_velocity` warns
#: below this.
V_MIN_SPHERE = 3.0e3


def downrange_threshold(projectile: Material, target: Material,
                        wall: Material | None = None,
                        which: str = "P_boil",
                        v_max: float = 60e3) -> dict:
    """Impact speed at which the *jet* vaporises a downrange wall.

    This is the question jetting actually licenses us to ask, and the reason
    it matters for shielding. Jet speed is kinematic and well-founded; jet
    *temperature* is not, because the jet is released material whose thermal
    state was set by the shock it already passed through, not by the speed it
    ends up at. So this routine does not claim the jet is hot. It claims the
    jet is **fast**, and asks what happens when that fast material hits the
    next surface downrange -- a Whipple shield's rear wall, an adjacent
    harness, a tank.

    A 6 km/s strike on a bumper delivers a ~36 km/s jet to whatever is behind
    it. The rear wall therefore sees a hypervelocity impact several times more
    energetic per unit mass than the one that hit the front, which is the
    mechanism behind ESA's Zero Debris concern about impacts affecting
    *internal* items (Technical Booklet section 3.5).

    Parameters
    ----------
    wall : Material or None
        The downrange surface. Defaults to `target`.
    which : {"P_melt", "P_boil", "P_vap"}
        Which `eos.phase_thresholds` entry to test against.

    Returns
    -------
    dict with ``v_bulk`` (impact speed at which the primary impact itself
    crosses the threshold in the target), ``v_downrange`` (impact speed at
    which the jet crosses it in the wall) and ``gain`` = ``v_bulk /
    v_downrange``, i.e. how much earlier the downrange damage starts.
    ``nan`` where a threshold is not reached below `v_max`.

    The mass carried by the jet is **not** predicted -- see the module
    docstring. A low `v_downrange` means "this can happen", not "this will
    deposit enough material to matter".
    """
    wall = wall if wall is not None else target
    th_t = phase_thresholds(target)
    th_w = phase_thresholds(wall)
    if which not in th_t:
        raise KeyError(f"unknown threshold {which!r}; have {sorted(th_t)}")
    up_needed_t = particle_velocity_from_pressure(target, th_t[which])
    up_needed_w = particle_velocity_from_pressure(wall, th_w[which])

    v_bulk = np.nan
    v_down = np.nan
    for vv in np.arange(V_MIN_SPHERE, v_max, 250.0):
        if np.isnan(v_bulk):
            st = impedance_match(projectile, target, vv, warn=False)
            if st.up_target >= up_needed_t:
                v_bulk = float(vv)
        if np.isnan(v_down):
            js = jet_velocity(projectile, target, vv, warn=False)
            # jetted target material striking the wall at v_fast
            sw = impedance_match(target, wall, js.v_fast, warn=False)
            if sw.up_target >= up_needed_w:
                v_down = float(vv)
        if not (np.isnan(v_bulk) or np.isnan(v_down)):
            break

    # The jet-speed bound rises as v falls, so a search that succeeds at the
    # very first sample has not found a threshold -- it has run off the
    # bottom of the validated range. Say so instead of returning the floor
    # as though it were a result.
    saturated = bool(v_down == V_MIN_SPHERE)
    gain = np.nan
    if not (np.isnan(v_bulk) or np.isnan(v_down)) and v_down > 0:
        gain = v_bulk / v_down
    return {"threshold": which,
            "v_bulk": v_bulk,
            "v_downrange": np.nan if saturated else v_down,
            "gain": np.nan if saturated else float(gain),
            "saturated": saturated,
            "note": (f"the jet exceeds this threshold at every speed down to "
                     f"the {V_MIN_SPHERE/1e3:.0f} km/s validity floor, so no "
                     f"threshold is reported; see V_MIN_SPHERE"
                     if saturated else ""),
            "wall": wall.name, "target": target.name}


__all__ = ["JetState", "critical_angle", "jet_velocity",
           "jet_velocity_at_angle", "downrange_threshold",
           "V_MIN_VALID", "V_MIN_SPHERE",
           "FAST_JET_FACTOR", "FAST_JET_FACTOR_SD", "FAST_JET_SOURCE"]
