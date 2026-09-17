"""Stage 1 -- impact, shock decay, vaporisation and the initial plasma inventory.

Model chain
-----------
1.  **Planar impact approximation** (`eos.impedance_match`) gives the peak
    "isobaric core" pressure P_ic and the particle velocities in projectile
    and target.
2.  **Shock decay.**  Outside the isobaric core the peak pressure falls as a
    power law, ``P(r) = P_ic (r / r_ic)^-n`` with n ~ 1.5-3 (Melosh 1989
    Sec. 5.4; Pierazzo, Vickery & Melosh, Icarus 127, 408 (1997) fit n ~ 1.2
    near-field steepening to ~3 far-field).  `n_decay` is a first-class
    parameter and `examples/03_sensitivity.py` quantifies its influence.
3.  **Phase state.**  For each shell the release waste heat
    (`eos.release_state`) is converted to melt and vapour *fractions* by the
    lever rule, so partial vaporisation is resolved rather than being a
    binary switch.  The projectile is shocked essentially uniformly to P_ic,
    which is what produces the sharp "complete vaporisation" velocity
    threshold reported by Fletcher (NRL/MR/6757--20-10,138, 2021).
4.  **Ionisation.**  The vapour is handed to `ionization.partition_energy` at
    the density it has reached once it has expanded to a few projectile radii
    (where it is optically thin and adiabatic), giving T_e, Zbar and hence the
    free-charge inventory Q.

The empirical charge-yield laws in `empirical_charge_yield` are for
*cross-checking only* -- see that docstring for the (substantial) caveats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import E_CHARGE
from .eos import (ImpactState, impedance_match, melt_fraction,
                  particle_velocity_from_pressure, phase_thresholds,
                  release_state, spinodal_volume, vapour_fraction)
from .ionization import IonisationState, get_table, saha_solve
from .materials import Material, mix_materials
from ._compat import trapezoid


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class Projectile:
    """Impacting debris particle or meteoroid."""
    material: Material
    mass: float                      # kg
    velocity: float                  # m/s
    angle_deg: float = 0.0           # from surface normal

    @classmethod
    def from_diameter(cls, material: Material, diameter: float,
                      velocity: float, angle_deg: float = 0.0) -> "Projectile":
        """Construct from an equivalent-sphere diameter [m]."""
        m = material.rho0 * np.pi * diameter**3 / 6.0
        return cls(material, m, velocity, angle_deg)

    @property
    def radius(self) -> float:
        """Equivalent-sphere radius [m]."""
        return (3.0 * self.mass / (4.0 * np.pi * self.material.rho0)) ** (1/3)

    @property
    def v_normal(self) -> float:
        """Velocity component normal to the surface [m/s].

        Only the normal component couples into the shock; the tangential
        component leaves in downrange ejecta and ricochet.  Standard
        first-order treatment (Melosh 1989 Sec. 5.3), good to ~20% for
        incidence angles within 60 deg of normal.
        """
        return self.velocity * np.cos(np.radians(self.angle_deg))

    @property
    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * self.velocity**2


@dataclass
class ImpactResult:
    """Everything Stage 1 hands to Stage 2."""
    projectile: Projectile
    target: Material
    state: ImpactState
    P_ic: float                      # Pa
    r_ic: float                      # m
    m_vapour_projectile: float       # kg
    m_vapour_target: float           # kg
    m_vapour: float                  # kg, total (incl. two-phase)
    m_plasma: float                  # kg, superheated plasma-forming mass
    m_melt_target: float             # kg
    f_vap_projectile: float          # -
    E_res_projectile: float          # J/kg
    E_res_core: float                # J/kg
    E_vapour_specific: float         # J/kg, superheat of the plasma component
    u_atom: float                    # J per heavy particle, above free atoms
    material_mix: Material
    plasma: IonisationState
    Q_free: float                    # C
    n_h0: float                      # m^-3
    rho_plume0: float                # kg/m^3
    r_plume0: float                  # m
    v_expansion: float               # m/s, characteristic plume speed
    thresholds: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def vaporisation_efficiency(self) -> float:
        """Vaporised mass per unit projectile mass."""
        return self.m_vapour / self.projectile.mass

    @property
    def plasma_efficiency(self) -> float:
        """Superheated (plasma-forming) mass per unit projectile mass."""
        return self.m_plasma / self.projectile.mass

    @property
    def charge_per_mass(self) -> float:
        """C per kg of projectile -- the quantity plotted in impact-ionisation
        experiments."""
        return self.Q_free / self.projectile.mass

    def summary(self) -> str:
        p = self.projectile
        return "\n".join([
            f"Impact: {p.material.name} ({p.mass:.3e} kg, {p.radius*1e6:.2f} um "
            f"radius) -> {self.target.name} at {p.velocity/1e3:.1f} km/s, "
            f"{p.angle_deg:.0f} deg from normal",
            f"  peak shock pressure     {self.P_ic/1e9:10.1f} GPa",
            f"  kinetic energy          {p.kinetic_energy:10.3e} J",
            f"  vapour mass (total)     {self.m_vapour:10.3e} kg "
            f"({self.vaporisation_efficiency:.2f} x m_proj)",
            f"  plasma mass (superheat) {self.m_plasma:10.3e} kg "
            f"({self.plasma_efficiency:.3f} x m_proj)",
            f"    from projectile       {self.m_vapour_projectile:10.3e} kg "
            f"(f_vap = {self.f_vap_projectile:.2f})",
            f"    from target           {self.m_vapour_target:10.3e} kg",
            f"  target melt mass        {self.m_melt_target:10.3e} kg",
            f"  plasma temperature      {self.plasma.T_eV:10.2f} eV",
            f"  mean charge state Zbar  {self.plasma.Zbar:10.3f}",
            f"  free charge Q           {self.Q_free:10.3e} C",
            f"  initial n_e             {self.plasma.n_e:10.3e} m^-3",
            f"  expansion speed         {self.v_expansion/1e3:10.2f} km/s",
        ])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def isobaric_core_radius(proj: Projectile, core_scale: float = 1.0) -> float:
    """Radius of the near-constant-pressure region [m].

    Melosh (1989) and hydrocode fits give r_ic ~ (0.5-1.5) x projectile
    radius; `core_scale` selects within that range.
    """
    return core_scale * proj.radius


def peak_pressure_profile(P_ic: float, r, r_ic: float, n_decay: float,
                          smooth: bool = True):
    """Peak shock pressure at distance ``r`` from the impact point.

    The textbook description is piecewise: constant P_ic inside an "isobaric
    core" of radius r_ic, then P_ic (r/r_ic)^-n outside it.  That description
    is a summary of hydrocode output, not a solution -- the real field rolls
    over smoothly, and the kink at r = r_ic is an artefact of writing it in
    two pieces.

    The artefact is not cosmetic. A perfectly flat core means every gram of
    material inside it crosses the vaporisation threshold at the *same*
    impact speed, which puts a step in plasma mass, temperature and charge:
    for a spherical projectile the core holds exactly 5/16 of its mass, and
    that 31% appeared all at once.

    The smooth form

        P(r) = P_ic / (1 + (r/r_ic)^n)

    has both correct asymptotes (-> P_ic for r << r_ic, -> P_ic (r/r_ic)^-n
    for r >> r_ic), is monotone and C-infinity, and introduces no new
    parameter. Pass ``smooth=False`` for the piecewise form.
    """
    r = np.asarray(r, dtype=float)
    r_ic = max(float(r_ic), 1e-30)
    if not smooth:
        return np.where(r <= r_ic, P_ic, P_ic * (r / r_ic) ** (-n_decay))
    return P_ic / (1.0 + (r / r_ic) ** n_decay)


def _shell_integrals(target: Material, P_ic: float, r_ic: float,
                     n_decay: float, n_shells: int = 400) -> dict:
    """Integrate hemispherical target shells for mass and energy inventories.

    The distinction that matters for EMP work is between

      * **plasma-forming mass** -- material whose release waste heat exceeds
        the complete-vaporisation threshold, so it becomes a *superheated*
        vapour with energy left over for ionisation, and
      * **condensable vapour / dust** -- material in the two-phase region
        (0 < f_vap < 1), which arrives at the boiling point with no superheat
        and recondenses into the dust component of the plume.

    Mass-weighting these together (as a single-temperature plume model does)
    is badly wrong: the shell mass grows as r^2 while the waste heat falls as
    a steep power of r, so the average is dominated by the *coldest*
    marginally-vaporised material and the plume temperature is underestimated
    by an order of magnitude.

    Returns
    -------
    dict with ``m_plasma``, ``E_plasma`` (mass-weighted waste heat of the
    superheated component), ``m_vapour`` (total, lever rule), ``m_melt``,
    and ``r_max``.
    """
    th = phase_thresholds(target)
    out = {"m_plasma": 0.0, "E_plasma": 0.0, "m_vapour": 0.0,
           "m_melt": 0.0, "r_max": r_ic}
    P_min = min(th["P_melt"], th["P_boil"])
    if not np.isfinite(P_min) or P_min >= P_ic:
        return out

    r_max = r_ic * (P_ic / P_min) ** (1.0 / n_decay)
    r = np.linspace(r_ic, r_max, n_shells)
    P_r = P_ic * (r / r_ic) ** (-n_decay)
    E_r = np.array([
        release_state(target, particle_velocity_from_pressure(target, p),
                      warn=False)["E_residual"] for p in P_r])
    fv = np.array([vapour_fraction(target, e) for e in E_r])
    fm = np.array([melt_fraction(target, e) for e in E_r])
    dm_dr = 2.0 * np.pi * r**2 * target.rho0

    # Isobaric core, at the core state.
    E_core = release_state(target,
                           particle_velocity_from_pressure(target, P_ic),
                           warn=False)["E_residual"]
    m_core = (2.0 / 3.0) * np.pi * r_ic**3 * target.rho0

    m_vap = (m_core * vapour_fraction(target, E_core)
             + float(trapezoid(fv * dm_dr, r)))
    m_melt = (m_core * melt_fraction(target, E_core)
              + float(trapezoid(fm * dm_dr, r)))

    # Superheated (plasma-forming) component only.
    hot = E_r >= th["E_vap"]
    m_hot = m_core if E_core >= th["E_vap"] else 0.0
    num = m_hot * E_core
    if hot.any():
        m_hot += float(trapezoid(dm_dr[hot], r[hot]))
        num += float(trapezoid(dm_dr[hot] * E_r[hot], r[hot]))

    out.update({"m_plasma": m_hot,
                "E_plasma": num / m_hot if m_hot > 0 else 0.0,
                "m_vapour": m_vap, "m_melt": m_melt, "r_max": r_max})
    return out


def _projectile_integrals(proj: "Projectile", P_ic: float, r_ic: float,
                          n_decay: float, n_shells: int = 400) -> dict:
    """Integrate the projectile against the same peak-pressure field.

    The target is integrated over hemispherical shells of the decaying peak
    pressure P(r) = P_ic (r/r_ic)^-n.  The projectile sits in the *other* half
    of that same field, so it should be integrated the same way -- not treated
    as a single lump at the interface pressure.

    Doing it as a lump makes the plasma-forming mass an all-or-nothing step in
    velocity: zero below the complete-vaporisation threshold and the entire
    projectile above it.  That is unphysical (real projectiles vaporise
    progressively, front to back, as release waves from the free rear and
    lateral surfaces cut the peak pressure with distance from the contact) and
    it puts a discontinuity in plasma mass, charge, temperature and every
    downstream EMP quantity, right in the middle of the velocity range this
    framework is used for.

    Geometry, exactly
    -----------------
    A sphere of radius ``a`` resting on the plane has its centre at height
    ``a`` above the impact point.  The mass at distance ``r`` from that point
    follows from the area of the sphere of radius ``r`` lying inside the
    projectile,

        dm/dr = rho0 (2 pi r^2 - pi r^3 / a),      0 <= r <= 2a

    which integrates to exactly (4/3) pi a^3 rho0.  No fitting, no free
    parameter: it is the geometry of a ball touching a plane.

    Returns the same keys as ``_shell_integrals``.
    """
    mat = proj.material
    a = proj.radius
    th = phase_thresholds(mat)
    out = {"m_plasma": 0.0, "E_plasma": 0.0, "m_vapour": 0.0, "m_melt": 0.0}

    r = np.linspace(1e-9 * a, 2.0 * a, n_shells)
    # exact mass distribution of a ball of radius a touching the origin
    dm_dr = mat.rho0 * (2.0 * np.pi * r**2 - np.pi * r**3 / a)
    dm_dr = np.maximum(dm_dr, 0.0)

    # same peak-pressure field as the target
    P_r = peak_pressure_profile(P_ic, r, r_ic, n_decay)

    E_r = np.array([
        release_state(mat, particle_velocity_from_pressure(mat, p),
                      warn=False)["E_residual"] for p in P_r])
    fv = np.array([vapour_fraction(mat, e) for e in E_r])
    fm = np.array([melt_fraction(mat, e) for e in E_r])

    m_total = float(trapezoid(dm_dr, r))
    # normalise out the O(1/n_shells) quadrature error so mass is exact
    scale = proj.mass / m_total if m_total > 0 else 1.0
    dm_dr = dm_dr * scale

    m_vap = float(trapezoid(fv * dm_dr, r))
    m_melt = float(trapezoid(fm * dm_dr, r))

    hot = E_r >= th["E_vap"]
    m_hot, num = 0.0, 0.0
    if hot.any():
        m_hot = float(trapezoid(dm_dr[hot], r[hot]))
        num = float(trapezoid(dm_dr[hot] * E_r[hot], r[hot]))

    out.update({"m_plasma": m_hot,
                "E_plasma": num / m_hot if m_hot > 0 else 0.0,
                "m_vapour": min(m_vap, proj.mass),
                "m_melt": min(m_melt, proj.mass)})
    return out


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def simulate_impact(proj: Projectile, target: Material,
                    n_decay: float = 2.0,
                    core_scale: float = 1.0,
                    plume_expansion_factor: float = 3.0,
                    warn: bool = True) -> ImpactResult:
    """Run Stage 1.

    Parameters
    ----------
    proj : Projectile
    target : Material
    n_decay : float
        Shock-decay exponent P ~ r^-n_decay (physical range 1.5-3).
    core_scale : float
        Isobaric-core radius in projectile radii (0.5-1.5).
    plume_expansion_factor : float
        Plume radius, in projectile radii, at which the Saha equilibrium is
        evaluated.  The ionisation state is density dependent; this is the
        point at which the plume has become optically thin and adiabatic.
        Freeze-out (Stage 2) is what actually sets the charge that survives.
    """
    state = impedance_match(proj.material, target, proj.v_normal, warn=warn)
    r_ic = isobaric_core_radius(proj, core_scale)

    rel_p = release_state(proj.material, state.up_projectile, warn=warn)
    rel_t = release_state(target, state.up_target, warn=warn)
    E_res_p, E_res_c = rel_p["E_residual"], rel_t["E_residual"]

    # The projectile is integrated against the same decaying peak-pressure
    # field as the target, over the exact geometry of a ball touching the
    # impact point. Treating it as a single lump at the interface pressure
    # made the plasma-forming mass a step function of velocity (0 below the
    # complete-vaporisation threshold, the whole projectile above it), which
    # put a discontinuity in every downstream quantity.
    pj = _projectile_integrals(proj, state.P, r_ic, n_decay)
    m_vap_p = pj["m_vapour"]
    f_vap_p = m_vap_p / proj.mass if proj.mass > 0 else 0.0
    m_plasma_p = pj["m_plasma"]
    E_plasma_p = pj["E_plasma"]
    th_p = phase_thresholds(proj.material)

    sh = _shell_integrals(target, state.P, r_ic, n_decay)
    m_vap_t, m_melt_t = sh["m_vapour"], sh["m_melt"]
    m_plasma_t, E_plasma_t = sh["m_plasma"], sh["E_plasma"]

    m_vap = m_vap_p + m_vap_t
    m_plasma = m_plasma_p + m_plasma_t

    # --- superheated (plasma-forming) component --------------------------
    if m_plasma > 0:
        w_p = m_plasma_p / m_plasma
        w_t = 1.0 - w_p
        # Number-weighted mixture rather than "whichever dominates": picking
        # a winner puts a spurious discontinuity in T_e and Zbar at the
        # crossover speed, because the mean atomic mass jumps by a factor of
        # two between projectile and target.
        mat_mix = mix_materials(proj.material, w_p, target, w_t)
        # Internal energy per heavy particle above free neutral atoms at rest:
        # the waste heat minus the sublimation energy already spent. Both
        # sides use the mass-weighted waste heat of their *superheated*
        # material -- using the bulk projectile value here (as an earlier
        # version did) mixes in material that never got hot enough to ionise.
        u_p = max(E_plasma_p - proj.material.E_sublimation, 0.0)
        u_t = max(E_plasma_t - target.E_sublimation, 0.0)
        u_specific = w_p * u_p + w_t * u_t          # J/kg
        u_atom = u_specific * mat_mix.m_atom        # J per heavy particle
    else:
        w_p = w_t = 0.0
        u_specific = u_atom = 0.0
        mat_mix = target

    # --- initial plume state --------------------------------------------
    # The natural Stage-1 -> Stage-2 handoff density is the spinodal volume,
    # where the release calculation ends and the material has fragmented into
    # vapour.  Starting exactly there would put the plasma at near-solid
    # density with coupling parameter Gamma >> 1, outside the validity of both
    # ideal Saha and the Coulomb-collision formulary, so the handoff is taken
    # after a further linear expansion by `plume_expansion_factor`.  Defining
    # the initial density this way (rather than by the projectile radius)
    # makes it independent of the plasma mass, which is what allows the charge
    # yield to scale cleanly with velocity.
    V_specific0 = plume_expansion_factor**3 * spinodal_volume(mat_mix)
    rho_plume = 1.0 / V_specific0 if m_plasma > 0 else 0.0
    V_plume = m_plasma * V_specific0
    r_plume0 = ((3.0 * V_plume) / (2.0 * np.pi)) ** (1.0 / 3.0) \
        if V_plume > 0 else proj.radius
    n_h0 = rho_plume / mat_mix.m_atom if rho_plume > 0 else 0.0

    if n_h0 > 0 and u_atom > 0:
        T0, Z0 = get_table(mat_mix).state(n_h0, u_atom)
        plasma = saha_solve(mat_mix, T0, n_h0)
    else:
        z = np.zeros(len(mat_mix.E_ion) + 1)
        z[0] = 1.0
        plasma = IonisationState(300.0, n_h0, 0.0, 0.0, z, 0.0)

    n_atoms = m_plasma / mat_mix.m_atom if m_plasma > 0 else 0.0
    Q_free = E_CHARGE * plasma.Zbar * n_atoms
    E_mix = u_specific

    # --- weakly-ionised two-phase vapour ---------------------------------
    # Material that only partially vaporises arrives at the boiling point with
    # no superheat, but it is not electrically inert: thermal ionisation at
    # T_vap still liberates a small free-electron population, and at
    # laboratory impact speeds (5-15 km/s) this is the *only* source of
    # charge.  It is what the AVGR-class experiments of Crawford & Schultz
    # measure, and it is why a pure "complete vaporisation" threshold model
    # predicts zero signal where experiments see one.
    m_twophase = max(m_vap - m_plasma, 0.0)
    Q_twophase = 0.0
    twophase = None
    if m_twophase > 0:
        rho_tp = 1.0 / (plume_expansion_factor**3 * spinodal_volume(target))
        n_tp = rho_tp / target.m_atom
        twophase = saha_solve(target, target.T_vap, n_tp)
        Q_twophase = (E_CHARGE * twophase.Zbar * m_twophase
                      / target.m_atom)
    Q_free = Q_free + Q_twophase

    # Characteristic expansion speed: the impact-driven bulk motion, of order
    # the impact speed, is what the hydrocode studies report; the thermal
    # sound speed sets the *additional* spread.  Take the larger.
    from .constants import K_B
    c_s = np.sqrt(5.0 / 3.0 * K_B * plasma.T * (1.0 + plasma.Zbar)
                  / mat_mix.m_atom) if plasma.T > 0 else 0.0
    v_exp = max(c_s, 0.5 * proj.v_normal)

    return ImpactResult(
        projectile=proj, target=target, state=state,
        P_ic=state.P, r_ic=r_ic,
        m_vapour_projectile=m_vap_p, m_vapour_target=m_vap_t,
        m_vapour=m_vap, m_plasma=m_plasma, m_melt_target=m_melt_t,
        f_vap_projectile=f_vap_p,
        E_res_projectile=E_res_p, E_res_core=E_res_c,
        E_vapour_specific=E_mix, u_atom=u_atom, material_mix=mat_mix,
        plasma=plasma, Q_free=Q_free, n_h0=n_h0,
        rho_plume0=rho_plume, r_plume0=r_plume0, v_expansion=v_exp,
        thresholds={"target": phase_thresholds(target),
                    "projectile": phase_thresholds(proj.material)},
        diagnostics={
            "w_projectile": w_p, "w_target": w_t,
            "n_decay": n_decay, "core_scale": core_scale,
            "plume_expansion_factor": plume_expansion_factor,
            "sound_speed": c_s,
            "KE": proj.kinetic_energy,
            "m_plasma_projectile": m_plasma_p,
            "m_plasma_target": m_plasma_t,
            "E_plasma_target": E_plasma_t,
            "shell_integrals": sh,
            "m_twophase": m_twophase,
            "Q_twophase": Q_twophase,
            "Q_superheated": Q_free - Q_twophase,
            "twophase_state": twophase,
            "V_specific0": V_specific0,
            "E_in_plasma": m_plasma * E_mix,
            "plasma_energy_fraction": (m_plasma * E_mix / proj.kinetic_energy
                                       if proj.kinetic_energy > 0 else 0.0),
            "release_projectile": rel_p,
            "release_target": rel_t,
        },
    )


# ---------------------------------------------------------------------------
# Empirical cross-checks
# ---------------------------------------------------------------------------

_EMPIRICAL_LAWS = {
    # name: (K, alpha, beta) with  Q[C] = K * m[kg]^alpha * v[km/s]^beta
    "fe_on_w": (0.7, 1.02, 3.48),
    "fe_on_al": (0.1, 1.00, 3.40),
    "al_on_dolomite": (1.5e-3, 1.00, 2.60),
}

_LAW_SOURCES = {
    "fe_on_w": "Close et al., Phys. Plasmas 20, 092102 (2013); "
               "McBride & McDonnell, Planet. Space Sci. 47, 1005 (1999).",
    "fe_on_al": "Ratcliff et al., Adv. Space Res. 20, 1471 (1997); "
                "Collette et al., J. Appl. Phys. 116, 084905 (2014).",
    "al_on_dolomite": "Crawford & Schultz, Int. J. Impact Eng. 14, 205 (1993); "
                      "ibid. 23, 169 (1999) [AVGR, porous carbonate].",
}


def empirical_charge_yield(mass_kg: float, v_m_s: float,
                           law: str = "fe_on_w") -> float:
    """Empirical impact-ionisation charge yield [C]: Q = K m^a v^b.

    m in kg, v in km/s.

    **Caveats.**  These fits come from different accelerators, target
    materials, chamber pressures and charge-collection geometries.  The
    *velocity exponent* is the robust, repeatedly-measured feature
    (3.4-3.5 for metal-on-metal, ~2.6 for porous silicates) and is what any
    first-principles model must reproduce; the *prefactor* carries roughly an
    order of magnitude of scatter and is strongly target dependent.  Treat
    absolute values from these laws as order-of-magnitude only.
    """
    if law not in _EMPIRICAL_LAWS:
        raise KeyError(f"Unknown law {law!r}. "
                       f"Available: {sorted(_EMPIRICAL_LAWS)}")
    K, a, b = _EMPIRICAL_LAWS[law]
    return K * mass_kg**a * (v_m_s / 1e3) ** b


def available_empirical_laws() -> dict:
    """Mapping of law name -> literature source."""
    return dict(_LAW_SOURCES)


__all__ = [
    "Projectile", "ImpactResult", "simulate_impact", "isobaric_core_radius",
    "empirical_charge_yield", "available_empirical_laws",
]
