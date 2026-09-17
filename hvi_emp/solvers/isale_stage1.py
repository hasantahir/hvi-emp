"""Stage 1 on iSALE: continuum shock-physics impact (the ALEGRA substitute).

Why iSALE
---------
Fletcher (NRL/MR/6757-20-10,138, 2021) used **ALEGRA** for the impact stage.
ALEGRA is a Sandia code and is not obtainable outside authorised users, so it
cannot be installed to reproduce that work.  The closest freely-available
equivalent is **iSALE** (Amsden/Ruppel/Hirt SALE lineage; Collins, Melosh,
Wunnemann, Elbeshausen, Davison), a multi-material multi-rheology shock
physics code maintained at MfN Berlin and Imperial College London.

iSALE covers what matters for Figs 4-5 of that report:

* Eulerian/ALE hydrodynamics with strength, damage and porosity models;
* ANEOS and Tillotson equations of state with phase change;
* axisymmetric (2D cylindrical) and 3D geometries, the latter for oblique
  impacts;
* Lagrangian tracers, so peak shock pressure per parcel is recorded and the
  melt/vapour inventory can be computed exactly as Fletcher does.

It is **not** open source: it is distributed free to academic users on
application (https://isale-code.github.io/access.html). Expect a few days for
the licence.

What iSALE does *not* give you
------------------------------
Two pieces of Fletcher's physics are outside iSALE:

1. **Ionisation state.** ALEGRA used the LMD conductivity model with pressure
   ionisation; iSALE has no ionisation model at all.  Post-process the iSALE
   tracer peak-pressure distribution through this framework's Saha machinery
   (`isale_tracers_to_handoff`) to recover T_e, Zbar and the charge yield.
2. **MHD / the diamagnetic cavity** (their Figs 6-7).  iSALE is pure
   hydrodynamics.  Use the OpenMHD bridge (`solvers.openmhd_stage2`) for
   that, initialised from the iSALE plume state.

Both substitutions are documented in docs/REPLICATION_FLETCHER2021.md with
what they cost you.

Equation of state -- the one real obstacle
------------------------------------------
iSALE ships EOS tables for the geologic and common engineering materials
(aluminium, iron, quartz, dunite, ice, ...).  **Tungsten is not standard.**
Options, in order of preference:

* use the aluminium-on-aluminium case, which Fletcher also ran (his Fig. 6 is
  Al->Al at 30 km/s, 30 deg) -- fully reproducible with shipped tables;
* obtain or build an ANEOS table for W (ANEOS input decks for refractory
  metals exist in the literature; iSALE accepts custom `aneos` tables);
* fit a Tillotson EOS for W and add it to `material.inp` -- crude at
  hypervelocity but adequate for crater scaling.

`FLETCHER_VELOCITIES` reproduces his six impact speeds.
"""

from __future__ import annotations

# --- running this file directly? ------------------------------------------
# Package module, not a script -- see the note in lammps_stage1.py.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    import sys as _sys
    _sys.exit(
        "\nisale_stage1.py is part of the `hvi_emp` package and cannot be run "
        "as a\nstandalone file -- it imports from its sibling modules.\n\n"
        "Keep it at  hvi_emp/solvers/isale_stage1.py  and use one of:\n\n"
        "    python -m hvi_emp.solvers.isale_stage1 --help\n"
        "    python examples/07_fletcher2021_replication.py\n\n"
        "or import it:\n\n"
        "    from hvi_emp.solvers.isale_stage1 import ISaleImpactConfig\n\n"
        "All of these must be run from the project root.\n")

import os
from dataclasses import dataclass, field

import numpy as np

from ..constants import AMU, EV, K_B

#: The six impact speeds of Fletcher (2021) Figs. 4 and 5 [m/s].
FLETCHER_VELOCITIES = (12e3, 22e3, 32e3, 42e3, 52e3, 62e3)

#: Snapshot times quoted in that report [s].
FLETCHER_T_DENSITY = 12e-6      # Fig. 4, mass density
FLETCHER_T_PRESSURE = 2.5e-6    # Fig. 5, pressure


# ---------------------------------------------------------------------------
# Material definitions for material.inp
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ISaleMaterial:
    """One entry in iSALE's ``material.inp``."""
    matname: str            # iSALE material label, e.g. "al-1100"
    eosname: str            # EOS table name, e.g. "aluminu"
    eostype: str = "tillo"  # "tillo" | "aneos"
    strmod: str = "JNCK"    # strength model
    dammod: str = "NONE"
    thsoft: str = "JNCK"
    pois: float = 0.33
    pmin: float = -2.44e9
    tmelt0: float = 933.0
    asimon: float = 6.0e9
    csimon: float = 3.0
    jc_a: float = 4.9e7
    jc_b: float = 1.57e8
    jc_n: float = 0.167
    jc_c: float = 0.016
    jc_m: float = 1.7
    jc_tref: float = 293.0

    def block(self) -> str:
        return f"""--------------------------------------------------------
MATNAME    Material name          : {self.matname}
EOSNAME    EOS name               : {self.eosname}
EOSTYPE    EOS type               : {self.eostype}
STRMOD     Strength model         : {self.strmod}
DAMMOD     Damage model           : {self.dammod}
ACFL       Acoustic fluidisation  : NONE
PORMOD     Porosity model         : NONE
THSOFT     Thermal softening      : {self.thsoft}
LDWEAK     Low density weakening  : NONE
----- Elastic strength parameters ----------------------
POIS       pois                   : {self.pois:.4E}
----- Minimum Pressure ---------------------------------
PMININ     minimum pressure       : {self.pmin:.3E}
----- Thermal parameters -------------------------------
TMELT0     tmelt0                 : {self.tmelt0:.4E}
ASIMON     a_simon                : {self.asimon:.4E}
CSIMON     c_simon                : {self.csimon:.4E}
----- Johnson-Cook strength parameters -----------------
JC_A       strain coeff. a        : {self.jc_a:.4E}
JC_B       strain coeff. b        : {self.jc_b:.4E}
JC_N       strain exponent        : {self.jc_n:.4E}
JC_C       str. rate coeff c      : {self.jc_c:.4E}
JC_M       thermal soft.          : {self.jc_m:.4E}
JC_TREF    ref. temperature       : {self.jc_tref:.4E}
--------------------------------------------------------
"""


#: Aluminium 1100-O, exactly as in the iSALE validation example (Pierazzo et
#: al. 2008 benchmark).  Ships with iSALE -- no extra EOS needed.
AL1100 = ISaleMaterial(
    matname="al-1100", eosname="aluminu", eostype="tillo",
    pois=0.33, pmin=-2.44e9, tmelt0=933.0,
    jc_a=4.9e7, jc_b=1.57e8, jc_n=0.167, jc_c=0.016, jc_m=1.7)

#: Tungsten -- Johnson-Cook constants from Meyers, *Dynamic Behavior of
#: Materials* (1994) / Johnson & Cook (1983) for W alloys.  **The EOS table is
#: the problem**: `wtungsten` is not a standard iSALE table. Supply an ANEOS
#: table or substitute a Tillotson fit; see the module docstring.
TUNGSTEN_ISALE = ISaleMaterial(
    matname="tungsten", eosname="wtungst", eostype="aneos",
    pois=0.28, pmin=-3.0e9, tmelt0=3695.0, asimon=6.0e9, csimon=3.0,
    jc_a=1.506e9, jc_b=1.766e8, jc_n=0.12, jc_c=0.016, jc_m=1.0,
    jc_tref=293.0)

ISALE_LIBRARY = {"al-1100": AL1100, "tungsten": TUNGSTEN_ISALE}


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class ISaleImpactConfig:
    """One iSALE impact, in the configuration of Fletcher (2021) Figs. 4-5.

    Parameters
    ----------
    velocity : float
        Impact speed [m/s], normal to the surface.
    projectile_diameter : float
        [m].  The NRL report does not state the projectile size; 1 mm is
        consistent with the ~12 us evolution and cm-scale plume it shows, and
        with the light-gas-gun/rail-gun experiments it is compared against.
        Treat it as a parameter of *your* replication, not of the paper, and
        say so when you present results.
    cppr : int
        Cells per projectile radius.  20 is the iSALE validation-example
        value; convergence for crater volume typically wants >= 20, and
        melt/vapour inventories want more (Pierazzo et al. 1997 recommend
        >= 20 CPPR for shock-pressure statistics).
    projectile_material, target_material : str
        Keys into `ISALE_LIBRARY`.
    t_end : float
        [s]. Defaults to the report's 12 us density-snapshot time.
    dt_save : float
        Output interval [s].
    grid_h, grid_v : int
        High-resolution zone cells (horizontal, vertical).
    ext_zone : int
        Extension-zone cells on each open side (cushions the boundaries).
    cyl : bool
        Cylindrical (axisymmetric) geometry.  True reproduces Figs. 4-5;
        set False and use iSALE3D for the oblique case of Fig. 6.
    """
    velocity: float = 32e3
    projectile_diameter: float = 1.0e-3
    cppr: int = 20
    projectile_material: str = "tungsten"
    target_material: str = "al-1100"
    t_end: float = FLETCHER_T_DENSITY
    dt_save: float = 1.0e-6
    grid_h: int = 200
    grid_v: int = 240
    ext_zone: int = 50
    cyl: bool = True
    surface_temperature: float = 293.0
    model_name: str = "fletcher2021"

    # -- derived ----------------------------------------------------------
    @property
    def grid_spacing(self) -> float:
        """Cell size [m] implied by the projectile radius and CPPR."""
        return 0.5 * self.projectile_diameter / self.cppr

    @property
    def target_depth(self) -> float:
        """Depth of the high-resolution target block [m]."""
        return self.grid_v * self.grid_spacing

    @property
    def dt_initial(self) -> float:
        """A stable first timestep: a small fraction of the cell transit."""
        return 0.2 * self.grid_spacing / max(self.velocity, 1.0)

    def crater_scaling_estimate(self) -> dict:
        """Crater-size estimate used to size the mesh.

        Cour-Palais penetration law for ductile metal targets, the standard
        hypervelocity-impact engineering correlation:

            P/d = 5.24 d^(1/19) BH^(-0.25) (rho_p/rho_t)^0.5 (v/c)^(2/3)

        with P the penetration depth, d the projectile diameter in cm, BH the
        target Brinell hardness and c the target bulk sound speed.  Crater
        diameter in a ductile metal is ~2P.  This is an order-of-magnitude
        *domain-sizing aid*, not a prediction -- the point is to catch a mesh
        that cannot contain the crater before you spend a day computing.

        The transient crater at the snapshot time is also reported: crater
        growth takes ~d_crater/c, so a run stopped at 12 us (Fletcher's Fig. 4)
        captures a crater that is still opening.  Sizing to the *final* crater
        is the safe choice and is what `auto_size()` uses.
        """
        rho_p = 19240.0 if self.projectile_material == "tungsten" else 2700.0
        rho_t, c_t, BH = 2700.0, 5100.0, 30.0
        d_cm = self.projectile_diameter * 100.0
        P_over_d = (5.24 * d_cm ** (1.0 / 19.0) * BH ** -0.25
                    * (rho_p / rho_t) ** 0.5
                    * (self.velocity / c_t) ** (2.0 / 3.0))
        d_crater = 2.0 * P_over_d * self.projectile_diameter
        t_grow = d_crater / c_t
        f_grown = min(1.0, self.t_end / t_grow) ** 0.5   # ~sqrt(t) opening
        return {
            "penetration_over_diameter": float(P_over_d),
            "d_crater_final": float(d_crater),
            "d_crater_over_d_proj": float(d_crater / self.projectile_diameter),
            "t_growth": float(t_grow),
            "d_crater_at_t_end": float(f_grown * d_crater),
            "cells_across_crater": float(d_crater / self.grid_spacing),
            "domain_width": float(self.grid_h * self.grid_spacing),
            "domain_depth": float(self.target_depth),
            "domain_adequate": bool(
                0.5 * d_crater < 0.8 * self.grid_h * self.grid_spacing
                and 0.5 * d_crater < 0.8 * self.target_depth),
        }

    def auto_size(self, margin: float = 1.6, max_cells: int = 3000) -> None:
        """Resize the mesh in place so it contains the crater.

        Axisymmetric, so the horizontal extent only needs the crater *radius*
        times `margin`; the vertical extent needs the depth plus the same
        margin.  Capped at `max_cells` per direction -- if the cap binds, the
        returned config will still report ``domain_adequate = False`` and you
        should reduce CPPR or the projectile size rather than run it.
        """
        est = self.crater_scaling_estimate()
        gs = self.grid_spacing
        self.grid_h = int(min(max_cells,
                              max(self.grid_h,
                                  np.ceil(margin * 0.5 * est["d_crater_final"]
                                          / gs))))
        self.grid_v = int(min(max_cells,
                              max(self.grid_v,
                                  np.ceil(margin * 0.5 * est["d_crater_final"]
                                          / gs))))

    #: Measured iSALE2D throughput [cell-updates/s/core], calibrated against
    #: the published aluminium_1100 example (200x240 cells, TEND 8.01e-5 s,
    #: documented as "several minutes").
    CELL_UPDATES_PER_SECOND = 1.5e6

    def cost_estimate(self, cores: int = 1, n_cases: int = 6) -> dict:
        """Runtime estimate for iSALE2D.

        iSALE2D is serial (one core per run), so a velocity series is
        embarrassingly parallel *across* runs -- which is how a many-core
        workstation should be used. Wall clock is therefore the slowest single
        case, not the sum, as long as cores >= n_cases.
        """
        cells = (self.grid_h + self.ext_zone) * (self.grid_v + self.ext_zone)
        c_s = 5000.0
        # CFL is set by the faster of sound speed and material velocity
        dt = 0.4 * self.grid_spacing / max(c_s, 0.25 * self.velocity)
        steps = self.t_end / dt
        cell_steps = cells * steps
        seconds = cell_steps / self.CELL_UPDATES_PER_SECOND
        concurrent = min(cores, n_cases)
        return {
            "cells": int(cells), "steps": int(steps),
            "cell_updates": float(cell_steps),
            "serial_hours": float(seconds / 3600.0),
            "wall_hours_series": float(seconds / 3600.0
                                       * np.ceil(n_cases / max(concurrent, 1))),
            "concurrent_cases": int(concurrent),
        }


# ---------------------------------------------------------------------------
# Deck generation
# ---------------------------------------------------------------------------

def write_asteroid_inp(cfg: ISaleImpactConfig, path: str | None = None) -> str:
    """Generate iSALE's ``asteroid.inp`` for this impact.

    Format follows the iSALE2D v4.1 input specification (the ``aluminum_1100``
    validation example).
    """
    gs = cfg.grid_spacing
    obj = ISALE_LIBRARY[cfg.projectile_material]
    tgt = ISALE_LIBRARY[cfg.target_material]

    text = f"""------------------- General Model Info ---------------------------------
VERSION               __DO NOT MODIFY__             : 4.1
DIMENSION             dimension of input file       : 2
PATH                  Data file path                : ./
MODEL                 Modelname                     : {cfg.model_name}_{cfg.velocity/1e3:.0f}kms
------------------- Mesh Geometry Parameters ---------------------------
GRIDH                 horizontal cells              : 0           : {cfg.grid_h}         : {cfg.ext_zone}
GRIDV                 vertical cells                : {cfg.ext_zone}          : {cfg.grid_v}         : 0
GRIDEXT               ext. factor                   : 1.05d0
GRIDSPC               grid spacing                  : {gs:.5E}
CYL                   Cylind. geometry              : {1.0 if cfg.cyl else 0.0:.1f}D0
GRIDSPCM              max. grid spacing             : {20*gs:.3E}
------------------- Global setup parameters -----------------------------
S_TYPE                setup type                    : DEFAULT
GRAD_TYPE             gradient type                 : NONE
T_SURF                Surface temp                  : {cfg.surface_temperature:.1f}D0
------------------- Projectile ("Object") Parameters --------------------
OBJNUM                number of objects             : 1
OBJRESH               CPPR horizontal               : {cfg.cppr}
OBJVEL                object velocity               : {-cfg.velocity:.4E}
OBJMAT                object material               : {obj.matname}
OBJTYPE               object type                   : SPHEROID
------------------- Target Parameters ----------------------------------
LAYNUM                layers number                 : 1
LAYPOS                layer position                : {cfg.grid_v}
LAYMAT                layer material                : {tgt.matname}
------------------- Time Parameters ------------------------------------
DT                    initial time increment        : {cfg.dt_initial:.4E}
DTMAX                 maximum timestep              : 5.D-3
TEND                  end time                      : {cfg.t_end:.4E}
DTSAVE                save interval                 : {cfg.dt_save:.4E}
------------------- Boundary Condition Parameters ----------------------
BND_L                 left                          : FREESLIP
BND_R                 right                         : FREESLIP
BND_B                 bottom                        : NOSLIP
BND_T                 top                           : OUTFLOW
------------------- Numerical Stability Parameters ---------------------
AVIS                  art. visc. linear             : 0.2D0
AVIS2                 art. visc. quad.              : 1.0D0
------------------- Tracer Particle Parameters -------------------------
TR_QUAL               integration qual.             : 1
TR_SPCH               tracer spacing X              : {gs:.5E}   : {gs:.5E}
TR_SPCV               tracer spacing Y              : {gs:.5E}   : {gs:.5E}
TR_VAR                add. tracer fiels             : #TrP-TrT-TrD#
------------------- (Material) Model parameters (global) ---------------
STRESS                Consider stress               : 1
PARTPRES              Pres. in part.                : 1
------------------- Data Saving Parameters -----------------------------
QUALITY               Compression rate              : -50
VARLIST               List of variables             : #Den-Tmp-Pre-Sie-Yld-VEL#
------------------------------------------------------------------------
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def write_material_inp(cfg: ISaleImpactConfig, path: str | None = None) -> str:
    """Generate iSALE's ``material.inp`` for the two materials."""
    mats = [ISALE_LIBRARY[cfg.projectile_material]]
    if cfg.target_material != cfg.projectile_material:
        mats.append(ISALE_LIBRARY[cfg.target_material])
    text = "".join(m.block() for m in mats)
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def write_velocity_series(directory: str,
                          velocities=FLETCHER_VELOCITIES,
                          cores: int = 128,
                          **cfg_kw) -> list:
    """Write one iSALE run directory per impact speed (Fletcher Figs. 4-5).

    Creates ``<directory>/v12kms/``, ``v22kms/`` ... each containing
    ``asteroid.inp`` and ``material.inp``, plus a ``run_all.sh`` that fires
    them off concurrently -- the right pattern for a many-core workstation,
    since iSALE2D itself is serial.
    """
    os.makedirs(directory, exist_ok=True)
    made = []
    for v in velocities:
        cfg = ISaleImpactConfig(velocity=v, **cfg_kw)
        cfg.auto_size()
        sub = os.path.join(directory, f"v{v/1e3:.0f}kms")
        os.makedirs(sub, exist_ok=True)
        write_asteroid_inp(cfg, os.path.join(sub, "asteroid.inp"))
        write_material_inp(cfg, os.path.join(sub, "material.inp"))
        made.append({"velocity": v, "dir": sub, "config": cfg,
                     "cost": cfg.cost_estimate(cores=cores,
                                               n_cases=len(velocities)),
                     "scaling": cfg.crater_scaling_estimate()})

    runner = os.path.join(directory, "run_all.sh")
    with open(runner, "w") as fh:
        fh.write("#!/usr/bin/env bash\n"
                 "# Fletcher (2021) Figs. 4-5 velocity series.\n"
                 "# iSALE2D is serial, so run the six cases concurrently.\n"
                 "# Point ISALE at your build, then:  bash run_all.sh\n"
                 'ISALE="${ISALE:-$HOME/iSALE/build/iSALE2D}"\n\n')
        for m in made:
            d = os.path.basename(m["dir"])
            fh.write(f'( cd {d} && "$ISALE" > isale.log 2>&1 ) &\n')
        fh.write("\nwait\n"
                 'echo "all runs finished"\n')
    os.chmod(runner, 0o755)
    return made


# ---------------------------------------------------------------------------
# Post-processing: iSALE tracers -> this framework's ionisation model
# ---------------------------------------------------------------------------

def isale_tracers_to_handoff(peak_pressures, tracer_mass, material="Al",
                             plume_radius=None, expansion_velocity=None):
    """Turn an iSALE tracer peak-pressure distribution into a plasma state.

    iSALE has no ionisation model, so this supplies the step ALEGRA did with
    its LMD conductivity model: each Lagrangian tracer's *peak shock pressure*
    is put through this framework's release/waste-heat calculation, the
    superheated fraction is identified, and Saha equilibrium gives T_e and
    Zbar.  This is the standard way melt/vapour inventories are extracted from
    iSALE (Pierazzo, Vickery & Melosh 1997) with the ionisation step added.

    Parameters
    ----------
    peak_pressures : array_like
        Peak shock pressure recorded by each tracer [Pa] (iSALE's ``TrP``).
    tracer_mass : float or array_like
        Mass represented by each tracer [kg].
    material : str
        Framework material name for the tracer population.
    plume_radius, expansion_velocity : float, optional
        Override the Stage-2 seed; otherwise estimated from the vapour mass.

    Returns
    -------
    dict with the vapour/plasma inventory and an `ImpactResult`-compatible
    handoff usable by `hvi_emp.simulate_expansion`.
    """
    from ..constants import E_CHARGE
    from ..eos import phase_thresholds, residual_energy_from_pressure, \
        vapour_fraction
    from ..impact import ImpactResult, Projectile
    from ..ionization import IonisationState, saha_solve
    from ..materials import get_material

    mat = get_material(material)
    P = np.atleast_1d(np.asarray(peak_pressures, dtype=float))
    m = np.broadcast_to(np.atleast_1d(np.asarray(tracer_mass, dtype=float)),
                        P.shape)
    th = phase_thresholds(mat)

    E_res = np.array([residual_energy_from_pressure(mat, p) for p in P])
    f_vap = np.array([vapour_fraction(mat, e) for e in E_res])
    hot = E_res >= th["E_vap"]                 # superheated -> plasma-forming

    m_vapour = float(np.sum(f_vap * m))
    m_plasma = float(np.sum(m[hot]))
    m_total = float(np.sum(m))

    if m_plasma > 0:
        u_specific = float(np.average(
            np.maximum(E_res[hot] - mat.E_sublimation, 0.0),
            weights=m[hot]))
    else:
        u_specific = 0.0
    u_atom = u_specific * mat.m_atom

    if plume_radius is None:
        from ..eos import spinodal_volume
        V = m_plasma * 27.0 * spinodal_volume(mat) if m_plasma > 0 else 0.0
        plume_radius = ((3.0 * V) / (2.0 * np.pi)) ** (1/3) if V > 0 else 1e-4
    rho = (m_plasma / ((2.0 / 3.0) * np.pi * plume_radius**3)
           if m_plasma > 0 else 0.0)
    n_h = rho / mat.m_atom if rho > 0 else 0.0

    if n_h > 0 and u_atom > 0:
        from ..ionization import get_table
        T0, _ = get_table(mat).state(n_h, u_atom)
        plasma = saha_solve(mat, T0, n_h)
    else:
        z = np.zeros(len(mat.E_ion) + 1)
        z[0] = 1.0
        plasma = IonisationState(300.0, n_h, 0.0, 0.0, z, 0.0)

    Q = E_CHARGE * plasma.Zbar * (m_plasma / mat.m_atom if m_plasma else 0.0)
    v_exp = expansion_velocity or 5e3

    handoff = ImpactResult(
        projectile=Projectile(mat, mass=max(m_total, mat.m_atom),
                              velocity=v_exp * 2.0),
        target=mat, state=None, P_ic=float(np.max(P)) if P.size else 0.0,
        r_ic=plume_radius,
        m_vapour_projectile=0.0, m_vapour_target=m_vapour,
        m_vapour=m_vapour, m_plasma=m_plasma, m_melt_target=0.0,
        f_vap_projectile=float("nan"),
        E_res_projectile=float("nan"), E_res_core=float(np.max(E_res))
        if E_res.size else 0.0,
        E_vapour_specific=u_specific, u_atom=u_atom,
        material_mix=mat, plasma=plasma, Q_free=Q, n_h0=n_h,
        rho_plume0=rho, r_plume0=plume_radius, v_expansion=v_exp,
        thresholds={"target": th},
        diagnostics={"source": "isale", "n_tracers": int(P.size),
                     "m_total": m_total, "f_hot": float(hot.mean())},
    )
    return {
        "m_total": m_total, "m_vapour": m_vapour, "m_plasma": m_plasma,
        "vapour_fraction": m_vapour / m_total if m_total else 0.0,
        "T_eV": plasma.T_eV, "Zbar": plasma.Zbar, "Q_free": Q,
        "handoff": handoff,
    }


def read_isale_tracers(jdata_path: str):        # pragma: no cover - needs data
    """Read tracer peak pressures from an iSALE ``jdata.dat`` via pySALEPlot.

    pySALEPlot ships with iSALE. Kept thin deliberately: it exists so the
    post-processing path is obvious, not to re-implement their reader.
    """
    try:
        import pySALEPlot as psp
    except ImportError as exc:
        raise ImportError(
            "pySALEPlot not found. It ships with iSALE; add its directory to "
            "PYTHONPATH (usually <isale>/Plotting/pySALEPlot)."
        ) from exc

    model = psp.opendatfile(jdata_path)
    step = model.readStep(["TrP"], model.nsteps - 1)
    return np.asarray(step.TrP), model


# ---------------------------------------------------------------------------
# Module demo
# ---------------------------------------------------------------------------

def _demo(argv=None) -> int:                     # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m hvi_emp.solvers.isale_stage1",
        description="Generate iSALE decks for the Fletcher (2021) velocity "
                    "series.")
    ap.add_argument("--out", default="isale_fletcher2021",
                    help="output directory")
    ap.add_argument("--diameter", type=float, default=1.0e-3,
                    help="projectile diameter [m]")
    ap.add_argument("--cppr", type=int, default=20)
    ap.add_argument("--projectile", default="tungsten",
                    choices=sorted(ISALE_LIBRARY))
    ap.add_argument("--target", default="al-1100",
                    choices=sorted(ISALE_LIBRARY))
    ap.add_argument("--cores", type=int, default=128)
    a = ap.parse_args(argv)

    made = write_velocity_series(
        a.out, cores=a.cores, projectile_diameter=a.diameter, cppr=a.cppr,
        projectile_material=a.projectile, target_material=a.target)

    print(f"wrote {len(made)} iSALE run directories under {a.out}/\n")
    print(f"{'v [km/s]':>9}{'grid':>12}{'cells':>10}{'steps':>9}"
          f"{'serial h':>10}{'crater/d_p':>12}{'domain ok':>11}")
    wall = 0.0
    for m in made:
        c, sc, cf = m["cost"], m["scaling"], m["config"]
        wall = max(wall, c["wall_hours_series"])
        print(f"{m['velocity']/1e3:>9.0f}{f'{cf.grid_h}x{cf.grid_v}':>12}"
              f"{c['cells']:>10d}{c['steps']:>9d}"
              f"{c['serial_hours']:>10.2f}{sc['d_crater_over_d_proj']:>12.1f}"
              f"{str(sc['domain_adequate']):>11}")
    print(f"\n{len(made)} cases, {made[0]['cost']['concurrent_cases']} "
          f"concurrent on {a.cores} cores -> ~{wall:.1f} h wall clock "
          "(iSALE2D is serial per run)")
    print(f"\n  cd {a.out} && bash run_all.sh")
    return 0


if __name__ == "__main__":                       # pragma: no cover
    import sys as _s
    _s.exit(_demo())


__all__ = [
    "ISaleMaterial", "ISaleImpactConfig", "ISALE_LIBRARY",
    "AL1100", "TUNGSTEN_ISALE", "FLETCHER_VELOCITIES",
    "FLETCHER_T_DENSITY", "FLETCHER_T_PRESSURE",
    "write_asteroid_inp", "write_material_inp", "write_velocity_series",
    "isale_tracers_to_handoff", "read_isale_tracers",
]
