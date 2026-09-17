"""Stage 1 on LAMMPS: molecular-dynamics hypervelocity impact.

Methodology
-----------
The simulation protocol follows Fraile, Dwivedi, Bonny & Polcar, "Analysis of
hypervelocity impacts: the tungsten case", Nucl. Fusion 62, 026034 (2022),
who ran W-on-W impacts up to 9 km/s with up to 4e7 atoms in LAMMPS:

* single-crystal target, (001) surface normal to the impact direction;
* periodic boundaries in x and y, free surfaces in z, with a vacuum gap
  above the surface large enough that ejecta do not re-enter through the
  periodic images;
* EAM interatomic forces (they use Marinica "EAM4" and Olsson; ZBL-screened
  at short range for the collisional core);
* conjugate-gradient minimisation, then equilibration at 300 K, then the
  production run in the **NVE** ensemble with the projectile given a uniform
  downward velocity;
* timestep 0.1 fs for energy conservation at impact speeds;
* optional `fix viscous` damping walls at the lateral edges (they verified
  these do not change results for a large enough target);
* their unit check: KE per atom = 0.952 eV x (v [km/s])^2 for tungsten.

Potentials
----------
The official LAMMPS wheel ships `W_zhou.eam.alloy` (Zhou et al., Phys. Rev. B
69, 144113 (2004)), which this bridge uses as the tungsten default so that it
runs out of the box.  Fraile et al. used the Marinica EAM4 potential; drop the
file in and pass ``potential_file=`` to reproduce their setup exactly (the
NIST Interatomic Potentials Repository hosts it: search "2013 Marinica W").
Al, Cu, Fe and Ni defaults also ship with the wheel.

Scale honesty
-------------
MD reaches nm-scale projectiles (Fraile: r ~ 1-13 nm) and 100 ps.  A real
micrometeoroid is micrometres and the plume evolves over microseconds.  The
bridge therefore serves two purposes: (i) resolving the *physics* of the
shock/vaporisation stage that the reduced model closes with the planar-impact
+ release approximation, at the scale where MD is exact; and (ii) providing
nm-scale anchor points against which the reduced model's scaling in size can
be checked (`compare_with_reduced_model`).  It does not replace Stage 1 at
projectile masses beyond ~1e-18 kg.

Ionisation
----------
Classical EAM MD has no electrons; it cannot ionise.  The handoff to Stage 2
therefore takes the *thermodynamic* state of the ejecta (mass, temperature,
velocity) from MD and assigns the charge state with the same Saha machinery
used by the reduced model.  This mirrors the standard practice of the
hydrocode -> PIC waterfall (Fletcher 2021), with MD in place of the hydrocode.
"""

from __future__ import annotations

# --- running this file directly? ------------------------------------------
# This is a package module, not a standalone script: the `..` imports below
# only resolve when it is imported as `hvi_emp.solvers.<name>`. Executing the
# file by path (`python lammps_stage1.py`) -- or after copying it somewhere else --
# fails at those imports with a bare "attempted relative import with no known
# parent package". Catch that here and say something useful instead.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    import sys as _sys
    _sys.exit(
        "\nlammps_stage1.py is part of the `hvi_emp` package and cannot be run as a\n"
        "standalone file -- it imports from its sibling modules.\n\n"
        "Keep it at  hvi_emp/solvers/lammps_stage1.py  and use one of:\n\n"
        "    python -m hvi_emp.solvers.lammps_stage1      # this module's own demo\n"
        "    python -m hvi_emp lammps --help       # the command-line front end\n"
        "    python examples/05_lammps_impact.py   # the worked example\n\n"
        "or import it:\n\n"
        "    from hvi_emp.solvers.lammps_stage1 import LammpsImpactConfig, run_impact\n\n"
        "All of these must be run from the project root (the directory\n"
        "containing the `hvi_emp/` folder).\n")


import os
from dataclasses import dataclass, field

import numpy as np

from ..constants import AMU, EV, K_B
from ..materials import Material, get_material
from . import _preload_mpi

# ---------------------------------------------------------------------------
# MD-specific material data: lattice + bundled potential
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MDMaterial:
    """Lattice and potential information needed by the MD bridge."""
    symbol: str
    lattice: str           # "bcc" | "fcc"
    a0: float              # lattice constant [Angstrom]
    mass_amu: float
    bundled_potential: str        # file name inside the LAMMPS potentials dir
    pair_style: str = "eam/alloy"


MD_LIBRARY: dict[str, MDMaterial] = {
    "W": MDMaterial("W", "bcc", 3.1652, 183.84, "W_zhou.eam.alloy"),
    "Al": MDMaterial("Al", "fcc", 4.0495, 26.9815, "Al_zhou.eam.alloy"),
    "Cu": MDMaterial("Cu", "fcc", 3.6150, 63.546, "Cu_mishin1.eam.alloy"),
    "Fe": MDMaterial("Fe", "bcc", 2.8553, 55.845, "Fe_mm.eam.fs", "eam/fs"),
    "Ni": MDMaterial("Ni", "fcc", 3.5200, 58.693, "Ni_smf7.eam", "eam"),
}


def bundled_potential_dir() -> str | None:
    """Path to the potentials directory shipped with the LAMMPS wheel."""
    try:
        from . import import_lammps
        _l = import_lammps()
        p = os.path.join(os.path.dirname(_l.__file__),
                         "share", "lammps", "potentials")
        return p if os.path.isdir(p) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Single-core EAM throughput, in atom-steps per second.
#:
#: **Measured, on tungsten, with this bridge's own deck.** Three sizes on one
#: core gave 4.79e5, 4.89e5 and 4.99e5 atom-steps/s at 3.9k, 11.7k and 25.2k
#: atoms -- flat to 4%, which is what you expect from EAM once the neighbour
#: list dominates, and flat enough to extrapolate from.
#:
#: It was measured on an aarch64 core, so treat it as a **floor** for a
#: modern x86 workstation core rather than a target. If your box is faster,
#: the estimates here are pessimistic, which is the right direction for a
#: number someone plans a weekend around.
ATOM_STEPS_PER_SEC_CORE = 4.9e5

#: MPI parallel efficiency assumed at high rank counts. **Not measured** --
#: it cannot be, from one rank. EAM with good domain decomposition typically
#: holds 0.8-0.9 to 64 ranks on a single node; `estimate_cost` returns
#: `efficiency_is_assumed=True` so this never passes for data.
MPI_EFFICIENCY = 0.85

#: Resident memory per atom for EAM with neighbour lists [bytes]. LAMMPS's
#: own guidance is ~100 B/atom for simple pair styles; EAM plus the neighbour
#: list and per-atom computes lands nearer 250.
BYTES_PER_ATOM = 250.0

#: Named sizes, so "beef it up" is a flag rather than an afternoon of
#: arithmetic. `cells` is the target block edge in unit cells; tungsten is
#: bcc, so the atom count is about `1.4 * cells^3` plus the projectile.
#:
#: The wall figures are for 64 cores at the measured rate and assumed
#: efficiency. They scale linearly in `atoms x steps`, so halving `ps` halves
#: them.
#:
#: ``fraile`` reproduces the largest run in this literature -- Fraile et al.
#: (Nucl. Fusion 62, 026034, 2022) ran 4e7 atoms. It is a weekend job, not an
#: afternoon one.
MD_SCALES = {
    "smoke": {"cells": 26, "radius": 0.8e-9, "ps": 4.0,
              "for": "does the machinery work at all"},
    "talk": {"cells": 80, "radius": 2.0e-9, "ps": 15.0,
             "for": "looks like a real impact in a slide"},
    "detailed": {"cells": 140, "radius": 3.5e-9, "ps": 25.0,
                 "for": "resolved ejecta and fragment statistics"},
    "paper": {"cells": 200, "radius": 5.0e-9, "ps": 40.0,
              "for": "publishable ejecta size distribution"},
    "fraile": {"cells": 306, "radius": 13.0e-9, "ps": 100.0,
               "for": "the largest run in this literature; a weekend"},
}


@dataclass
class LammpsImpactConfig:
    """One MD impact, in the Fraile et al. configuration.

    Parameters
    ----------
    material : str
        Element symbol; projectile and target are the same species (as in the
        paper).  Must be in `MD_LIBRARY` unless `potential_file`,
        `pair_style`, `lattice`, `a0` and `mass_amu` are all given.
    velocity : float
        Impact speed [m/s], along -z (surface normal).
    projectile_radius : float
        Sphere radius [m].  Fraile et al. span ~0.5-13 nm.
    target_cells : tuple
        Target size in unit cells (nx, ny, nz).  If None, sized automatically:
        lateral 6x the projectile diameter, depth 8x the projectile radius
        plus penetration allowance -- the paper's criterion of "large enough
        that boundaries do not matter", scaled down.
    timestep : float
        [s]; the paper uses 1e-16 (0.1 fs).
    t_equilibrate, t_production : float
        [s]; the paper uses 20 ps and 100 ps.  Defaults are shorter because
        the bridge's default use is smoke-scale; pass the paper's values for
        production runs.
    temperature : float
        Initial target temperature [K].
    damping_walls : bool
        Add `fix viscous` damping strips at the x/y edges (the paper's
        checked-and-optional shock absorber).
    vacuum : float
        Vacuum gap above the surface [m].
    potential_file : str, optional
        Override the bundled potential (e.g. the Marinica EAM4 file).
    seed : int
        Velocity-initialisation RNG seed.
    """
    material: str = "W"
    velocity: float = 9.0e3
    projectile_radius: float = 1.3e-9
    target_cells: tuple | None = None
    timestep: float = 1.0e-16
    t_equilibrate: float = 2.0e-12
    t_production: float = 1.0e-11
    temperature: float = 300.0
    damping_walls: bool = False
    damping_gamma: float = 0.2
    vacuum: float = 8.0e-9
    potential_file: str | None = None
    pair_style: str | None = None
    lattice: str | None = None
    a0: float | None = None
    mass_amu: float | None = None
    seed: int = 42
    dump_every: int = 0            # 0 = no trajectory dump
    dump_path: str = "impact.dump"

    # -- resolved views ---------------------------------------------------
    def md(self) -> MDMaterial:
        if self.material in MD_LIBRARY:
            base = MD_LIBRARY[self.material]
            return MDMaterial(
                base.symbol, self.lattice or base.lattice,
                self.a0 or base.a0, self.mass_amu or base.mass_amu,
                self.potential_file or base.bundled_potential,
                self.pair_style or base.pair_style)
        missing = [k for k in ("potential_file", "pair_style", "lattice",
                               "a0", "mass_amu") if getattr(self, k) is None]
        if missing:
            raise KeyError(
                f"{self.material!r} is not in MD_LIBRARY; provide {missing}")
        return MDMaterial(self.material, self.lattice, self.a0,
                          self.mass_amu, self.potential_file, self.pair_style)

    def resolved_cells(self) -> tuple:
        if self.target_cells is not None:
            return tuple(self.target_cells)
        md = self.md()
        a0_m = md.a0 * 1e-10
        r_cells = self.projectile_radius / a0_m
        lateral = max(int(np.ceil(12.0 * r_cells)), 12)
        depth = max(int(np.ceil(8.0 * r_cells)), 10)
        return (lateral, lateral, depth)

    @property
    def ke_per_atom_eV(self) -> float:
        """Impact kinetic energy per projectile atom [eV].

        For tungsten this is the paper's stated 0.952 eV x (v [km/s])^2.
        """
        md = self.md()
        return 0.5 * md.mass_amu * AMU * self.velocity**2 / EV

    def estimated_atom_count(self) -> int:
        md = self.md()
        per_cell = 2 if md.lattice == "bcc" else 4
        nx, ny, nz = self.resolved_cells()
        n_target = per_cell * nx * ny * nz
        a0_m = md.a0 * 1e-10
        n_proj = int(per_cell * (4.0 / 3.0) * np.pi
                     * (self.projectile_radius / a0_m) ** 3)
        return n_target + n_proj

    def estimate_cost(self, cores: int = 64,
                      mpi_efficiency: float = MPI_EFFICIENCY) -> dict:
        """Atoms, steps, memory and wall time.

        The throughput constant is **measured**, not guessed -- see
        `ATOM_STEPS_PER_SEC_CORE`. The parallel efficiency is *not*: it is an
        assumption, flagged as such in the return value, because this was
        calibrated on a single core and MPI scaling cannot be measured from
        one rank.
        """
        n = self.estimated_atom_count()
        steps = int(round((self.t_equilibrate + self.t_production)
                          / self.timestep))
        atom_steps = float(n) * steps
        rate = ATOM_STEPS_PER_SEC_CORE * max(cores, 1) * mpi_efficiency
        return {
            "atoms": n,
            "steps": steps,
            "atom_steps": atom_steps,
            "memory_GB": n * BYTES_PER_ATOM / 1e9,
            "core_seconds": atom_steps / ATOM_STEPS_PER_SEC_CORE,
            "wall_seconds": atom_steps / rate,
            "wall_hours": atom_steps / rate / 3600.0,
            "cores": cores,
            "rate_is_measured": True,
            "efficiency_is_assumed": True,
        }


# ---------------------------------------------------------------------------
# Input-deck generation (always available, no LAMMPS required)
# ---------------------------------------------------------------------------

def config_for_scale(scale: str = "talk", material: str = "W",
                     velocity: float = 9.0e3,
                     timestep: float = 5.0e-16,
                     **overrides) -> LammpsImpactConfig:
    """A `LammpsImpactConfig` at one of the named `MD_SCALES`.

    The default timestep here is 0.5 fs rather than the 0.2 fs the smoke
    configuration uses. 0.2 fs is the safe choice while the shock front is
    passing; over tens of picoseconds it costs 2.5x for accuracy that the
    ejecta statistics do not need. Pass `timestep=2e-16` back if you are
    resolving the shock itself.
    """
    if scale not in MD_SCALES:
        raise KeyError(f"unknown scale {scale!r}; have {sorted(MD_SCALES)}")
    spec = MD_SCALES[scale]
    n = spec["cells"]
    kw = dict(material=material, velocity=velocity,
              projectile_radius=spec["radius"],
              target_cells=(n, n, int(n * 0.7)),
              t_equilibrate=2e-13,
              t_production=spec["ps"] * 1e-12,
              timestep=timestep)
    kw.update(overrides)
    return LammpsImpactConfig(**kw)


def scale_table(cores: int = 64, material: str = "W") -> list:
    """Every named scale with its measured-rate cost estimate."""
    rows = []
    for name in MD_SCALES:
        cfg = config_for_scale(name, material=material)
        est = cfg.estimate_cost(cores=cores)
        rows.append({"scale": name, "for": MD_SCALES[name]["for"],
                     "cells": MD_SCALES[name]["cells"],
                     "radius_nm": MD_SCALES[name]["radius"] * 1e9,
                     "ps": MD_SCALES[name]["ps"], **est})
    return rows


def generate_input_deck(cfg: LammpsImpactConfig,
                        potential_dir: str | None = None) -> str:
    """Complete LAMMPS input deck implementing the Fraile et al. protocol.

    The returned string is a standalone ``in.impact`` file for the LAMMPS
    executable; `run_impact` feeds the same commands through the Python
    interface.
    """
    md = cfg.md()
    nx, ny, nz = cfg.resolved_cells()
    a0 = md.a0                                    # Angstrom, metal units
    # first-neighbour distance: bcc a0*sqrt(3)/2, fcc a0/sqrt(2); +20%
    r_coord = (a0 * 0.8660 if md.lattice == "bcc" else a0 * 0.7071) * 1.2
    r_A = cfg.projectile_radius * 1e10
    vac_A = cfg.vacuum * 1e10
    v_Aps = -cfg.velocity * 1e10 * 1e-12          # m/s -> Angstrom/ps
    dt_ps = cfg.timestep * 1e12
    n_eq = max(int(round(cfg.t_equilibrate / cfg.timestep)), 1)
    n_run = max(int(round(cfg.t_production / cfg.timestep)), 1)

    pot_dir = potential_dir or bundled_potential_dir() or "."
    pot_path = (cfg.potential_file if cfg.potential_file and
                os.path.isabs(cfg.potential_file)
                else os.path.join(pot_dir, md.bundled_potential))

    z_top = nz * a0
    z_gap = 2.0                                   # standoff before release
    zc = z_top + z_gap + r_A                      # projectile centre

    pair_coeff = {
        "eam/alloy": f"pair_coeff      * * {pot_path} {md.symbol} {md.symbol}",
        "eam/fs":    f"pair_coeff      * * {pot_path} {md.symbol} {md.symbol}",
        "eam":       f"pair_coeff      * * {pot_path}",
    }[md.pair_style]

    lines = f"""# Hypervelocity impact, protocol of Fraile et al., Nucl. Fusion 62, 026034 (2022)
# {md.symbol} sphere (r = {r_A:.1f} A) -> {md.symbol} ({md.lattice}) target at {cfg.velocity/1e3:.1f} km/s
# KE = {cfg.ke_per_atom_eV:.3f} eV/atom   (paper check: 0.952 eV x v[km/s]^2 for W)
units           metal
dimension       3
# z is shrink-wrapped (m): the box follows the ejecta instead of deleting
# atoms that outrun a fixed vacuum gap -- the automatic version of the
# paper's "sufficiently large vacuum in z" requirement.
boundary        p p m
atom_style      atomic

lattice         {md.lattice} {a0:.4f}
# target block; free surface at z = {z_top:.1f} A, vacuum above
region          simbox block 0 {nx} 0 {ny} 0 {nz + (2*r_A + z_gap + vac_A)/a0:.3f} units lattice
create_box      2 simbox
region          target block 0 {nx} 0 {ny} 0 {nz} units lattice
create_atoms    1 region target

# spherical projectile above the surface
region          proj sphere {nx*a0/2.0:.3f} {ny*a0/2.0:.3f} {zc:.3f} {r_A:.3f} units box
create_atoms    2 region proj

mass            1 {md.mass_amu:.4f}
mass            2 {md.mass_amu:.4f}

pair_style      {md.pair_style}
{pair_coeff}

group           gtarget type 1
group           gproj type 2

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# --- minimise, then equilibrate the target at {cfg.temperature:.0f} K ------------------
minimize        1.0e-6 1.0e-8 1000 10000

velocity        gtarget create {cfg.temperature:.1f} {cfg.seed} mom yes rot yes
timestep        {dt_ps:.6f}
fix             feq gtarget nvt temp {cfg.temperature:.1f} {cfg.temperature:.1f} $(100.0*dt)
run             {n_eq}
unfix           feq

# --- impact: uniform downward velocity, microcanonical ensemble -----------
velocity        gproj set 0.0 0.0 {v_Aps:.4f} units box
fix             fnve all nve
"""
    if cfg.damping_walls:
        wall = max(2.0 * a0, 6.0)
        lines += f"""
# damped strips at the lateral edges (fix viscous, gamma = {cfg.damping_gamma})
region          rwallx1 block 0 {wall/a0:.2f} 0 {ny} 0 {nz} units lattice
region          rwallx2 block {nx - wall/a0:.2f} {nx} 0 {ny} 0 {nz} units lattice
region          rwally1 block 0 {nx} 0 {wall/a0:.2f} 0 {nz} units lattice
region          rwally2 block 0 {nx} {ny - wall/a0:.2f} {ny} 0 {nz} units lattice
region          rwalls union 4 rwallx1 rwallx2 rwally1 rwally2
group           gwalls region rwalls
fix             fdamp gwalls viscous {cfg.damping_gamma}
"""
    lines += f"""
# coordination cutoff: first-neighbour shell + 20%, so a perfect
# lattice gives the full coordination number and damaged/ejected
# atoms give less. That contrast is what draws the crater.
compute         ke all ke/atom
compute         pe all pe/atom
compute         coord all coord/atom cutoff {r_coord:.4f}
compute         Tall all temp
thermo_style    custom step time temp pe ke etotal press
thermo_modify   lost warn
thermo          {max(n_run // 20, 1)}
"""
    if cfg.dump_every > 0:
        # Per-atom KE, PE and coordination are what make the trajectory
        # *interpretable*: coordination shows lattice damage and the crater
        # lip, and the velocities let lammps_postprocess reconstruct a local
        # temperature (dispersion about the cell mean, not raw KE) to feed
        # Saha. Positions alone give a pretty picture and nothing else.
        lines += (f"dump            d1 all custom {cfg.dump_every} "
                  f"{cfg.dump_path} id type x y z vx vy vz "
                  f"c_ke c_pe c_coord\n"
                  f"dump_modify     d1 sort id\n")
    lines += f"""
run             {n_run}
"""
    return lines


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class LammpsImpactResult:
    """Post-processed outcome of one MD impact."""
    config: LammpsImpactConfig
    n_atoms: int
    n_ejecta: int                  # atoms above the surface, moving up
    m_ejecta: float                # kg
    v_ejecta_mean: float           # m/s, mass-weighted upward speed
    v_ejecta_std: float            # m/s
    T_ejecta: float                # K, from c.o.m.-frame velocity variance
    T_target_final: float          # K
    ke_initial: float              # J, projectile kinetic energy
    energy_drift: float            # relative total-energy drift over the run
    z_surface: float               # m
    positions: np.ndarray | None = None   # ejecta positions [m]
    velocities: np.ndarray | None = None  # ejecta velocities [m/s]
    diagnostics: dict = field(default_factory=dict)

    @property
    def ejecta_fraction_of_projectile(self) -> float:
        """Ejecta mass over projectile mass -- compare with the reduced
        model's vaporisation efficiency."""
        md = self.config.md()
        m_p = (self.diagnostics.get("n_projectile", 0) * md.mass_amu * AMU)
        return self.m_ejecta / m_p if m_p > 0 else 0.0

    def summary(self) -> str:
        c = self.config
        return "\n".join([
            f"LAMMPS impact: {c.material} r = {c.projectile_radius*1e9:.2f} nm "
            f"at {c.velocity/1e3:.1f} km/s "
            f"({self.diagnostics.get('n_projectile', 0)} projectile atoms, "
            f"{self.n_atoms} total)",
            f"  KE per projectile atom  {c.ke_per_atom_eV:10.3f} eV",
            f"  ejecta atoms            {self.n_ejecta:10d} "
            f"({self.ejecta_fraction_of_projectile:.2f} x m_proj)",
            f"  ejecta mean speed       {self.v_ejecta_mean/1e3:10.2f} km/s",
            f"  ejecta temperature      {self.T_ejecta:10.0f} K "
            f"({self.T_ejecta * K_B / EV:.3f} eV)",
            f"  final target T          {self.T_target_final:10.0f} K",
            f"  energy drift            {self.energy_drift:10.2e}",
        ])

    # -- Stage-2 handoff --------------------------------------------------
    def to_impact_handoff(self, framework_material: Material | None = None):
        """Duck-typed `ImpactResult` for `hvi_emp.simulate_expansion`.

        MD gives mass, temperature and bulk velocity of the ejecta; the
        charge state is assigned by the same Saha machinery the reduced model
        uses (classical MD has no electrons).  The plume seed radius is the
        r.m.s. extent of the ejecta cloud at the end of the run.
        """
        from ..impact import ImpactResult, Projectile
        from ..ionization import IonisationState, saha_solve

        mat = framework_material or get_material(self.config.material)
        md = self.config.md()
        m_atom = md.mass_amu * AMU

        m_p = self.diagnostics.get("n_projectile", 0) * m_atom
        proj = Projectile(mat, mass=max(m_p, m_atom),
                          velocity=self.config.velocity)

        if self.positions is not None and len(self.positions):
            rel = self.positions - self.positions.mean(axis=0)
            r_plume = float(np.sqrt((rel**2).sum(axis=1).mean()))
        else:
            r_plume = 3.0 * self.config.projectile_radius
        r_plume = max(r_plume, self.config.projectile_radius)

        V = (2.0 / 3.0) * np.pi * r_plume**3
        n_h = (self.n_ejecta / V) if V > 0 else 0.0

        if n_h > 0 and self.T_ejecta > 0:
            plasma = saha_solve(mat, self.T_ejecta, n_h)
        else:
            z = np.zeros(len(mat.E_ion) + 1)
            z[0] = 1.0
            plasma = IonisationState(300.0, n_h, 0.0, 0.0, z, 0.0)

        from ..constants import E_CHARGE
        Q = E_CHARGE * plasma.Zbar * self.n_ejecta

        return ImpactResult(
            projectile=proj, target=mat, state=None,
            P_ic=float("nan"), r_ic=self.config.projectile_radius,
            m_vapour_projectile=self.m_ejecta, m_vapour_target=0.0,
            m_vapour=self.m_ejecta, m_plasma=self.m_ejecta,
            m_melt_target=0.0, f_vap_projectile=float("nan"),
            E_res_projectile=float("nan"), E_res_core=float("nan"),
            E_vapour_specific=1.5 * K_B * self.T_ejecta / m_atom,
            u_atom=1.5 * K_B * self.T_ejecta,
            material_mix=mat, plasma=plasma, Q_free=Q,
            n_h0=n_h, rho_plume0=n_h * m_atom, r_plume0=r_plume,
            v_expansion=max(self.v_ejecta_mean, 1.0),
            thresholds={}, diagnostics={"source": "lammps",
                                        "md_result": self},
        )


# ---------------------------------------------------------------------------
# In-process execution
# ---------------------------------------------------------------------------

def run_impact(cfg: LammpsImpactConfig,
               potential_dir: str | None = None,
               keep_ejecta_arrays: bool = True,
               screen: bool = False) -> LammpsImpactResult:
    """Run the impact through the LAMMPS Python interface.

    Requires the ``lammps`` wheel (``pip install lammps mpich``).  Runtime is
    roughly proportional to ``estimated_atom_count() * steps``; the default
    configuration (~1e5 atoms, 1e5 steps) is minutes on one core.  For
    paper-scale runs (1e7+ atoms, 1e6 steps) generate the deck instead and
    submit it to a cluster with the standalone executable.
    """
    _preload_mpi()
    # Via import_lammps(), not `from lammps import ...`: conda-forge builds
    # raise ValueError at import time from LAMMPS's own version parsing.
    from . import import_lammps
    _lammps = import_lammps().lammps

    md = cfg.md()
    m_atom = md.mass_amu * AMU
    deck = generate_input_deck(cfg, potential_dir=potential_dir)

    args = ["-log", "none"]
    if not screen:
        args += ["-screen", "none"]
    lmp = _lammps(cmdargs=args)
    try:
        # Everything except the final production run, so the energy baseline
        # can be captured right after the projectile velocity is applied.
        head, tail = deck.rsplit("run", 1)
        n_run = int(tail.split()[0])
        lmp.commands_string(head)
        lmp.command("run 0 post no")          # initialise thermo state
        e0 = (lmp.get_thermo("ke") + lmp.get_thermo("pe"))
        lmp.command(f"run {n_run}")
        e1 = (lmp.get_thermo("ke") + lmp.get_thermo("pe"))

        n_atoms = lmp.get_natoms()
        x = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)
        v = np.array(lmp.gather_atoms("v", 1, 3)).reshape(-1, 3)
        typ = np.array(lmp.gather_atoms("type", 0, 1))
        T_final = lmp.get_thermo("temp")
    finally:
        lmp.close()

    # --- ejecta identification ------------------------------------------
    nx, ny, nz = cfg.resolved_cells()
    z_surf_A = nz * md.a0                       # Angstrom
    above = x[:, 2] > (z_surf_A + 2.0 * md.a0)
    upward = v[:, 2] > 0.0
    ej = above & upward
    n_ej = int(ej.sum())

    A_ps_to_m_s = 1e-10 / 1e-12                  # metal units: A/ps
    v_ej = v[ej] * A_ps_to_m_s
    x_ej = x[ej] * 1e-10

    if n_ej > 0:
        v_cm = v_ej.mean(axis=0)
        speed = float(np.linalg.norm(v_cm))
        spread = v_ej - v_cm
        # 3/2 kT = 1/2 m <dv^2>
        T_ej = float(m_atom * (spread**2).sum(axis=1).mean() / (3.0 * K_B))
        v_std = float(np.sqrt((spread**2).sum(axis=1).mean()))
    else:
        speed = v_std = T_ej = 0.0

    n_proj = int((typ == 2).sum())
    ke0 = 0.5 * n_proj * m_atom * cfg.velocity**2

    return LammpsImpactResult(
        config=cfg, n_atoms=n_atoms, n_ejecta=n_ej,
        m_ejecta=n_ej * m_atom,
        v_ejecta_mean=speed, v_ejecta_std=v_std, T_ejecta=T_ej,
        T_target_final=float(T_final),
        ke_initial=ke0,
        energy_drift=float((e1 - e0) / e0) if e0 else 0.0,
        z_surface=z_surf_A * 1e-10,
        positions=x_ej if keep_ejecta_arrays else None,
        velocities=v_ej if keep_ejecta_arrays else None,
        diagnostics={"n_projectile": n_proj,
                     "deck": deck,
                     "z_surface_A": z_surf_A},
    )


# ---------------------------------------------------------------------------
# Cross-check against the reduced model
# ---------------------------------------------------------------------------

def compare_with_reduced_model(res: LammpsImpactResult) -> dict:
    """MD outcome vs the reduced-order Stage 1 for the same impact.

    Expectations, not equalities: at nm scale, surface energy and the absence
    of a developed shock make MD ejecta systematically *less* thermalised
    than the continuum model assumes, and the reduced model's vaporisation
    thresholds are bulk quantities.  Agreement to a factor of a few in ejecta
    fraction and temperature is what the literature's nm-scale MD (Fraile
    2022; Anders et al.) supports.
    """
    from ..impact import Projectile, simulate_impact

    cfg = res.config
    mat = get_material(cfg.material)
    md = cfg.md()
    m_p = res.diagnostics.get("n_projectile", 0) * md.mass_amu * AMU
    reduced = simulate_impact(Projectile(mat, max(m_p, 1e-24), cfg.velocity),
                              mat, warn=False)
    return {
        "md_ejecta_fraction": res.ejecta_fraction_of_projectile,
        "reduced_vapour_fraction": reduced.vaporisation_efficiency,
        "md_T_ejecta_eV": res.T_ejecta * K_B / EV,
        "reduced_T_eV": reduced.plasma.T_eV,
        "md_ejecta_speed_km_s": res.v_ejecta_mean / 1e3,
        "reduced_expansion_speed_km_s": reduced.v_expansion / 1e3,
        "ke_per_atom_eV": cfg.ke_per_atom_eV,
        "reduced_result": reduced,
    }


__all__ = [
    "MD_SCALES", "config_for_scale", "scale_table",
    "ATOM_STEPS_PER_SEC_CORE", "MPI_EFFICIENCY", "BYTES_PER_ATOM",
    "MDMaterial", "MD_LIBRARY", "LammpsImpactConfig", "LammpsImpactResult",
    "generate_input_deck", "run_impact", "compare_with_reduced_model",
    "bundled_potential_dir",
]


# ---------------------------------------------------------------------------
# Module demo:  python -m hvi_emp.solvers.lammps_stage1
# ---------------------------------------------------------------------------

def _demo(argv=None) -> int:                     # pragma: no cover - CLI
    """Print a ready-to-run LAMMPS deck for the Fraile et al. configuration."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m hvi_emp.solvers.lammps_stage1",
        description="Generate (and optionally run) a LAMMPS hypervelocity "
                    "impact in the Fraile et al. (2022) configuration.")
    ap.add_argument("--material", default="W")
    ap.add_argument("--velocity", type=float, default=9e3, help="m/s")
    ap.add_argument("--radius", type=float, default=1.3e-9,
                    help="projectile radius [m]")
    ap.add_argument("--out", default=None, help="write the deck to this file")
    ap.add_argument("--run", action="store_true",
                    help="run it in-process (needs the lammps wheel)")
    a = ap.parse_args(argv)

    cfg = LammpsImpactConfig(material=a.material, velocity=a.velocity,
                             projectile_radius=a.radius)
    print(f"# {cfg.estimated_atom_count():,} atoms, "
          f"{cfg.ke_per_atom_eV:.2f} eV/atom, cells {cfg.resolved_cells()}")
    deck = generate_input_deck(cfg)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(deck)
        print(f"# wrote {a.out}  ->  mpirun -np <N> lmp -in {a.out}")
    else:
        print(deck)
    if a.run:
        print(run_impact(cfg).summary())
    return 0


if __name__ == "__main__":                       # pragma: no cover
    import sys as _s
    _s.exit(_demo())
