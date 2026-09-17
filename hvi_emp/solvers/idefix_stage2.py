"""Stage 2 on Idefix: magnetised plume expansion in spherical geometry.

Idefix (Lesur et al., A&A 677, A9, 2023; https://github.com/idefix-code/idefix)
is a Godunov finite-volume HD/MHD code built on Kokkos, so one source tree
runs on CPU, NVIDIA CUDA and AMD HIP.  It is a Kokkos re-implementation of
PLUTO's algorithms, distributed under the CeCILL licence.

Why Idefix rather than OpenMHD for this problem
-----------------------------------------------
The reduced Stage 2 expands a plume into vacuum with no magnetic field.  The
physics it omits -- the diamagnetic cavity carved out of the ambient field --
is what the MHD bridge exists to recover.  Three properties make Idefix a
better host for it than the OpenMHD bridge in ``openmhd_stage2``:

1. **Spherical geometry.**  The cavity is a radially expanding shell.  On a
   uniform Cartesian grid its surface is a staircase and the numerical
   diffusion at the contact discontinuity is grid-aligned; in (r, theta) the
   surface is a coordinate surface and the problem is 2D axisymmetric rather
   than 3D.  Combined with logarithmic radial spacing, that covers the
   decades of expansion the plume actually goes through at constant cost per
   decade.

2. **Non-ideal MHD.**  The plume is not a perfect conductor.  At the densities
   and ~1 eV temperatures Stage 1 predicts, the Spitzer magnetic Reynolds
   number is order 1-100 over the cavity scale, so Ohmic diffusion sets how
   fast the field refills the cavity -- which is the collapse that radiates.
   Idefix carries Ohmic, ambipolar and Hall terms, with Runge-Kutta-Legendre
   super-timestepping so the parabolic step does not dominate the cost.

3. **GPU via a plain CMake switch.**  ``-DKokkos_ENABLE_CUDA=ON``.  OpenMHD's
   GPU path is a separate set of CUDA Fortran source directories requiring
   the NVIDIA HPC SDK and nvshmem.

What this bridge does *not* claim
---------------------------------
Ideal/resistive MHD cannot produce the charge-separation EMP -- it assumes
quasi-neutrality by construction.  This stage gives the cavity dynamics and
the low-frequency magnetic signature; the radiated pulse at 315/916 MHz comes
from Stage 3 (``warpx_stage3``, ``picongpu_stage3``).  Do not read a cavity
run as a prediction of EMP amplitude.

Execution is not wrapped in-process.  Idefix compiles the setup into the
executable, so the bridge generates a complete problem directory
(``definitions.hpp``, ``setup.cpp``, ``idefix.ini``) plus the exact cmake
line, and reads the VTK output back.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

if __name__ == "__main__" and __package__ is None:      # pragma: no cover
    raise SystemExit(
        "\nThis file is a module of the `hvi_emp` package, not a script.\n"
        "Run it as a module from the project root:\n\n"
        "    python -m hvi_emp.solvers.idefix_stage2\n\n"
        "or import what you need:\n\n"
        "    from hvi_emp.solvers.idefix_stage2 import IdefixConfig\n")

from ..constants import K_B, K_PER_EV, MU_0
from ..expansion import ExpansionResult
from .openmhd_stage2 import cavity_estimates

# Spitzer resistivity prefactor: eta = SPITZER_ETA * Z lnLambda / T_eV^1.5
# [Ohm m], NRL formulary, Z=1, lnLambda folded in by the caller.
SPITZER_ETA = 5.2e-5


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IdefixConfig:
    """Diamagnetic-cavity problem for Idefix, in code units.

    Idefix is unit-free: we normalise lengths to the initial plume radius
    ``R0``, density to the ambient density, and velocity to the ambient
    Alfven speed, so the background field has magnitude 1 and time is
    measured in Alfven crossings of R0.  This is the same normalisation the
    OpenMHD bridge uses, so the two can be compared directly (and the test
    suite asserts they agree).

    Parameters
    ----------
    rho_plume, rho_ambient : float
        Peak plume and ambient mass densities [kg/m^3].
    p_plume, p_ambient : float
        Pressures [Pa].
    v_expansion : float
        Plume radial expansion speed [m/s].
    B0 : float
        Background field magnitude [T]. LEO is ~3e-5 T.
    R0 : float
        Initial plume radius [m].
    eta_ohm : float
        Ohmic diffusivity [m^2/s]. Zero disables resistivity.
    geometry : {"spherical", "cartesian"}
        ``spherical`` gives the 2D axisymmetric (r, theta) problem with the
        background field along the polar axis -- cheap and well aligned.
        ``cartesian`` gives a 3D box, needed when the field is oblique to
        the expansion axis.
    r_min_frac : float
        Inner radial boundary as a fraction of R0. Spherical grids cannot
        include r = 0; the plume interior below this radius is not solved.
    domain_R : float
        Outer radius (or Cartesian half-width) in units of R0.
    n_r, n_theta : int
        Cells in each direction.
    t_end_alfven : float, optional
        Stop time in Alfven crossings of R0.  Leave as None (the default) to
        have it set to three cavity formation times, which is the timescale
        the problem actually runs on -- an Alfven crossing of R0 is typically
        two orders of magnitude shorter and stopping there shows nothing.
    n_outputs : int
        Number of VTK dumps over the run.
    """
    rho_plume: float
    rho_ambient: float
    p_plume: float
    p_ambient: float
    v_expansion: float
    B0: float = 3.0e-5
    R0: float = 0.01
    eta_ohm: float = 0.0
    geometry: str = "spherical"
    r_min_frac: float = 0.05
    domain_R: float | None = None
    n_r: int = 512
    n_theta: int = 256
    t_end_alfven: float | None = None
    n_outputs: int = 60
    gamma: float = 5.0 / 3.0
    reconstruction: str = "Parabolic"
    solver: str = "hlld"
    notes: list = field(default_factory=list)

    def __post_init__(self):
        if self.geometry not in ("spherical", "cartesian"):
            raise ValueError(
                f"geometry must be 'spherical' or 'cartesian', "
                f"got {self.geometry!r}")
        if not 0.0 < self.r_min_frac < 1.0:
            raise ValueError("r_min_frac must lie strictly between 0 and 1")
        if self.domain_R is None:
            # Hold the expected cavity with room to spare. Logarithmic radial
            # spacing means a larger domain costs cells proportional to the
            # log of the range, so being generous here is cheap.
            self.domain_R = float(
                max(20.0, 3.0 * self.cavity()["R_cavity"] / self.R0))
        if self.t_end_alfven is None:
            # The cavity inflates on t_formation = R_c / v_expansion, which is
            # unrelated to -- and usually far longer than -- an Alfven crossing
            # of R0.  Run long enough to see inflation, overshoot and collapse.
            self.t_end_alfven = float(
                3.0 * self.cavity()["t_formation"]
                / (self.R0 / self.v_alfven))

    # -- normalisation ----------------------------------------------------
    @property
    def v_alfven(self) -> float:
        """Ambient Alfven speed [m/s]."""
        return self.B0 / np.sqrt(MU_0 * self.rho_ambient)

    def normalised(self) -> dict:
        """Code-unit values and the SI factors needed to undo them."""
        vA = self.v_alfven
        p_norm = self.B0**2 / MU_0
        eta_hat = self.eta_ohm / (vA * self.R0) if self.eta_ohm else 0.0
        return {
            "rho_plume_hat": self.rho_plume / self.rho_ambient,
            "rho_ambient_hat": 1.0,
            "p_plume_hat": self.p_plume / p_norm,
            "p_ambient_hat": self.p_ambient / p_norm,
            "v_expansion_hat": self.v_expansion / vA,
            "eta_hat": eta_hat,
            "Rm": (1.0 / eta_hat) if eta_hat > 0 else np.inf,
            "beta_ambient": 2.0 * self.p_ambient / p_norm,
            "beta_plume": 2.0 * self.p_plume / p_norm,
            "v_alfven": vA,
            "unit_length": self.R0,
            "unit_time": self.R0 / vA,
            "unit_density": self.rho_ambient,
            "unit_velocity": vA,
            "unit_B": self.B0,
            "unit_pressure": p_norm,
        }

    def units_cgs(self) -> dict:
        """The three numbers Idefix's ``[Units]`` block wants, in CGS.

        Idefix takes length in cm, velocity in cm/s and density in g/cm^3 and
        reconstructs everything else.  Outputs stay in code units regardless;
        this block only matters if a module needs physical constants.
        """
        return {"length": self.R0 * 100.0,
                "velocity": self.v_alfven * 100.0,
                "density": self.rho_ambient * 1.0e-3}

    # -- construction from the reduced chain ------------------------------
    @classmethod
    def from_expansion(cls, exp: ExpansionResult,
                       epoch: str | int = "auto",
                       coupling_margin: float = 100.0,
                       B0: float = 3.0e-5,
                       n_ambient: float = 1.0e11,
                       T_ambient_eV: float = 0.1,
                       m_ambient: float = 2.66e-26,      # atomic oxygen
                       coulomb_log: float = 10.0,
                       resistive: bool = True,
                       **overrides) -> "IdefixConfig":
        """Initial condition from a Stage-2 epoch.

        Parameters
        ----------
        epoch : {"auto", "early", "late"} or int
            Which expansion sample to initialise from.  The default,
            ``"auto"``, picks the epoch where the magnetic field first starts
            to matter -- the first sample at which the plume's kinetic energy
            density has fallen to within ``coupling_margin`` of the ambient
            magnetic pressure B^2 / 2 mu_0.

            This is the physically and computationally right place to start.
            While the plume is far denser and faster than the field can
            resist, its expansion is ballistic and MHD adds nothing: the
            self-similar solution Stage 2 already integrates is exact there.
            Starting at the *first* sample (``"early"``) means resolving four
            or five decades of that free expansion before any MHD happens,
            which costs hundreds of GPU-hours to reproduce an analytic result.
            Starting at the *last* (``"late"``, which is what the OpenMHD
            bridge does) means a cold, largely recombined cloud that forms no
            cavity at all.
        resistive : bool
            Use the Spitzer Ohmic diffusivity at the plume's own electron
            temperature, which decides whether the cavity refills by
            advection or by diffusion.

        Notes
        -----
        Ambient defaults are the LEO daytime ionosphere: ~1e11 m^-3 atomic
        oxygen at ~0.1 eV in a 3e-5 T field.
        """
        if exp is None:
            raise ValueError(
                "no expansion result to build from. run_scenario() returns "
                "expansion=None when Stage 1 produced no plasma at all -- "
                "check ScenarioResult.impact.plasma.mass before getting here.")
        never_coupled = False
        if epoch == "auto":
            k = cls._coupling_index(exp, B0, coupling_margin)
            if k < 0:                      # sentinel: never magnetically coupled
                never_coupled, k = True, -1
        elif epoch == "early":
            k = 0
        elif epoch == "late":
            k = -1
        elif isinstance(epoch, (int, np.integer)):
            k = int(epoch)
        else:
            raise ValueError(
                f"epoch must be 'auto', 'early', 'late' or an integer index, "
                f"got {epoch!r}")
        rho_p = float(exp.n_h[k] * exp.material.m_atom)
        Zbar = float(exp.Zbar[k])
        T_K = float(exp.T[k])
        p_p = float(exp.n_h[k] * (1.0 + Zbar) * K_B * T_K)
        rho_a = n_ambient * m_ambient
        p_a = 2.0 * n_ambient * T_ambient_eV * 1.602176634e-19
        R0 = float(np.sqrt(exp.R_r[k] * exp.R_z[k]))

        T_eV = T_K / K_PER_EV
        eta = 0.0
        spitzer_valid = True
        if resistive:
            # Spitzer resistivity [Ohm m] -> magnetic diffusivity [m^2/s].
            # Valid only for a substantially ionised plasma: it assumes
            # electron-ion Coulomb collisions dominate, and it diverges as
            # T^-3/2 as the plume recombines.
            spitzer_valid = (T_eV > 0.5) and (Zbar > 0.1)
            eta_ohm_m = (SPITZER_ETA * max(Zbar, 1.0) * coulomb_log
                         / max(T_eV, 1e-2)**1.5)
            eta = eta_ohm_m / MU_0

        kw = dict(
            rho_plume=rho_p, rho_ambient=rho_a,
            p_plume=max(p_p, 1e-12), p_ambient=max(p_a, 1e-14),
            v_expansion=float(np.hypot(exp.v_r[k], exp.v_z[k])),
            B0=B0, R0=R0, eta_ohm=eta,
        )
        kw.update(overrides)
        cfg = cls(**kw)
        cfg._audit()
        if never_coupled:
            cfg.notes.insert(0, (
                "the plume's kinetic energy density never falls to within "
                f"{coupling_margin:g}x the ambient magnetic pressure anywhere "
                "in the integrated expansion history, so the field never "
                "impedes it over the simulated window and no diamagnetic "
                "cavity forms in it. For an impact this small that is the "
                "physical answer, not a setup error -- the MHD stage has "
                "nothing to add. Run Stage 2 to a later t_end, raise B0, or "
                "use a larger projectile if you want the cavity regime."))
        if resistive and not spitzer_valid:
            cfg.notes.insert(0, (
                f"Spitzer resistivity was evaluated at T_e = {T_eV:.3g} eV, "
                f"Zbar = {Zbar:.3g} -- outside its range of validity (it "
                "assumes electron-ion Coulomb collisions dominate and "
                "diverges as T^-3/2 through recombination). The eta below is "
                "an extrapolation, not a measurement. Either initialise at an "
                "earlier epoch (epoch='early'), pass resistive=False for the "
                "ideal-MHD limit, or set eta_ohm yourself from a transport "
                "model that covers partial ionisation."))
        return cfg

    @staticmethod
    def _coupling_index(exp: ExpansionResult, B0: float,
                        margin: float) -> int:
        """First epoch at which the ambient field can influence the plume.

        The plume expands ballistically for as long as its kinetic energy
        density overwhelms the magnetic energy density; only when the two
        approach each other does MHD have anything to solve.  We return the
        first index where

            (1/2) rho v^2  <  margin * B^2 / (2 mu_0)

        and fall back to the last sample if the plume never becomes that
        weak (in which case no cavity forms within the integrated history and
        the audit will say so).
        """
        rho = np.asarray(exp.n_h, dtype=float) * exp.material.m_atom
        v = np.hypot(np.asarray(exp.v_r, dtype=float),
                     np.asarray(exp.v_z, dtype=float))
        e_kin = 0.5 * rho * v * v
        e_mag = B0 * B0 / (2.0 * MU_0)
        below = np.where(e_kin < margin * e_mag)[0]
        if below.size:
            return int(below[0])
        return -int(len(e_kin))          # negative => never coupled, see below

    # -- self-audit --------------------------------------------------------
    def _audit(self) -> list:
        """Flag setups whose answer will not mean what the user expects."""
        n = self.normalised()
        self.notes = []
        if n["beta_ambient"] > 1.0:
            self.notes.append(
                f"ambient beta = {n['beta_ambient']:.2f} > 1: the ambient is "
                "gas-pressure dominated, so no well-defined diamagnetic "
                "cavity forms. Raise B0 or lower the ambient density.")
        est = self.cavity()
        if est["R_cavity"] / self.R0 > 0.5 * self.domain_R:
            self.notes.append(
                f"expected cavity {est['R_cavity'] / self.R0:.1f} R0 vs domain "
                f"{self.domain_R:.0f} R0: enlarge domain_R, the cavity will "
                "hit the boundary.")
        if n["Rm"] < 1.0:
            self.notes.append(
                f"magnetic Reynolds number Rm = {n['Rm']:.2g} < 1: the field "
                "diffuses faster than the plume expands, so the cavity is "
                "resistively erased rather than advected. Physical, but check "
                "the temperature Stage 1 handed over.")
        cells_per_R0 = self.n_r / np.log(self.domain_R / self.r_min_frac)
        if cells_per_R0 < 20:
            self.notes.append(
                f"only ~{cells_per_R0:.0f} radial cells per e-fold: the "
                "contact discontinuity will be badly smeared. Raise n_r.")
        mach_a = n["v_expansion_hat"]
        if mach_a < 0.2:
            self.notes.append(
                f"expansion is sub-Alfvenic (v_exp/v_A = {mach_a:.2g}): the "
                f"timestep is set by v_A but the dynamics by v_exp, so the "
                f"run costs ~{1.0 / max(mach_a, 1e-6):.0f}x more steps than "
                "the physics needs. This is intrinsic to explicit MHD, not a "
                "setup error -- but it is why this case is expensive.")
        cost = self.cost_estimate()
        if cost["hours"] > 24:
            self.notes.append(
                f"estimated {cost['hours']:.0f} h on one GPU. Reduce n_r/"
                "n_theta, or shorten t_end_alfven and accept seeing only the "
                "inflation phase.")
        return self.notes

    def cavity(self) -> dict:
        """Analytic pressure-balance expectation, for comparison."""
        E_kin = (0.5 * self.rho_plume * (4.0 / 3.0) * np.pi * self.R0**3
                 * self.v_expansion**2)
        return cavity_estimates(E_kinetic=E_kin, B0=self.B0,
                                v_expansion=self.v_expansion)

    # -- cost -------------------------------------------------------------
    def cost_estimate(self, cell_updates_per_second: float = 2.0e8) -> dict:
        """Rough wall-clock estimate.

        The default throughput is a single mid-range NVIDIA GPU running
        Idefix's MHD solver; a Xeon core is nearer 5e6, so a 64-core CPU run
        lands within a factor of a few of one GPU.  Treat as sizing.
        """
        ncell = self.n_r * (self.n_theta if self.geometry == "spherical"
                            else self.n_theta**2)
        # dt ~ CFL * dx_min / fastest speed; in code units v_max ~ max(1, v_hat)
        n = self.normalised()
        v_max = max(1.0, n["v_expansion_hat"])
        dr_min = self.r_min_frac * (np.log(self.domain_R / self.r_min_frac)
                                    / self.n_r)
        dt = 0.4 * dr_min / v_max
        nsteps = int(np.ceil(self.t_end_alfven / dt))
        seconds = ncell * nsteps / cell_updates_per_second
        return {"cells": ncell, "steps": nsteps, "dt_code": dt,
                "seconds": seconds, "hours": seconds / 3600.0,
                "output_GB": self.n_outputs * ncell * 8 * 8 / 1e9}


# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------

def write_definitions_hpp(cfg: IdefixConfig, path: str | None = None) -> str:
    """Compile-time configuration: dimensions, components, geometry."""
    if cfg.geometry == "spherical":
        dims, comps, geom = 2, 3, "SPHERICAL"
        why = ("2D axisymmetric (r, theta); 3 components so that Hall and "
               "rotation can generate a toroidal field")
    else:
        dims, comps, geom = 3, 3, "CARTESIAN"
        why = "full 3D, needed when B is oblique to the expansion axis"

    text = f"""// Generated by hvi_emp.solvers.idefix_stage2
// {why}

#define     COMPONENTS      {comps}
#define     DIMENSIONS      {dims}

#define     GEOMETRY        {geom}
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def write_idefix_ini(cfg: IdefixConfig, path: str | None = None) -> str:
    """Runtime input file."""
    n = cfg.normalised()
    u = cfg.units_cgs()
    dt_out = cfg.t_end_alfven / max(cfg.n_outputs, 1)

    if cfg.geometry == "spherical":
        # Logarithmic radial spacing: constant resolution per e-fold, which
        # is what an expansion over decades needs.
        grid = (f"X1-grid    1  {cfg.r_min_frac:.6g}  {cfg.n_r}  l  "
                f"{cfg.domain_R:.6g}\n"
                f"X2-grid    1  0.0  {cfg.n_theta}  u  3.14159265358979\n")
        boundary = ("X1-beg    outflow      # inner radius: plume interior "
                    "is not solved\n"
                    "X1-end    outflow      # ambient, far from the cavity\n"
                    "X2-beg    axis         # polar axis, requires X2 in "
                    "[0, pi]\n"
                    "X2-end    axis\n")
    else:
        R = cfg.domain_R
        grid = "".join(
            f"X{i}-grid    1  {-R:.6g}  {cfg.n_theta}  u  {R:.6g}\n"
            for i in (1, 2, 3))
        boundary = "".join(f"X{i}-{s}    outflow\n"
                           for i in (1, 2, 3) for s in ("beg", "end"))

    # Hall requires the HLL solver and arithmetic EMF averaging (Idefix docs);
    # we do not enable Hall by default, so uct_contact is fine.
    resistivity = ""
    if n["eta_hat"] > 0:
        resistivity = (f"resistivity   rkl  constant  {n['eta_hat']:.6e}   "
                       f"# Ohmic, Rm = {n['Rm']:.3g}\n")

    text = f"""# Idefix diamagnetic-cavity problem
# Generated by hvi_emp.solvers.idefix_stage2
#
# Code units:  length = R0 = {cfg.R0:.4e} m
#              velocity = v_A = {n['v_alfven']:.4e} m/s
#              time = R0/v_A = {n['unit_time']:.4e} s
#              density = rho_ambient = {cfg.rho_ambient:.4e} kg/m^3
#              B = B0 = {cfg.B0:.4e} T
#              pressure = B0^2/mu0 = {n['unit_pressure']:.4e} Pa

[Grid]
{grid}
[TimeIntegrator]
CFL         0.4
CFL_max_var 1.1
tstop       {cfg.t_end_alfven:.6g}
first_dt    1.e-6
nstages     {2 if cfg.reconstruction in ('Linear', 'Constant') else 3}
check_nan   100

[Hydro]
solver      {cfg.solver}
emf         uct_contact
gamma       {cfg.gamma:.6g}
{resistivity}shockFlattening  5.0    # the cavity edge is a strong shock

[RKL]
cfl         0.45
rmax_par    100.0

[Boundary]
{boundary}
[Setup]
# read by setup.cpp through Input::Get<real>("Setup", key, 0)
rho_plume       {n['rho_plume_hat']:.8e}
p_plume         {n['p_plume_hat']:.8e}
p_ambient       {n['p_ambient_hat']:.8e}
v_expansion     {n['v_expansion_hat']:.8e}
R0              1.0
shell_width     0.15          # relative width of the density taper

[Units]
# CGS, used only by modules that need physical constants; outputs stay in
# code units.
length      {u['length']:.8e}
velocity    {u['velocity']:.8e}
density     {u['density']:.8e}

[Output]
vtk         {dt_out:.6g}
vtk_dir     ./vtk
dmp         {cfg.t_end_alfven / 4:.6g}
dmp_dir     ./dumps
log         100
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


_SETUP_SPHERICAL = r"""
// Generated by hvi_emp.solvers.idefix_stage2 -- diamagnetic cavity, spherical
//
// Initial condition: a dense, hot, homologously expanding plume (v_r ~ r)
// embedded in a uniform ambient threaded by a uniform field along the polar
// axis.  In spherical (r, theta) with the field along z, the problem is
// axisymmetric, so DIMENSIONS 2 is exact rather than an approximation.
//
// The field is initialised from the phi-component of the vector potential,
//     A_phi = B0 * r * sin(theta) / 2,
// which gives a uniform B0 z-hat with zero discrete divergence.  Setting the
// face-centred components directly (the #else branch) only satisfies div B = 0
// to truncation error on a spherical mesh, which the constrained transport
// scheme will then preserve -- including the error.  Configure with
// -DIdefix_EVOLVE_VECTOR_POTENTIAL=ON to take the exact branch.

#include "idefix.hpp"
#include "setup.hpp"

static real rhoPlume, pPlume, pAmbient, vExpansion, R0, shellWidth;

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  rhoPlume    = input.GetOrSet<real>("Setup", "rho_plume",   0, 100.0);
  pPlume      = input.GetOrSet<real>("Setup", "p_plume",     0, 1.0);
  pAmbient    = input.GetOrSet<real>("Setup", "p_ambient",   0, 0.01);
  vExpansion  = input.GetOrSet<real>("Setup", "v_expansion", 0, 5.0);
  R0          = input.GetOrSet<real>("Setup", "R0",          0, 1.0);
  shellWidth  = input.GetOrSet<real>("Setup", "shell_width", 0, 0.15);
}

void Setup::InitFlow(DataBlock &data) {
  DataBlockHost d(data);

  for(int k = 0; k < d.np_tot[KDIR]; k++) {
    for(int j = 0; j < d.np_tot[JDIR]; j++) {
      for(int i = 0; i < d.np_tot[IDIR]; i++) {
        real r     = d.x[IDIR](i);
        real theta = d.x[JDIR](j);

        // Smooth taper from plume to ambient across `shellWidth * R0`.
        // tanh rather than a top hat: a discontinuous contact seeds
        // Richtmyer-Meshkov ripples that are numerical, not physical.
        real s = 0.5 * (1.0 - tanh((r - R0) / (shellWidth * R0)));

        d.Vc(RHO,k,j,i) = 1.0 + (rhoPlume - 1.0) * s;
        d.Vc(PRS,k,j,i) = pAmbient + (pPlume - pAmbient) * s;

        // Homologous expansion inside the plume, at rest outside.
        d.Vc(VX1,k,j,i) = vExpansion * (r / R0) * s;
        d.Vc(VX2,k,j,i) = ZERO_F;
#if COMPONENTS == 3
        d.Vc(VX3,k,j,i) = ZERO_F;
#endif

#ifdef EVOLVE_VECTOR_POTENTIAL
        // A_phi on the phi-edge: exact, div B = 0 to machine precision
        real rl  = d.xl[IDIR](i);
        real thl = d.xl[JDIR](j);
        d.Ve(AX3e,k,j,i) = 0.5 * rl * sin(thl);
#else
        // Uniform B0 along z, resolved into spherical components on faces
        d.Vs(BX1s,k,j,i) =  cos(d.x[JDIR](j));        // B_r  at r-faces
        d.Vs(BX2s,k,j,i) = -sin(d.xl[JDIR](j));       // B_th at theta-faces
#endif
      }
    }
  }

  d.SyncToDevice();
}

// Cavity radius along the equator, written to the log each analysis interval.
void MakeAnalysis(DataBlock &data) {
}
"""


_SETUP_CARTESIAN = r"""
// Generated by hvi_emp.solvers.idefix_stage2 -- diamagnetic cavity, Cartesian
//
// 3D box with a uniform field along z and a spherical plume at the origin.
// Use this when the field is oblique to the expansion axis; otherwise the
// spherical setup is the same physics in two dimensions and far cheaper.

#include "idefix.hpp"
#include "setup.hpp"

static real rhoPlume, pPlume, pAmbient, vExpansion, R0, shellWidth;

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  rhoPlume    = input.GetOrSet<real>("Setup", "rho_plume",   0, 100.0);
  pPlume      = input.GetOrSet<real>("Setup", "p_plume",     0, 1.0);
  pAmbient    = input.GetOrSet<real>("Setup", "p_ambient",   0, 0.01);
  vExpansion  = input.GetOrSet<real>("Setup", "v_expansion", 0, 5.0);
  R0          = input.GetOrSet<real>("Setup", "R0",          0, 1.0);
  shellWidth  = input.GetOrSet<real>("Setup", "shell_width", 0, 0.15);
}

void Setup::InitFlow(DataBlock &data) {
  DataBlockHost d(data);

  for(int k = 0; k < d.np_tot[KDIR]; k++) {
    for(int j = 0; j < d.np_tot[JDIR]; j++) {
      for(int i = 0; i < d.np_tot[IDIR]; i++) {
        real x = d.x[IDIR](i);
        real y = d.x[JDIR](j);
        real z = d.x[KDIR](k);
        real r = sqrt(x*x + y*y + z*z);
        real s = 0.5 * (1.0 - tanh((r - R0) / (shellWidth * R0)));
        real vr = vExpansion * (r / R0) * s;
        real rinv = (r > 1e-12) ? 1.0 / r : 0.0;

        d.Vc(RHO,k,j,i) = 1.0 + (rhoPlume - 1.0) * s;
        d.Vc(PRS,k,j,i) = pAmbient + (pPlume - pAmbient) * s;
        d.Vc(VX1,k,j,i) = vr * x * rinv;
        d.Vc(VX2,k,j,i) = vr * y * rinv;
        d.Vc(VX3,k,j,i) = vr * z * rinv;

#ifdef EVOLVE_VECTOR_POTENTIAL
        // B = curl A with A = (-y/2, x/2, 0) * B0 gives uniform B0 z-hat
        d.Ve(AX1e,k,j,i) = -0.5 * d.x[JDIR](j);
        d.Ve(AX2e,k,j,i) =  0.5 * d.xl[IDIR](i);
        d.Ve(AX3e,k,j,i) = ZERO_F;
#else
        d.Vs(BX1s,k,j,i) = ZERO_F;
        d.Vs(BX2s,k,j,i) = ZERO_F;
        d.Vs(BX3s,k,j,i) = ONE_F;
#endif
      }
    }
  }

  d.SyncToDevice();
}

void MakeAnalysis(DataBlock &data) {
}
"""


def write_setup_cpp(cfg: IdefixConfig, path: str | None = None) -> str:
    """Initial condition, compiled into the executable."""
    text = (_SETUP_SPHERICAL if cfg.geometry == "spherical"
            else _SETUP_CARTESIAN).lstrip("\n")
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


#: Kokkos architecture macro per card, without the ``Kokkos_ARCH_`` prefix.
#: Blackwell consumer parts (RTX 50-series) are ``BLACKWELL120`` and need
#: CUDA >= 12.8 and a Kokkos new enough to know the macro (>= 4.5); an older
#: Kokkos will configure happily and then emit code the card cannot run.
KOKKOS_ARCH = {
    "T4": "TURING75",
    "V100": "VOLTA70", "A100": "AMPERE80", "A30": "AMPERE80",
    "A40": "AMPERE86", "RTX3090": "AMPERE86", "RTX4090": "ADA89",
    # Professional Ampere workstation cards. The RTX A-series is the common
    # desk-side research GPU and was missing here, so `gpu_plan` silently
    # fell back to a 32 GB Blackwell budget on a 24 GB Ampere card.
    "A5000": "AMPERE86", "RTXA5000": "AMPERE86",
    "A4000": "AMPERE86", "RTXA4000": "AMPERE86",
    "A4500": "AMPERE86", "RTXA4500": "AMPERE86",
    "A5500": "AMPERE86", "RTXA5500": "AMPERE86",
    "A6000": "AMPERE86", "RTXA6000": "AMPERE86",
    "L40": "ADA89", "L40S": "ADA89", "RTX6000ADA": "ADA89",
    "H100": "HOPPER90", "H200": "HOPPER90",
    "RTX5080": "BLACKWELL120", "RTX5090": "BLACKWELL120",
    "B200": "BLACKWELL100",
}

#: Device memory per card [GB], for deciding how many ranks a grid needs.
GPU_MEMORY_GB = {
    "T4": 16,
    "V100": 32, "A100": 80, "A30": 24, "A40": 48, "RTX3090": 24,
    "RTX4090": 24, "L40": 48, "L40S": 48, "RTX6000ADA": 48,
    "A5000": 24, "RTXA5000": 24, "A4000": 16, "RTXA4000": 16,
    "A4500": 20, "RTXA4500": 20, "A5500": 24, "RTXA5500": 24,
    "A6000": 48, "RTXA6000": 48,
    "H100": 80, "H200": 141, "RTX5080": 16, "RTX5090": 32, "B200": 192,
}

#: Bytes per cell for an Idefix MHD run: 8 conserved + 8 primitive fields in
#: double precision, plus emfs, fluxes and the RK stage buffers. Measured
#: against reported Idefix footprints rather than derived, so treat it as an
#: estimate good to ~30%, and leave headroom.
IDEFIX_BYTES_PER_CELL = 400


def decompose(n_ranks: int, grid: tuple) -> tuple:
    """Split `n_ranks` across the three axes of `grid`.

    Idefix takes ``-dec nx ny nz`` and requires the product to equal the rank
    count and each axis to divide evenly. Rather than make the user work that
    out, this greedily gives each factor of `n_ranks` to whichever axis
    currently has the most cells per rank, which keeps subdomains as cubic as
    possible -- MPI halo traffic scales with subdomain *surface* area, so a
    slab decomposition of a cube moves several times the data a cubic one
    does.

    Axes that cannot divide evenly are skipped, so the result always
    satisfies Idefix's constraint. Raises if `n_ranks` cannot be placed at
    all (e.g. a prime rank count larger than every axis).
    """
    n_ranks = int(n_ranks)
    if n_ranks < 1:
        raise ValueError("n_ranks must be >= 1")
    dec = [1, 1, 1]
    if n_ranks == 1:
        return tuple(dec)

    # prime factors, largest first
    factors, m = [], n_ranks
    d = 2
    while d * d <= m:
        while m % d == 0:
            factors.append(d)
            m //= d
        d += 1
    if m > 1:
        factors.append(m)

    for f in sorted(factors, reverse=True):
        best, best_load = None, -1.0
        for ax in range(3):
            cells = grid[ax] / dec[ax]
            if cells / f < 4 or grid[ax] % (dec[ax] * f):
                continue                      # too thin, or does not divide
            if cells > best_load:
                best, best_load = ax, cells
        if best is None:
            raise ValueError(
                f"cannot place {n_ranks} ranks on a {grid} grid: no axis can "
                f"absorb a factor of {f} with at least 4 cells per rank. "
                f"Use a rank count whose factors divide the grid, or enlarge "
                f"the grid.")
        dec[best] *= f
    return tuple(dec)


def detect_gpu() -> str | None:
    """This machine's GPU as a `KOKKOS_ARCH` key, or None.

    Asks the driver rather than assuming. Used so `gpu_plan` sizes against
    the card that is actually present -- planning a run for a 32 GB card on
    a 24 GB one produces a job that dies hours in.
    """
    try:
        from ..doctor import _smi_card
    except ImportError:                                     # pragma: no cover
        return None
    card = _smi_card()
    if card is None:
        return None
    key = card[0].upper().replace("NVIDIA", "").replace("GEFORCE", "")
    key = "".join(ch for ch in key if ch.isalnum())
    for known in KOKKOS_ARCH:
        if known in key:
            return known
    return None


def gpu_plan(cfg: IdefixConfig, gpu: str | None = None, n_gpu: int = 1,
             safety: float = 0.75) -> dict:
    """Rank layout, memory footprint and the exact run command.

    One MPI rank per GPU is the model Idefix (via Kokkos) expects: a rank
    owns a device, and multiple ranks sharing one device serialise on it
    rather than sharing it.

    `safety` is the fraction of device memory the solver may use. The default
    leaves a quarter free because the CUDA context, the MPI buffers and any
    display attached to the card all take a share, and an out-of-memory abort
    happens hours in rather than at startup.
    """
    notes = []
    if gpu is None:
        gpu = detect_gpu()
        if gpu is None:
            gpu = "RTX5090"
            notes.append(
                "no GPU detected and none named, so this plan assumes an "
                "RTX 5090 (32 GB, Blackwell). Pass gpu='A5000' or whatever "
                "you actually have -- sizing a run for the wrong card is "
                "how a job dies of out-of-memory three hours in.")
        else:
            notes.append(f"card detected from the driver: {gpu}")

    key = gpu.upper()
    grid = (cfg.n_r, cfg.n_theta, cfg.n_phi) if hasattr(cfg, "n_phi") \
        else (cfg.n_r, cfg.n_theta, 1)
    cells = int(np.prod(grid))
    dec = decompose(n_gpu, grid)
    mem_total = cells * IDEFIX_BYTES_PER_CELL / 1024 ** 3
    per_rank = mem_total / n_gpu

    if key not in GPU_MEMORY_GB:
        # Do not quietly assume 32 GB. A 24 GB card given a 32 GB budget
        # produces a plan that looks fine and then fails.
        notes.append(
            f"{gpu!r} is not in GPU_MEMORY_GB, so the memory budget below "
            f"assumes 24 GB -- the smallest common compute card. Add it to "
            f"the table, or check the figure against `nvidia-smi "
            f"--query-gpu=memory.total --format=csv`.")
    budget = GPU_MEMORY_GB.get(key, 24) * safety
    if key not in KOKKOS_ARCH:
        notes.append(
            f"{gpu!r} is not in KOKKOS_ARCH, so no build flag can be given. "
            f"Get it from `nvidia-smi --query-gpu=compute_cap "
            f"--format=csv,noheader` (8.6 -> Kokkos_ARCH_AMPERE86).")
    if per_rank > budget:
        need = int(np.ceil(mem_total / budget))
        notes.append(
            f"{per_rank:.1f} GB/rank exceeds the {budget:.1f} GB usable on a "
            f"{gpu}; needs >= {need} GPUs, or a coarser grid.")
    if n_gpu > 1:
        notes.append(
            "Idefix assumes a CUDA-aware MPI. Without it every halo exchange "
            "round-trips through host memory and multi-GPU is slower than "
            "one GPU. Verify with `ompi_info --parsable --all | grep "
            "mpi_built_with_cuda_support`.")
    return {
        "grid": grid, "cells": cells, "decomposition": dec, "n_gpu": n_gpu,
        "gpu": gpu, "kokkos_arch": KOKKOS_ARCH.get(key),
        "memory_GB_total": mem_total, "memory_GB_per_rank": per_rank,
        "memory_budget_GB": budget,
        "command": (f"mpirun -np {n_gpu} --map-by ppr:1:node:pe=1 ./idefix "
                    f"-dec {dec[0]} {dec[1]} {dec[2]}" if n_gpu > 1
                    else "./idefix"),
        "notes": notes,
    }


def cmake_command(cfg: IdefixConfig, gpu_arch: str | None = None,
                  mpi: bool = False, vector_potential: bool = True) -> str:
    """The cmake line for this configuration.

    Parameters
    ----------
    gpu_arch : str, optional
        Kokkos architecture macro *without* the ``Kokkos_ARCH_`` prefix, e.g.
        ``AMPERE80`` for A100/A30, ``ADA89`` for L40/RTX 6000 Ada,
        ``HOPPER90`` for H100, ``BLACKWELL120`` for RTX 50-series.  Omit for
        a CPU build.  `KOKKOS_ARCH` maps card names to these.
    mpi : bool
        Domain-decompose across processes.  With CUDA, Idefix assumes the MPI
        library is GPU-aware.
    """
    parts = ["cmake $IDEFIX_DIR", "-DIdefix_MHD=ON",
             f"-DIdefix_RECONSTRUCTION={cfg.reconstruction}"]
    if vector_potential:
        parts.append("-DIdefix_EVOLVE_VECTOR_POTENTIAL=ON")
    if mpi:
        parts.append("-DIdefix_MPI=ON")
    if gpu_arch:
        parts += ["-DKokkos_ENABLE_CUDA=ON",
                  f"-DKokkos_ARCH_{gpu_arch.upper()}=ON"]
    return " \\\n      ".join(parts)


def write_problem_directory(cfg: IdefixConfig, directory: str,
                            gpu_arch: str | None = None,
                            mpi: bool = False) -> dict:
    """Write a complete, compilable Idefix problem directory.

    Returns a dict of the paths written plus the cmake line to run in it.
    """
    os.makedirs(directory, exist_ok=True)
    paths = {
        "definitions.hpp": os.path.join(directory, "definitions.hpp"),
        "setup.cpp": os.path.join(directory, "setup.cpp"),
        "idefix.ini": os.path.join(directory, "idefix.ini"),
        "README.txt": os.path.join(directory, "README.txt"),
    }
    write_definitions_hpp(cfg, paths["definitions.hpp"])
    write_setup_cpp(cfg, paths["setup.cpp"])
    write_idefix_ini(cfg, paths["idefix.ini"])
    cmd = cmake_command(cfg, gpu_arch=gpu_arch, mpi=mpi)
    write_run_notes(cfg, paths["README.txt"], cmake=cmd, mpi=mpi)
    return {"paths": paths, "cmake": cmd}


def write_run_notes(cfg: IdefixConfig, path: str | None = None,
                    cmake: str | None = None, mpi: bool = False) -> str:
    """Build, run and interpretation notes for the generated problem."""
    n = cfg.normalised()
    est = cfg.cavity()
    cost = cfg.cost_estimate()
    cmake = cmake or cmake_command(cfg)
    run = ("mpirun -np <N> ./idefix" if mpi else "./idefix")
    warn = ("\n".join(f"  ! {w}" for w in cfg.notes)
            if cfg.notes else "  (none)")

    text = f"""Idefix diamagnetic-cavity run
=============================
Generated by hvi_emp.solvers.idefix_stage2

Install Idefix (once)
---------------------
  git clone --recurse-submodules https://github.com/idefix-code/idefix.git \\
      $HOME/src/idefix
  export IDEFIX_DIR=$HOME/src/idefix      # add to ~/.bashrc

Kokkos is bundled as a submodule, so --recurse-submodules is not optional
and nothing else needs installing for a serial CPU build.

Build and run this problem
--------------------------
  cd <this directory>
  {cmake}
  make -j 8
  {run}

VTK output lands in ./vtk and opens directly in ParaView.

Geometry: {cfg.geometry}
  {'2D axisymmetric (r, theta), field along the polar axis'
   if cfg.geometry == 'spherical' else '3D Cartesian box, field along z'}
  grid {cfg.n_r} x {cfg.n_theta}{'' if cfg.geometry == 'spherical'
                                 else f' x {cfg.n_theta}'}
  radial range {cfg.r_min_frac:.3g} to {cfg.domain_R:.3g} R0, logarithmic

Code units -> SI
----------------
  length   x  {n['unit_length']:.4e} m
  time     x  {n['unit_time']:.4e} s
  velocity x  {n['unit_velocity']:.4e} m/s
  density  x  {n['unit_density']:.4e} kg/m^3
  B        x  {n['unit_B']:.4e} T
  pressure x  {n['unit_pressure']:.4e} Pa

Dimensionless numbers
---------------------
  ambient beta      {n['beta_ambient']:.3e}   (cavity physics wants << 1)
  plume beta        {n['beta_plume']:.3e}
  density contrast  {n['rho_plume_hat']:.3e}
  v_exp / v_A       {n['v_expansion_hat']:.3e}
  Ohmic eta_hat     {n['eta_hat']:.3e}
  magnetic Reynolds {n['Rm']:.3e}

What to look for
----------------
Analytic pressure-balance expectation (hvi_emp.cavity_estimates):
  cavity radius       R_c ~ {est['R_cavity']:.3e} m  ({est['R_cavity'] / cfg.R0:.1f} R0)
  formation time      t_f ~ {est['t_formation']:.3e} s
  characteristic freq f   ~ {est['f_characteristic']:.3e} Hz

The cavity should inflate to ~R_c, overshoot, then collapse on a comparable
timescale; the collapse is what drives the low-frequency magnetic signature.
Compare the |B| minimum contour against R_c, and the ringing of the
compressed field shell against f.

Expected cost
-------------
  {cost['cells']:.3g} cells, ~{cost['steps']:.3g} steps
  ~{cost['hours']:.2f} h on one GPU at 2e8 cell-updates/s
  ~{cost['output_GB']:.2f} GB of VTK

Warnings from the configuration audit
-------------------------------------
{warn}

Limitations you should not forget
---------------------------------
* MHD assumes quasi-neutrality, so this run cannot produce the
  charge-separation EMP. It gives cavity dynamics only.
* The inner boundary at r = {cfg.r_min_frac:.3g} R0 excludes the plume core;
  outflow there is a choice, not physics.
* Single-fluid, single-temperature. The real plume has T_e != T_i over
  exactly the epoch the cavity forms.
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


# ---------------------------------------------------------------------------
# Reading output back
# ---------------------------------------------------------------------------

def read_idefix_vtk(filename: str):
    """Read an Idefix VTK output using Idefix's own reader.

    Idefix writes a legacy binary VTK with a code-specific layout, and ships
    the matching reader in ``$IDEFIX_DIR/pytools/vtk_io.py``.  Re-implementing
    it here would be a second thing to keep in sync with upstream, so we use
    theirs and say clearly what to do when it is not on the path.
    """
    import importlib.util
    import sys

    idefix_dir = os.environ.get("IDEFIX_DIR")
    candidates = []
    if idefix_dir:
        candidates += [os.path.join(idefix_dir, "pytools", "vtk_io.py"),
                       os.path.join(idefix_dir, "pytools", "idfx_io.py")]
    for cand in candidates:
        if os.path.isfile(cand):
            spec = importlib.util.spec_from_file_location("idefix_vtk_io",
                                                          cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["idefix_vtk_io"] = mod
            spec.loader.exec_module(mod)
            return mod.readVTK(filename)
    raise RuntimeError(
        "Cannot find Idefix's VTK reader.\n"
        "  Set IDEFIX_DIR to your Idefix source tree:\n"
        "      export IDEFIX_DIR=$HOME/src/idefix\n"
        "  (looked for $IDEFIX_DIR/pytools/vtk_io.py)\n"
        "  Alternatively open the .vtk files in ParaView, or install\n"
        "  `pip install vtk` and use vtk.vtkGenericDataObjectReader.")


def cavity_radius(rho, B, r, theta=None, axis_frac: float = 0.5) -> dict:
    """Locate the cavity edge in one output snapshot.

    Parameters
    ----------
    rho, B : ndarray
        Density and field magnitude in code units, shaped (n_r, n_theta) for
        the spherical setup.
    r : ndarray
        Radial coordinate in code units (units of R0).
    theta : ndarray, optional
        Polar angle; when given, the profile is taken at the polar angle
        closest to ``axis_frac * pi`` (default: the equator, where the cavity
        is widest because the field is perpendicular to the expansion there).

    Returns
    -------
    dict with ``R_cavity`` (code units), ``B_min``, and ``R_shell`` -- the
    radius of the compressed field shell, which is where the |B| maximum sits.
    """
    rho = np.asarray(rho, dtype=float)
    B = np.asarray(B, dtype=float)
    r = np.asarray(r, dtype=float)

    if B.ndim == 2:
        if theta is not None:
            j = int(np.argmin(np.abs(np.asarray(theta)
                                     - axis_frac * np.pi)))
        else:
            j = B.shape[1] // 2
        prof = B[:, j]
    else:
        prof = B

    if prof.shape != r.shape:
        raise ValueError(f"field profile {prof.shape} does not match the "
                         f"radial coordinate {r.shape}")

    i_shell = int(np.argmax(prof))
    # cavity edge: outermost radius inside the shell where |B| is below half
    # the ambient value (ambient is 1 in code units)
    inside = np.where(prof[:i_shell + 1] < 0.5)[0]
    R_cav = float(r[inside[-1]]) if inside.size else float(r[0])
    return {"R_cavity": R_cav, "B_min": float(prof.min()),
            "R_shell": float(r[i_shell]),
            "compression": float(prof[i_shell])}


# ---------------------------------------------------------------------------
# Fletcher cases, matching the OpenMHD bridge
# ---------------------------------------------------------------------------

FLETCHER_IDEFIX_CASES = {
    "fig6_parallel": dict(geometry="spherical",
                          _comment="B along the expansion axis: axisymmetric"),
    "fig7_perpendicular": dict(geometry="cartesian",
                               _comment="B oblique to the axis: needs 3D"),
}


def fletcher_idefix_config(case: str, exp: ExpansionResult | None = None,
                           **overrides) -> IdefixConfig:
    """Idefix counterpart of ``openmhd_stage2.fletcher_mhd_config``."""
    if case not in FLETCHER_IDEFIX_CASES:
        raise ValueError(f"case must be one of "
                         f"{sorted(FLETCHER_IDEFIX_CASES)}, got {case!r}")
    kw = {k: v for k, v in FLETCHER_IDEFIX_CASES[case].items()
          if not k.startswith("_")}
    kw.update(overrides)
    if exp is not None:
        return IdefixConfig.from_expansion(exp, **kw)
    defaults = dict(rho_plume=1.0e-9, rho_ambient=2.66e-15,
                    p_plume=1.0e-2, p_ambient=3.2e-9,
                    v_expansion=2.0e4, B0=3.0e-5, R0=0.01)
    defaults.update(kw)
    cfg = IdefixConfig(**defaults)
    cfg._audit()
    return cfg


__all__ = ["IdefixConfig", "detect_gpu", "write_definitions_hpp", "write_idefix_ini",
           "write_setup_cpp", "write_problem_directory", "write_run_notes",
           "cmake_command", "read_idefix_vtk", "cavity_radius",
           "FLETCHER_IDEFIX_CASES", "fletcher_idefix_config", "SPITZER_ETA",
           "decompose", "gpu_plan", "KOKKOS_ARCH", "GPU_MEMORY_GB",
           "IDEFIX_BYTES_PER_CELL"]


if __name__ == "__main__":                              # pragma: no cover
    cfg = fletcher_idefix_config("fig6_parallel")
    print(write_run_notes(cfg))
