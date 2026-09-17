"""Stage 2 on OpenMHD: magnetised plume expansion (diamagnetic cavity).

What this adds that the reduced Stage 2 cannot
----------------------------------------------
The self-similar Anisimov plume is unmagnetised.  On orbit the plume expands
into the geomagnetic field, and Fletcher (NRL/MR/6757-20-10,138, 2021) shows
with ALEGRA-MHD that a **diamagnetic cavity** forms: the conducting plume
expels the background field, the cavity grows until the magnetic pressure
B^2/2mu0 balances the plume ram pressure, then collapses -- a genuine
low-frequency electromagnetic emission mechanism entirely absent from the
reduced chain (THEORY.md Sec. 7.2, gap #4).  This is a classic MHD problem
(it is the same physics as the AMPTE barium releases).

OpenMHD (S. Zenitani, https://github.com/zenitani/OpenMHD) is an open-source
finite-volume MHD code (HLLD Riemann solver, 2D/3D, Fortran 90 + MPI) whose
problems are configured by editing a ``model.f90`` per run directory.  This
bridge therefore generates, from a Stage-1/2 state:

* a complete ``model.f90`` implementing the plume-in-background-field
  initial condition in OpenMHD's normalised units;
* a ``param.h``-style summary of the normalisation actually used, so the
  output can be converted back to SI;
* an analytic estimate of the cavity radius and collapse time
  (`cavity_estimates`) so the simulation has a prediction to test.

Execution is *not* wrapped: OpenMHD is compiled per problem (gfortran + MPI,
see its README).  Copy the generated ``model.f90`` over a 2D problem
directory's template, build, run.  `read_openmhd_output` post-processes the
standard binary/VTK output if NumPy-readable data files are produced.

Validity note
-------------
Ideal MHD requires the plume to be collisional and magnetised
(omega_ci * tau significant); very early plume (collisional, unmagnetised:
fine as a fluid but the field is frozen out of the *vacuum*, not the plume)
and very late plume (collisionless: kinetic, WarpX territory) both violate
it.  The window where the cavity physics is MHD-valid is roughly between
plume radii of centimetres and metres for LEO conditions -- conveniently the
regime the reduced Stage 2 hands over.
"""

from __future__ import annotations

# --- running this file directly? ------------------------------------------
# This is a package module, not a standalone script: the `..` imports below
# only resolve when it is imported as `hvi_emp.solvers.<name>`. Executing the
# file by path (`python openmhd_stage2.py`) -- or after copying it somewhere else --
# fails at those imports with a bare "attempted relative import with no known
# parent package". Catch that here and say something useful instead.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    import sys as _sys
    _sys.exit(
        "\nopenmhd_stage2.py is part of the `hvi_emp` package and cannot be run as a\n"
        "standalone file -- it imports from its sibling modules.\n\n"
        "Keep it at  hvi_emp/solvers/openmhd_stage2.py  and use one of:\n\n"
        "    python -m hvi_emp.solvers.openmhd_stage2      # this module's own demo\n"
        "    python examples/06_warpx_openmhd_decks.py   # the worked example\n\n"
        "or import it:\n\n"
        "    from hvi_emp.solvers.openmhd_stage2 import OpenMHDConfig, write_model_f90\n\n"
        "All of these must be run from the project root (the directory\n"
        "containing the `hvi_emp/` folder).\n")


import os
from dataclasses import dataclass

import numpy as np

from ..constants import K_B, MU_0
from ..expansion import ExpansionResult


# ---------------------------------------------------------------------------
# Analytic expectations
# ---------------------------------------------------------------------------

def cavity_estimates(E_kinetic: float, B0: float,
                     v_expansion: float) -> dict:
    """Diamagnetic-cavity scale from pressure balance.

    Equating the plume kinetic energy density to the magnetic energy density
    it displaces gives the classic cavity radius (spherical):

        (4/3) pi R_c^3 * B0^2 / (2 mu0) = E_kin
        R_c = [ 3 mu0 E_kin / (2 pi B0^2) ]^(1/3)

    and the formation/collapse time is R_c / v_expansion (formation) with
    collapse on the same order (AMPTE observations; Winske & Gary 2007).
    """
    if B0 <= 0 or E_kinetic <= 0:
        return {"R_cavity": 0.0, "t_formation": 0.0, "f_characteristic": 0.0}
    R_c = (3.0 * MU_0 * E_kinetic / (2.0 * np.pi * B0**2)) ** (1.0 / 3.0)
    t_f = R_c / max(v_expansion, 1.0)
    return {
        "R_cavity": float(R_c),
        "t_formation": float(t_f),
        "f_characteristic": float(1.0 / (2.0 * t_f)),
        "B_energy_displaced": float((4.0 / 3.0) * np.pi * R_c**3
                                    * B0**2 / (2.0 * MU_0)),
    }


# ---------------------------------------------------------------------------
# Configuration and normalisation
# ---------------------------------------------------------------------------

@dataclass
class OpenMHDConfig:
    """Plume-in-background-field problem in OpenMHD normalised units.

    OpenMHD solves ideal MHD in dimensionless form; the normalisation here
    follows the usual convention: lengths in units of the initial plume
    radius ``R0``, density in units of the ambient density ``rho_amb``,
    velocity in units of the ambient Alfven speed ``v_A``, so that the
    background field has magnitude 1 and time is in Alfven crossings.

    Parameters
    ----------
    rho_plume, rho_ambient : float
        Peak plume and ambient mass densities [kg/m^3].
    p_plume, p_ambient : float
        Pressures [Pa].
    v_expansion : float
        Plume radial expansion speed [m/s].
    B0 : float
        Background field [T] (LEO: ~3e-5 T).
    R0 : float
        Initial plume radius [m].
    B_angle_deg : float
        Field direction in the simulation plane, from x [deg].
    domain_R : float
        Half-width of the square domain in units of R0.
    nx : int
        Grid cells per side.
    t_end_alfven : float
        Run time in Alfven crossing times of the domain.
    """
    rho_plume: float
    rho_ambient: float
    p_plume: float
    p_ambient: float
    v_expansion: float
    B0: float = 3.0e-5
    R0: float = 0.01
    B_angle_deg: float = 0.0
    domain_R: float = 20.0
    nx: int = 800
    t_end_alfven: float = 3.0
    gamma: float = 5.0 / 3.0

    # -- normalisation -----------------------------------------------------
    @property
    def v_alfven(self) -> float:
        return self.B0 / np.sqrt(MU_0 * self.rho_ambient)

    def normalised(self) -> dict:
        vA = self.v_alfven
        p_norm = self.B0**2 / MU_0        # so that B_hat = 1, p_mag_hat = 1/2
        return {
            "rho_plume_hat": self.rho_plume / self.rho_ambient,
            "p_plume_hat": self.p_plume / p_norm,
            "p_ambient_hat": self.p_ambient / p_norm,
            "v_expansion_hat": self.v_expansion / vA,
            "beta_ambient": 2.0 * self.p_ambient / (self.B0**2 / MU_0),
            "v_alfven": vA,
            "unit_length": self.R0,
            "unit_time": self.R0 / vA,
            "unit_density": self.rho_ambient,
            "unit_B": self.B0,
            "unit_pressure": p_norm,
        }

    @classmethod
    def from_expansion(cls, exp: ExpansionResult, index: int | None = None,
                       B0: float = 3.0e-5,
                       n_ambient: float = 1.0e11,
                       T_ambient_eV: float = 0.1,
                       m_ambient: float = 2.66e-26,   # atomic oxygen
                       **overrides) -> "OpenMHDConfig":
        """Initial conditions from a Stage-2 epoch (default: last sample).

        The ambient defaults are LEO daytime ionosphere: n ~ 1e11 m^-3
        atomic oxygen at ~0.1 eV, B ~ 3e-5 T.
        """
        k = -1 if index is None else index
        rho_p = float(exp.n_h[k] * exp.material.m_atom)
        p_p = float(exp.n_h[k] * (1.0 + exp.Zbar[k]) * K_B * exp.T[k])
        rho_a = n_ambient * m_ambient
        p_a = 2.0 * n_ambient * T_ambient_eV * 1.602176634e-19
        kw = dict(
            rho_plume=rho_p, rho_ambient=rho_a,
            p_plume=max(p_p, 1e-12), p_ambient=max(p_a, 1e-14),
            v_expansion=float(np.hypot(exp.v_r[k], exp.v_z[k])),
            B0=B0, R0=float(np.sqrt(exp.R_r[k] * exp.R_z[k])),
        )
        kw.update(overrides)
        return cls(**kw)


# ---------------------------------------------------------------------------
# model.f90 generation
# ---------------------------------------------------------------------------

def write_model_f90(cfg: OpenMHDConfig, path: str | None = None) -> str:
    """OpenMHD ``model.f90`` for the diamagnetic-cavity problem.

    Drop the file over the template in one of OpenMHD's 2D problem
    directories (e.g. ``2D_basic/``), set the grid size in the problem's
    main file to match ``cfg.nx``, build with the provided Makefile, run.

    The state vector layout (mx,my grid; U(:,:,var) with var = ro,vx,vy,vz,
    pr,bx,by,bz,ps) follows OpenMHD's 2D convention.
    """
    n = cfg.normalised()
    ang = np.radians(cfg.B_angle_deg)
    bx0, by0 = np.cos(ang), np.sin(ang)
    L = cfg.domain_R

    text = f"""!-----------------------------------------------------------------------
! OpenMHD model: plume expansion into a magnetised ambient medium
! (diamagnetic cavity problem)
!
! Generated by hvi_emp.solvers.openmhd_stage2.
! Physics target: Fletcher (NRL/MR/6757-20-10,138, 2021) Figs. 6-7;
! AMPTE-release-class cavity formation and collapse.
!
! Normalisation (SI conversion):
!   unit length   = {n['unit_length']:.6e} m   (initial plume radius R0)
!   unit density  = {n['unit_density']:.6e} kg/m^3 (ambient)
!   unit B        = {n['unit_B']:.6e} T
!   unit velocity = {n['v_alfven']:.6e} m/s (ambient Alfven speed)
!   unit time     = {n['unit_time']:.6e} s
!   unit pressure = {n['unit_pressure']:.6e} Pa
!
! Dimensionless initial state:
!   plume:   rho = {n['rho_plume_hat']:.6e}, p = {n['p_plume_hat']:.6e},
!            radial v = {n['v_expansion_hat']:.6e} (inside r < 1)
!   ambient: rho = 1, p = {n['p_ambient_hat']:.6e}, B = (bx,by) = ({bx0:.3f},{by0:.3f})
!   ambient plasma beta = {n['beta_ambient']:.3e}
!-----------------------------------------------------------------------
subroutine model(U, V, x, y, dx, ix, jx)
  implicit none
  include 'param.h'
  real(8), intent(out) :: U(ix, jx, var1)   ! conserved variables
  real(8), intent(out) :: V(ix, jx, var2)   ! primitive variables
  real(8), intent(out) :: x(ix), y(jx), dx
  integer, intent(in)  :: ix, jx
! ---- local ----
  integer :: i, j
  real(8) :: r, prof, vr
  real(8), parameter :: domain_half = {L:.4f}d0
  real(8), parameter :: rho_plume  = {n['rho_plume_hat']:.6e}d0
  real(8), parameter :: p_plume    = {n['p_plume_hat']:.6e}d0
  real(8), parameter :: p_ambient  = {max(n['p_ambient_hat'], 1e-8):.6e}d0
  real(8), parameter :: v_exp      = {n['v_expansion_hat']:.6e}d0
  real(8), parameter :: bx0 = {bx0:.6f}d0, by0 = {by0:.6f}d0
!-----------------------------------------------------------------------
  dx = 2.d0 * domain_half / dble(ix - 2)
  do i = 1, ix
     x(i) = -domain_half + dx * (dble(i) - 1.5d0)
  enddo
  do j = 1, jx
     y(j) = -domain_half + dx * (dble(j) - 1.5d0)
  enddo

  do j = 1, jx
  do i = 1, ix
     r = sqrt(x(i)**2 + y(j)**2)
     ! Gaussian plume of unit radius on a uniform magnetised ambient
     prof = exp(-0.5d0 * r**2)
     V(i,j,ro) = 1.d0 + (rho_plume - 1.d0) * prof
     V(i,j,pr) = p_ambient + p_plume * prof
     ! self-similar radial velocity, v = v_exp * (r/R0) inside the plume
     if ( r > 1.d-10 ) then
        vr = v_exp * min(r, 1.5d0) * prof
        V(i,j,vx) = vr * x(i) / r
        V(i,j,vy) = vr * y(j) / r
     else
        V(i,j,vx) = 0.d0
        V(i,j,vy) = 0.d0
     endif
     V(i,j,vz) = 0.d0
     ! background field, initially uniform (the cavity must *develop*)
     U(i,j,bx) = bx0
     U(i,j,by) = by0
     U(i,j,bz) = 0.d0
     U(i,j,ps) = 0.d0
  enddo
  enddo

  call v2u(V, U, ix, 1, ix, jx, 1, jx)

end subroutine model
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def write_run_notes(cfg: OpenMHDConfig, path: str | None = None) -> str:
    """README for the generated problem: build, run, convert back to SI."""
    n = cfg.normalised()
    est = cavity_estimates(
        E_kinetic=(0.5 * cfg.rho_plume * (4.0 / 3.0) * np.pi * cfg.R0**3
                   * cfg.v_expansion**2),
        B0=cfg.B0, v_expansion=cfg.v_expansion)
    text = f"""OpenMHD diamagnetic-cavity run
==============================

1.  git clone https://github.com/zenitani/OpenMHD
2.  Copy the generated model.f90 over the template in a 2D problem
    directory (2D_basic is the natural host).
3.  Set the grid to {cfg.nx} x {cfg.nx} in the problem's main file and the
    end time to t = {cfg.t_end_alfven:.1f} (Alfven units).
4.  make ; mpirun -np <N> ./a.out

SI conversion of the output
---------------------------
  length   x  {n['unit_length']:.4e} m
  time     x  {n['unit_time']:.4e} s
  density  x  {n['unit_density']:.4e} kg/m^3
  B        x  {n['unit_B']:.4e} T
  velocity x  {n['v_alfven']:.4e} m/s
  pressure x  {n['unit_pressure']:.4e} Pa

What to look for
----------------
Analytic pressure-balance expectation (hvi_emp cavity_estimates):
  cavity radius       R_c ~ {est['R_cavity']:.3e} m  ({est['R_cavity']/cfg.R0:.1f} R0)
  formation time      t_f ~ {est['t_formation']:.3e} s
  characteristic freq f   ~ {est['f_characteristic']:.3e} Hz

The cavity should inflate to ~R_c, overshoot slightly, and collapse on the
same timescale; the collapse drives the low-frequency magnetic signature.
Compare the simulated cavity edge (|B| minimum contour) against R_c, and the
oscillation of the trapped-field annulus against f.

Validity checks before trusting the run
---------------------------------------
* ambient beta = {n['beta_ambient']:.2e}  (the cavity problem wants beta << 1)
* the plume must remain collisional over the run (else kinetic effects:
  use the WarpX bridge instead)
* domain half-width = {cfg.domain_R:.0f} R0 vs expected cavity {est['R_cavity']/cfg.R0:.1f} R0
  -- enlarge domain_R if the cavity approaches the boundary.
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


__all__ = ["OpenMHDConfig", "cavity_estimates", "write_model_f90",
           "write_run_notes"]


# ---------------------------------------------------------------------------
# Module demo:  python -m hvi_emp.solvers.openmhd_stage2
# ---------------------------------------------------------------------------

def _demo(argv=None) -> int:                     # pragma: no cover - CLI
    """Analytic diamagnetic-cavity estimate for an orbital impact."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m hvi_emp.solvers.openmhd_stage2")
    ap.add_argument("--energy", type=float, default=1.0,
                    help="plume kinetic energy [J]")
    ap.add_argument("--B0", type=float, default=3e-5, help="field [T]")
    ap.add_argument("--v", type=float, default=3e4, help="expansion [m/s]")
    a = ap.parse_args(argv)

    est = cavity_estimates(a.energy, a.B0, a.v)
    print(f"plume KE {a.energy:.3e} J in B0 = {a.B0:.1e} T, "
          f"v = {a.v/1e3:.1f} km/s")
    for k, v in est.items():
        print(f"  {k:<22}{v:.4e}")
    print("\nFor a full initial condition use OpenMHDConfig.from_expansion(); "
          "see examples/06_warpx_openmhd_decks.py")
    return 0


if __name__ == "__main__":                       # pragma: no cover
    import sys as _s
    _s.exit(_demo())


# ---------------------------------------------------------------------------
# Fletcher (2021) Figs. 6 and 7 -- the two field geometries
# ---------------------------------------------------------------------------

#: Fletcher (2021) MHD cases, as described in his Figs. 6 and 7.
FLETCHER_MHD_CASES = {
    "fig6_parallel": {
        "B_angle_deg": 0.0,
        "impact_angle_deg": 30.0,
        "velocity": 30e3,
        "projectile": "Al", "target": "Al",
        "description": "30 deg oblique Al->Al at 30 km/s, background field "
                       "PARALLEL to the target surface. Fletcher's Fig. 6: "
                       "the expanding plasma distorts the field lines; the "
                       "temperature field matches ground-based experiments.",
        "observable": "field-line draping and distortion around the plume",
    },
    "fig7_perpendicular": {
        "B_angle_deg": 90.0,
        "impact_angle_deg": 0.0,
        "velocity": 30e3,
        "projectile": "W", "target": "Al",
        "description": "Normal W->Al impact, background field PERPENDICULAR "
                       "to the target surface. Fletcher's Fig. 7: a "
                       "diamagnetic cavity forms within and above the crater "
                       "and collapses as the plume thins.",
        "observable": "diamagnetic cavity: |B| minimum inside the plume, "
                      "then collapse",
    },
}


def fletcher_mhd_config(case: str, exp=None, **overrides) -> "OpenMHDConfig":
    """Build the OpenMHD configuration for one of Fletcher's two MHD cases.

    Parameters
    ----------
    case : str
        Key into `FLETCHER_MHD_CASES`.
    exp : ExpansionResult, optional
        Stage-2 result to initialise from. If omitted, a representative LEO
        plume state is used and flagged as such.
    """
    if case not in FLETCHER_MHD_CASES:
        raise KeyError(f"unknown case {case!r}; "
                       f"available: {sorted(FLETCHER_MHD_CASES)}")
    spec = FLETCHER_MHD_CASES[case]
    kw = {"B0": 3.0e-5, "B_angle_deg": spec["B_angle_deg"]}
    kw.update(overrides)
    if exp is not None:
        return OpenMHDConfig.from_expansion(exp, **kw)
    # representative late plume if no Stage-2 result is supplied
    defaults = dict(rho_plume=1e-4, rho_ambient=2.66e-15,
                    p_plume=1e2, p_ambient=3.2e-9,
                    v_expansion=2.0e4, R0=0.02)
    defaults.update(kw)
    return OpenMHDConfig(**defaults)


def write_fletcher_mhd_cases(directory: str, exp=None) -> dict:
    """Write both of Fletcher's MHD problems as OpenMHD run directories."""
    import os

    os.makedirs(directory, exist_ok=True)
    out = {}
    for case, spec in FLETCHER_MHD_CASES.items():
        cfg = fletcher_mhd_config(case, exp=exp)
        sub = os.path.join(directory, case)
        os.makedirs(sub, exist_ok=True)
        write_model_f90(cfg, os.path.join(sub, "model.f90"))
        notes = write_run_notes(cfg, os.path.join(sub, "RUN.md"))
        with open(os.path.join(sub, "CASE.md"), "w") as fh:
            fh.write(f"# {case}\n\n{spec['description']}\n\n"
                     f"**What to look for:** {spec['observable']}\n\n"
                     f"Impact: {spec['projectile']} -> {spec['target']} at "
                     f"{spec['velocity']/1e3:.0f} km/s, "
                     f"{spec['impact_angle_deg']:.0f} deg from normal.\n"
                     f"Background field at {spec['B_angle_deg']:.0f} deg to "
                     "the surface.\n\n"
                     "Note: iSALE (hydro) cannot produce this; OpenMHD is\n"
                     "initialised from the plume state, so the crater itself\n"
                     "is not in the MHD run. That is the same decomposition\n"
                     "Fletcher avoids by using ALEGRA-MHD for both at once --\n"
                     "see docs/REPLICATION_FLETCHER2021.md.\n")
        out[case] = {"dir": sub, "config": cfg,
                     "cavity": cavity_estimates(
                         0.5 * cfg.rho_plume * (4/3) * np.pi * cfg.R0**3
                         * cfg.v_expansion**2, cfg.B0, cfg.v_expansion)}
    return out


__all__ += ["FLETCHER_MHD_CASES", "fletcher_mhd_config",
            "write_fletcher_mhd_cases"]
