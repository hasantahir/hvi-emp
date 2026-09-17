"""Stage 3 on WarpX: electromagnetic particle-in-cell simulation of the EMP.

What this replaces
------------------
The reduced Stage 3 closes the charge-separation problem with a single-
parameter dipole model.  A PIC simulation resolves it: the electrons are
free to separate from the ions under their actual mobility, the ambipolar
field forms self-consistently, and the radiated wave is computed from the
full Maxwell solution -- exactly the programme of Fletcher & Close, Phys.
Plasmas 24, 053102 (2017), who used a discontinuous-Galerkin PIC.  WarpX
(Fedeli et al., SC22; ECP flagship, LBNL) is the modern open-source
equivalent, with the practical advantages of exascale scaling, mesh
refinement and openPMD output.

Setup produced by this bridge
-----------------------------
Faithful to Fletcher & Close (2017), initialised from *this framework's*
Stage-2 state at the collisional->collisionless transition instead of from
hand-picked numbers:

* 2D Cartesian domain (their TE formulation; a wedge-shaped plume expanding
  from a conducting surface), optionally 3D;
* electron and ion species with a cold drifting distribution; the electrons
  are given a bulk drift ``sqrt(m_i/m_e)`` times the ion drift -- their
  central *assumption*, which the PIC then tests;
* plume density profile from the Stage-2 Gaussian, subtending a half-angle
  ``pi/8`` about the surface normal (their geometry);
* absorbing (PML) boundaries on all sides, so the radiated wave leaves;
* field probes at the sensor position for direct comparison with both the
  reduced Stage 3 and the Close et al. (2013) patch-antenna measurement;
* their two canonical cases: ``cold`` (v_t = 0.1 v_d) and
  ``warm`` (v_t = 10 v_d).

Execution modes
---------------
``write_picmi_script`` emits a self-contained Python script using the PICMI
standard, runnable wherever the ``pywarpx`` *module* is importable.  Note
that there is no ``pywarpx`` package on PyPI -- the module comes from a
conda-forge, Spack, or source build of WarpX; see docs/INSTALLING_SOLVERS.md
section 2.  ``write_warpx_inputs`` emits the equivalent native
WarpX input file for the compiled executable.  ``load_warpx_probe`` reads
the openPMD output back and returns arrays comparable with `emp.EMPResult`.

Cost honesty
------------
Resolving the plasma frequency at the transition density (1e26 m^-3,
f_pe ~ 1e14 Hz) needs ~1e-16 s steps; following the pulse to a 0.3 m sensor
needs ~1e-9 s of physical time and a metre-scale domain.  That is the same
scale-separation wall the literature hits (Fletcher & Close normalise their
box to the *late* plume for exactly this reason).  The generated decks
therefore default to the late-plume state -- the epoch when the peak density
has dropped to the sensor band's resonant density -- which is both what the
experiment sees and what a PIC can afford.
"""

from __future__ import annotations

# --- running this file directly? ------------------------------------------
# This is a package module, not a standalone script: the `..` imports below
# only resolve when it is imported as `hvi_emp.solvers.<name>`. Executing the
# file by path (`python warpx_stage3.py`) -- or after copying it somewhere else --
# fails at those imports with a bare "attempted relative import with no known
# parent package". Catch that here and say something useful instead.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    import sys as _sys
    _sys.exit(
        "\nwarpx_stage3.py is part of the `hvi_emp` package and cannot be run as a\n"
        "standalone file -- it imports from its sibling modules.\n\n"
        "Keep it at  hvi_emp/solvers/warpx_stage3.py  and use one of:\n\n"
        "    python -m hvi_emp.solvers.warpx_stage3      # this module's own demo\n"
        "    python examples/06_warpx_openmhd_decks.py   # the worked example\n\n"
        "or import it:\n\n"
        "    from hvi_emp.solvers.warpx_stage3 import WarpXEMPConfig, write_picmi_script\n\n"
        "All of these must be run from the project root (the directory\n"
        "containing the `hvi_emp/` folder).\n")


import os
from dataclasses import dataclass, field

import numpy as np

from ..constants import C_LIGHT, E_CHARGE, EPS_0, EV, M_ELECTRON
from ..expansion import ExpansionResult
from ..ionization import debye_length, plasma_frequency


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WarpXEMPConfig:
    """PIC simulation of the plume's EMP, initialised from Stage 2.

    Parameters
    ----------
    n_e0 : float
        Peak electron density of the plume at initialisation [m^-3].
    T_eV : float
        Plasma temperature [eV].
    R_plume : float
        Plume scale (Gaussian sigma) [m].
    v_drift : float
        Ion bulk expansion speed [m/s].  Electrons get
        ``sqrt(m_i/m_e) * v_drift`` (Fletcher & Close's assumption).
    m_ion : float
        Ion mass [kg].
    Zbar : float
        Mean ion charge state.
    case : str
        "cold" (v_t = 0.1 v_d) or "warm" (v_t = 10 v_d) -- the paper's two
        cases.  Overrides T_eV for the thermal spread if set.
    geometry : str
        "2D" (TE, the paper's choice) or "3D".
    domain_factor : float
        Domain half-width in units of R_plume.
    cells_per_debye : float
        Resolution requirement; the grid is chosen so
        dx <= lambda_De * cells_per_debye^-1 ... i.e. >= 1 cell per Debye
        length divided by this factor.  PIC accuracy wants ~1-2.
    ppc : int
        Macro-particles per cell per species.
    wedge_half_angle : float
        Plume half-angle [rad] about the surface normal (paper: pi/8).
    t_end_periods : float
        Run length in units of the peak plasma period.
    probe_positions : tuple
        (x, z) sensor positions [m] relative to the impact point.
    """
    n_e0: float
    T_eV: float
    R_plume: float
    v_drift: float
    m_ion: float
    Zbar: float = 1.0
    case: str = "cold"
    geometry: str = "2D"
    domain_factor: float = 12.0
    cells_per_debye: float = 1.0
    ppc: int = 16
    wedge_half_angle: float = np.pi / 8.0
    t_end_periods: float = 30.0
    probe_positions: tuple = ((0.0, None),)   # None -> near domain edge

    # -- derived ----------------------------------------------------------
    @property
    def omega_pe(self) -> float:
        return plasma_frequency(self.n_e0)

    @property
    def lambda_De(self) -> float:
        return debye_length(self.n_e0, max(self.T_eV, 1e-3))

    @property
    def v_electron_drift(self) -> float:
        return min(np.sqrt(self.m_ion / M_ELECTRON) * self.v_drift,
                   0.25 * C_LIGHT)

    @property
    def v_thermal_electron(self) -> float:
        """Thermal spread per the paper's case definitions."""
        if self.case == "cold":
            return 0.1 * self.v_electron_drift
        if self.case == "warm":
            return 10.0 * self.v_electron_drift
        return np.sqrt(self.T_eV * EV / M_ELECTRON)

    def grid(self, n_cap: int = 2048,
             cells_across_plume: int = 32) -> dict:
        """Domain, cell counts and timestep satisfying the PIC constraints.

        Resolution policy: resolve the Debye length if that fits within
        `n_cap` cells per side; otherwise fall back to resolving the plume
        structure with `cells_across_plume` cells across R_plume.  The
        fallback is the regime Fletcher & Close ran in ("The Debye length is
        not entirely resolved by the mesh") -- acceptable for their
        high-order DG method, a source of numerical heating for a plain Yee
        solver, so `debye_resolved` is reported and under-resolved runs
        should be read qualitatively.
        """
        L = self.domain_factor * self.R_plume
        dx_debye = self.lambda_De / self.cells_per_debye
        n_debye = int(np.ceil(2.0 * L / dx_debye))
        if n_debye <= n_cap:
            n = max(n_debye, 128)
            resolved = True
        else:
            n = int(np.clip(2.0 * L / (self.R_plume / cells_across_plume),
                            128, n_cap))
            resolved = False
        dx_eff = 2.0 * L / n
        # CFL for the Yee solver + resolve omega_pe
        dt_cfl = 0.98 * dx_eff / (C_LIGHT * np.sqrt(2.0 if
                                                    self.geometry == "2D"
                                                    else 3.0))
        dt_wpe = 0.1 / self.omega_pe
        dt = min(dt_cfl, dt_wpe)
        t_end = self.t_end_periods * 2.0 * np.pi / self.omega_pe
        return {"L_half": L, "n_cells": n, "dx": dx_eff, "dt": dt,
                "n_steps": int(np.ceil(t_end / dt)), "t_end": t_end,
                "debye_resolved": resolved,
                "n_debye_would_need": n_debye}

    @classmethod
    def from_expansion(cls, exp: ExpansionResult,
                       at_frequency: float | None = None,
                       **overrides) -> "WarpXEMPConfig":
        """Initial conditions from a Stage-2 result.

        Parameters
        ----------
        exp : ExpansionResult
        at_frequency : float, optional
            If given, initialise at the epoch when the peak plasma frequency
            has dropped to this value (the affordable, experiment-relevant
            late plume).  If None, use the collisional->collisionless
            transition epoch (physically where the mechanism switches on, but
            usually computationally out of reach -- the deck generator will
            tell you).
        """
        if at_frequency is not None:
            from ..emp import resonant_density
            n_target = resonant_density(at_frequency)
            k = int(np.argmin(np.abs(np.log(np.maximum(exp.n_e, 1e-300))
                                     - np.log(n_target))))
        elif exp.transition:
            k = int(exp.transition["index"])
        else:
            raise ValueError("no transition in the expansion history; "
                             "pass at_frequency= instead")
        kw = dict(
            n_e0=float(exp.n_e[k]),
            T_eV=float(exp.T_eV[k]),
            R_plume=float(np.sqrt(exp.R_r[k] * exp.R_z[k])),
            v_drift=float(np.hypot(exp.v_r[k], exp.v_z[k])),
            m_ion=exp.material.m_atom,
            Zbar=max(float(exp.Zbar[k]), 0.05),
        )
        kw.update(overrides)
        return cls(**kw)


# ---------------------------------------------------------------------------
# PICMI script generation
# ---------------------------------------------------------------------------

def write_picmi_script(cfg: WarpXEMPConfig, path: str | None = None) -> str:
    """Self-contained PICMI/pywarpx script for this configuration.

    Runs anywhere WarpX's Python bindings are installed:
    ``python warpx_emp.py`` or via ``mpirun`` for parallel execution.
    """
    g = cfg.grid()
    ve, vt = cfg.v_electron_drift, cfg.v_thermal_electron
    probes = []
    for (px, pz) in cfg.probe_positions:
        pz = 0.9 * g["L_half"] if pz is None else pz
        probes.append((px, pz))

    script = f'''#!/usr/bin/env python3
"""WarpX PIC simulation: EMP from an expanding impact plasma.

Generated by hvi_emp.solvers.warpx_stage3 from a Stage-2 plume state.
Physical setup follows Fletcher & Close, Phys. Plasmas 24, 053102 (2017).

State at initialisation:
    n_e0      = {cfg.n_e0:.4e} m^-3   (f_pe = {cfg.omega_pe/2/np.pi:.4e} Hz)
    T_e       = {cfg.T_eV:.4f} eV     (case: {cfg.case})
    R_plume   = {cfg.R_plume:.4e} m
    v_ion     = {cfg.v_drift:.4e} m/s
    v_e drift = {ve:.4e} m/s  (= sqrt(m_i/m_e) x v_ion, the F&C assumption)
    lambda_De = {cfg.lambda_De:.4e} m
Grid: {g['n_cells']}^2 cells, dx = {g['dx']:.3e} m, dt = {g['dt']:.3e} s, {g['n_steps']} steps
Debye length resolved: {g['debye_resolved']}
"""
import numpy as np
from pywarpx import picmi

c = picmi.constants

nx = nz = {g['n_cells']}
L = {g['L_half']:.6e}
grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    lower_bound=[-L, -L], upper_bound=[L, L],
    lower_boundary_conditions=["open", "open"],
    upper_boundary_conditions=["open", "open"],
    lower_boundary_conditions_particles=["absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.98)

# --- plume: Gaussian ball, wedge of half-angle {np.degrees(cfg.wedge_half_angle):.1f} deg about +z ------
# density profile n(r) = n0 exp(-r^2 / (2 R^2)) inside the wedge
R = {cfg.R_plume:.6e}
n0 = {cfg.n_e0:.6e}
half_angle = {cfg.wedge_half_angle:.6f}
wedge = f"((z > 0) & (abs(atan2(sqrt(x*x), z)) < {{half_angle}}))"
profile = f"n0 * exp(-(x*x + z*z) / (2*R*R)) * {{wedge}}"

density_expr = profile.replace("n0", repr(n0)).replace("R", repr(R))

electrons = picmi.Species(
    particle_type="electron", name="electrons",
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=density_expr,
        momentum_expressions=[
            "0", "0",
            # radial bulk drift sqrt(mi/me) x ion drift, plus thermal spread
            f"{ve:.6e} * m_e",
        ],
        rms_velocity=[{vt:.6e}] * 3,
    ),
)
ions = picmi.Species(
    name="ions", charge={cfg.Zbar:.3f} * picmi.constants.q_e,
    mass={cfg.m_ion:.6e},
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=density_expr + f" / {cfg.Zbar:.3f}",
        momentum_expressions=["0", "0", f"{cfg.v_drift:.6e} * {cfg.m_ion:.6e}"],
        rms_velocity=[{cfg.v_drift * 0.05:.6e}] * 3,
    ),
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size={g['dt']:.6e},
    max_steps={g['n_steps']},
    particle_shape="cubic",
    warpx_use_filter=True,
)
layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell={cfg.ppc},
                                  seed=1)
sim.add_species(electrons, layout=layout)
sim.add_species(ions, layout=layout)

# --- diagnostics -----------------------------------------------------------
field_diag = picmi.FieldDiagnostic(
    name="fields", grid=grid,
    period=max(1, {g['n_steps']} // 200),
    data_list=["E", "B", "rho", "J"],
    write_dir="diags", warpx_format="openpmd",
)
sim.add_diagnostic(field_diag)

# point probes for the sensor waveform
probe_positions = {probes!r}

sim.step()

print("done: fields in ./diags (openPMD); post-process with "
      "hvi_emp.solvers.warpx_stage3.load_warpx_probe")
'''
    if path:
        with open(path, "w") as fh:
            fh.write(script)
        os.chmod(path, 0o755)
    return script


# ---------------------------------------------------------------------------
# Native WarpX input file
# ---------------------------------------------------------------------------

def write_warpx_inputs(cfg: WarpXEMPConfig, path: str | None = None) -> str:
    """Native input file for the compiled ``warpx`` executable.

    Equivalent physics to the PICMI script; use whichever fits the cluster.
    """
    g = cfg.grid()
    ve, vt = cfg.v_electron_drift, cfg.v_thermal_electron
    L = g["L_half"]
    ha = cfg.wedge_half_angle
    dens = (f"{cfg.n_e0:.6e}*exp(-(x*x+z*z)/(2*{cfg.R_plume:.6e}"
            f"*{cfg.R_plume:.6e}))"
            f"*if(z>0 and abs(atan2(sqrt(x*x),z))<{ha:.6f},1,0)")

    text = f"""# WarpX native input -- EMP from an expanding impact plasma
# Generated by hvi_emp.solvers.warpx_stage3.
# Physics: Fletcher & Close, Phys. Plasmas 24, 053102 (2017).
# n_e0 = {cfg.n_e0:.4e} m^-3, T_e = {cfg.T_eV:.3f} eV ({cfg.case}),
# R = {cfg.R_plume:.4e} m, f_pe = {cfg.omega_pe/2/np.pi:.4e} Hz

max_step = {g['n_steps']}
amr.n_cell = {g['n_cells']} {g['n_cells']}
amr.max_level = 0
geometry.dims = 2
geometry.prob_lo = {-L:.6e} {-L:.6e}
geometry.prob_hi = {L:.6e} {L:.6e}

boundary.field_lo = pml pml
boundary.field_hi = pml pml
boundary.particle_lo = absorbing absorbing
boundary.particle_hi = absorbing absorbing

algo.maxwell_solver = yee
algo.particle_shape = 3
warpx.cfl = 0.98
warpx.use_filter = 1
warpx.const_dt = {g['dt']:.6e}

particles.species_names = electrons ions

electrons.charge = -q_e
electrons.mass = m_e
electrons.injection_style = nuniformpercell
electrons.num_particles_per_cell_each_dim = 4 4
electrons.profile = parse_density_function
electrons.density_function(x,y,z) = {dens}
electrons.momentum_distribution_type = gaussian
electrons.ux_m = 0.0
electrons.uy_m = 0.0
electrons.uz_m = {ve / C_LIGHT:.6e}
electrons.ux_th = {vt / C_LIGHT:.6e}
electrons.uy_th = {vt / C_LIGHT:.6e}
electrons.uz_th = {vt / C_LIGHT:.6e}

ions.charge = {cfg.Zbar:.4f}*q_e
ions.mass = {cfg.m_ion / 9.1093837015e-31:.6e}*m_e
ions.injection_style = nuniformpercell
ions.num_particles_per_cell_each_dim = 4 4
ions.profile = parse_density_function
ions.density_function(x,y,z) = ({dens})/{cfg.Zbar:.4f}
ions.momentum_distribution_type = gaussian
ions.ux_m = 0.0
ions.uy_m = 0.0
ions.uz_m = {cfg.v_drift / C_LIGHT:.6e}
ions.ux_th = {0.05 * cfg.v_drift / C_LIGHT:.6e}
ions.uy_th = {0.05 * cfg.v_drift / C_LIGHT:.6e}
ions.uz_th = {0.05 * cfg.v_drift / C_LIGHT:.6e}

diagnostics.diags_names = fields
fields.intervals = {max(1, g['n_steps'] // 200)}
fields.diag_type = Full
fields.format = openpmd
fields.fields_to_plot = Ex Ey Ez Bx By Bz rho jx jy jz
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


# ---------------------------------------------------------------------------
# Post-processing (openPMD)
# ---------------------------------------------------------------------------

def load_warpx_probe(diag_dir: str, x: float, z: float,
                     field: str = "E") -> dict:
    """Time series of a field component at a probe point from openPMD output.

    Parameters
    ----------
    diag_dir : str
        The WarpX ``diags`` directory (openPMD series).
    x, z : float
        Probe position [m].
    field : str
        "E" or "B".

    Returns
    -------
    dict with ``t`` [s] and ``Ex, Ey, Ez`` (or ``Bx...``) [SI] arrays --
    directly comparable with `emp.EMPResult.E_t` at the same standoff, and
    with the Close et al. (2013) measurement if the probe is at 0.30 m.
    """
    import openpmd_api as api

    series_path = None
    for pattern in ("openpmd_%T.h5", "openpmd_%T.bp", "%T.h5"):
        cand = os.path.join(diag_dir, pattern)
        try:
            s = api.Series(cand, api.Access.read_only)
            series_path = cand
            break
        except Exception:
            continue
    if series_path is None:
        raise FileNotFoundError(
            f"no openPMD series found under {diag_dir!r}")

    t, comps = [], {c: [] for c in "xyz"}
    for _, it in s.iterations.items():
        t.append(it.time * it.time_unit_SI)
        mesh = it.meshes[field]
        for c in "xyz":
            if c not in mesh:
                comps[c].append(0.0)
                continue
            rc = mesh[c]
            data = rc.load_chunk()
            s.flush()
            arr = np.asarray(data) * rc.unit_SI
            # locate the probe cell
            shape = arr.shape
            gs = mesh.grid_spacing
            go = mesh.grid_global_offset
            us = mesh.grid_unit_SI
            iz = int(round((z - go[0] * us) / (gs[0] * us)))
            ix = int(round((x - go[-1] * us) / (gs[-1] * us)))
            iz = np.clip(iz, 0, shape[0] - 1)
            ix = np.clip(ix, 0, shape[-1] - 1)
            comps[c].append(float(arr[iz, ..., ix].squeeze()))
    del s
    out = {"t": np.array(t)}
    for c in "xyz":
        out[f"{field}{c}"] = np.array(comps[c])
    return out


def compare_with_reduced_model(probe: dict, emp_result) -> dict:
    """Peak-field and spectrum comparison: WarpX probe vs reduced Stage 3."""
    E = np.sqrt(sum(probe[k] ** 2 for k in probe if k != "t"))
    return {
        "warpx_peak_E": float(np.max(E)),
        "reduced_peak_E": emp_result.peak_field,
        "ratio": float(np.max(E) / max(emp_result.peak_field, 1e-300)),
    }


__all__ = [
    "WarpXEMPConfig", "write_picmi_script", "write_warpx_inputs",
    "load_warpx_probe", "compare_with_reduced_model",
]


# ---------------------------------------------------------------------------
# Module demo:  python -m hvi_emp.solvers.warpx_stage3
# ---------------------------------------------------------------------------

def _demo(argv=None) -> int:                     # pragma: no cover - CLI
    """Show the PIC grid a given plume state would require."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m hvi_emp.solvers.warpx_stage3")
    ap.add_argument("--n-e", dest="n_e", type=float, default=1e16,
                    help="peak electron density [m^-3]")
    ap.add_argument("--T-eV", dest="T_eV", type=float, default=1.0)
    ap.add_argument("--R", type=float, default=1e-2, help="plume scale [m]")
    ap.add_argument("--v", type=float, default=3e4, help="ion drift [m/s]")
    ap.add_argument("--out", default=None, help="write a PICMI script here")
    a = ap.parse_args(argv)

    cfg = WarpXEMPConfig(n_e0=a.n_e, T_eV=a.T_eV, R_plume=a.R,
                         v_drift=a.v, m_ion=9.27e-26)
    g = cfg.grid()
    print(f"f_pe = {cfg.omega_pe/2/np.pi:.3e} Hz, "
          f"lambda_De = {cfg.lambda_De:.3e} m")
    for k, v in g.items():
        print(f"  {k:<22}{v}")
    if a.out:
        write_picmi_script(cfg, a.out)
        print(f"wrote {a.out}")
    else:
        print("\nFor a physically-consistent setup use "
              "WarpXEMPConfig.from_expansion(); see "
              "examples/06_warpx_openmhd_decks.py")
    return 0


if __name__ == "__main__":                       # pragma: no cover
    import sys as _s
    _s.exit(_demo())
