"""Stage 3 on PIConGPU: charge-separation EMP from the expanding plume.

PIConGPU (Bussmann et al., SC'13; https://github.com/ComputationalRadiationPhysics/picongpu)
is a fully relativistic 3D3V electromagnetic PIC code that has been GPU-native
since its first release.  It sits on alpaka, so the same source runs on CUDA,
HIP, SYCL and OpenMP, it writes openPMD natively, and it is GPLv3.

Why PIConGPU for this stage
---------------------------
Stage 3 is where the framework is weakest: the charge-separation closure is a
single parameter, and the absolute EMP amplitude is uncertain by two orders of
magnitude.  Resolving it needs a kinetic EM code that actually separates
electrons from ions, and the calculation is expensive enough that GPU
efficiency decides whether it happens at all.

Two features map onto this problem unusually well:

* **Probe particles.**  A stationary, non-interacting species carrying
  ``probeE`` and ``probeB`` records the fields at chosen points over time --
  which is precisely a patch antenna at 0.30 m, the measurement in Close et
  al. (2013) that Stage 3 is calibrated against.  No full-field dumps needed.
* **Anisotropic Gaussian density.**  ``GaussianCloudImpl`` takes a per-axis
  sigma, so the Anisimov ellipsoidal plume that Stage 2 integrates maps onto
  it directly rather than through a hand-written density functor.

The scale-separation wall is unchanged
--------------------------------------
Resolving the Debye length and the plasma period at the collisional ->
collisionless transition density (~1e26 m^-3) needs ~1e-16 s steps and
sub-nanometre cells, while the pulse takes ~1e-9 s to reach a sensor 0.3 m
away.  No code removes that; PIConGPU makes the affordable window bigger, not
infinite.  Like the WarpX bridge, this one therefore defaults to the *late*
plume -- the epoch when the peak density has fallen to the resonant density of
the band you are measuring -- and ``grid()`` says plainly when even that is
under-resolved.

Execution is not wrapped in-process: PIConGPU compiles its .param files into
the binary.  The bridge writes a complete input set and the exact
pic-create / pic-build / tbg commands, then reads the openPMD output back.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np

if __name__ == "__main__" and __package__ is None:      # pragma: no cover
    raise SystemExit(
        "\nThis file is a module of the `hvi_emp` package, not a script.\n"
        "Run it as a module from the project root:\n\n"
        "    python -m hvi_emp.solvers.picongpu_stage3\n\n"
        "or import what you need:\n\n"
        "    from hvi_emp.solvers.picongpu_stage3 import PIConGPUConfig\n")

from ..constants import AMU, C_LIGHT, E_CHARGE, K_PER_EV, M_ELECTRON
from ..expansion import ExpansionResult
from ..ionization import debye_length, plasma_frequency

#: float32 machine epsilon.  A Lorentz factor closer to 1 than this is
#: indistinguishable from unity in single precision, which is PIConGPU's
#: default build.
FLOAT32_EPS = 1.19209290e-7

#: Compute capabilities worth naming, for the pic-build backend string.
#: Blackwell (sm_120) needs CUDA >= 12.8 to target at all; earlier toolkits
#: fail with "nvcc fatal: Unsupported gpu architecture".
CUDA_ARCH = {"V100": "70", "A100": "80", "A30": "80", "A40": "86",
             "RTX3090": "86", "RTX4090": "89", "L40": "89", "L40S": "89",
             "RTX6000Ada": "89", "H100": "90", "H200": "90",
             "B200": "100", "GB200": "100",
             "RTX5070Ti": "120", "RTX5080": "120", "RTX5090": "120",
             "RTXPRO6000": "120"}

#: Approximate device memory [GB], used to size the particle-count warning.
GPU_MEMORY_GB = {"V100": 32, "A100": 80, "A30": 24, "A40": 48,
                 "RTX3090": 24, "RTX4090": 24, "L40": 48, "L40S": 48,
                 "RTX6000Ada": 48, "H100": 80, "H200": 141,
                 "B200": 180, "GB200": 186,
                 "RTX5070Ti": 16, "RTX5080": 16, "RTX5090": 32,
                 "RTXPRO6000": 96}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PIConGPUConfig:
    """EMP problem for PIConGPU, in SI units throughout.

    PIConGPU's .param files take SI quantities and normalise internally, so
    unlike the MHD bridges there is no unit bookkeeping to get wrong here.

    Parameters
    ----------
    n_e0 : float
        Peak electron density [m^-3].
    T_eV : float
        Electron temperature [eV].
    sigma : tuple of 3 floats
        Gaussian plume 1/e half-widths along x, y, z [m].  The Anisimov
        ellipsoid maps straight onto this.
    v_drift : float
        Ion bulk expansion speed [m/s].
    drift_axis : tuple of 3 floats
        Direction of the bulk drift; normalised internally.  PIConGPU's Drift
        manipulator assigns a single direction per species, so this bridge
        models a directed plume rather than an isotropic one -- see
        ``notes``.
    m_ion : float
        Ion mass [kg].
    Zbar : float
        Mean ion charge state.
    electron_drift_ratio : float, optional
        Electron bulk speed as a multiple of the ion speed.  Default is
        sqrt(m_i / Z m_e), the Fletcher & Close (2017) assumption that the
        electrons carry the same kinetic energy per particle; this is the
        assumption the PIC run exists to test, so it is exposed.
    thermal_ratio : float
        Electron thermal speed as a multiple of the electron drift, giving
        Fletcher & Close's "cold" (0.1) and "warm" (10) cases.
    probe_distance : float
        Sensor distance along the drift axis [m]. 0.30 m is Close et al.
    cells_per_debye : float
        Target cells per Debye length. PIConGPU's FDTD heats numerically when
        this drops below ~1.
    max_cells : int
        Cap on cells per dimension, so a request that cannot be resolved
        produces an honest warning instead of an impossible grid.
    particles_per_cell : int
    gpu_memory_GB : float
        Device memory of the target GPU, used to size the memory warning.
        ``GPU_MEMORY_GB`` has values for common cards; an RTX 5090 is 32.
    dim : int
        2 or 3.  2D3V halves the cost and keeps all three velocity components.
    """
    n_e0: float
    T_eV: float
    sigma: tuple = (1.0e-3, 1.0e-3, 1.0e-3)
    v_drift: float = 2.0e4
    drift_axis: tuple = (0.0, 1.0, 0.0)
    m_ion: float = 27.0 * AMU
    Zbar: float = 1.0
    electron_drift_ratio: float | None = None
    thermal_ratio: float = 0.1
    probe_distance: float = 0.30
    cells_per_debye: float = 1.0
    max_cells: int = 2048
    particles_per_cell: int = 8
    gpu_memory_GB: float = 24.0
    dim: int = 3
    n_outputs: int = 100
    notes: list = field(default_factory=list)

    def __post_init__(self):
        if self.dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {self.dim}")
        if self.n_e0 <= 0 or self.T_eV <= 0:
            raise ValueError("n_e0 and T_eV must be positive")
        if self.electron_drift_ratio is None:
            self.electron_drift_ratio = float(
                np.sqrt(self.m_ion / (max(self.Zbar, 1e-3) * M_ELECTRON)))

    # -- derived plasma quantities ----------------------------------------
    @property
    def lambda_debye(self) -> float:
        return debye_length(self.n_e0, self.T_eV * K_PER_EV)

    @property
    def omega_pe(self) -> float:
        return plasma_frequency(self.n_e0)

    @property
    def v_electron_drift(self) -> float:
        return self.v_drift * self.electron_drift_ratio

    @property
    def v_thermal_electron(self) -> float:
        return self.thermal_ratio * self.v_electron_drift

    def _gamma(self, v: float) -> float:
        beta = min(v / C_LIGHT, 0.999999)
        return float(1.0 / np.sqrt(1.0 - beta * beta))

    @property
    def gamma_ion(self) -> float:
        return self._gamma(self.v_drift)

    @property
    def gamma_electron(self) -> float:
        return self._gamma(self.v_electron_drift)

    @property
    def temperature_keV(self) -> float:
        """Electron temperature in PIConGPU's units for the manipulator.

        The Temperature manipulator takes keV, and takes it as the thermal
        energy that sets the velocity spread.  We derive it from the requested
        thermal_ratio rather than from T_eV, because the cold/warm cases are
        defined relative to the drift, not to the Stage-2 temperature.
        """
        return float(0.5 * M_ELECTRON * self.v_thermal_electron**2
                     / E_CHARGE / 1000.0)

    # -- grid --------------------------------------------------------------
    def grid(self) -> dict:
        """Cell size, timestep and extent, with an honest resolution verdict.

        The cell size wants to resolve the Debye length; the timestep is then
        set by the FDTD Courant condition, and separately must resolve the
        plasma period.  When the domain needed to reach the probe makes that
        unaffordable, we cap the cell count and say so rather than silently
        returning a grid that will numerically heat.
        """
        lam_D = self.lambda_debye
        dx_debye = lam_D / self.cells_per_debye

        # Domain must hold several plume sigmas and reach the probe.
        extent = max(6.0 * max(self.sigma), 1.2 * self.probe_distance)
        n_ideal = extent / dx_debye

        debye_resolved = n_ideal <= self.max_cells
        n_cells = int(min(np.ceil(n_ideal), self.max_cells))
        n_cells = max(n_cells, 32)
        # round up to a multiple of 16: PIConGPU decomposes into supercells
        n_cells = int(np.ceil(n_cells / 16.0) * 16)
        dx = extent / n_cells

        # Yee/FDTD Courant limit
        ndim_factor = np.sqrt(float(self.dim))
        dt_courant = 0.995 * dx / (C_LIGHT * ndim_factor)
        # plasma period must also be resolved
        dt_plasma = 0.2 / self.omega_pe
        dt = min(dt_courant, dt_plasma)

        # run long enough for light to cross to the probe and back
        t_end = 2.5 * self.probe_distance / C_LIGHT
        steps = int(np.ceil(t_end / dt))

        cells_total = n_cells**self.dim
        return {
            "dx": dx, "n_cells": n_cells, "extent": extent,
            "dt": dt, "dt_courant": dt_courant, "dt_plasma": dt_plasma,
            "steps": steps, "t_end": t_end,
            "lambda_debye": lam_D, "cells_per_debye_actual": lam_D / dx,
            "debye_resolved": bool(debye_resolved),
            "omega_pe": self.omega_pe,
            "cells_total": cells_total,
            "macroparticles": cells_total * self.particles_per_cell * 2,
        }

    def cost_estimate(self, particle_updates_per_second: float = 1.0e9)->dict:
        """Wall clock on one GPU.

        1e9 particle-pushes/s is a conservative single modern NVIDIA GPU
        figure for PIConGPU with cubic shapes; it scales close to linearly
        across GPUs.  Sizing, not a promise.
        """
        g = self.grid()
        pushes = g["macroparticles"] * g["steps"]
        seconds = pushes / particle_updates_per_second
        return {"particle_pushes": pushes, "seconds": seconds,
                "hours": seconds / 3600.0,
                "gpu_memory_GB": g["macroparticles"] * 60 / 1e9,
                "steps": g["steps"]}

    # -- construction ------------------------------------------------------
    @classmethod
    def from_expansion(cls, exp: ExpansionResult,
                       at_frequency: float | None = 916e6,
                       **overrides) -> "PIConGPUConfig":
        """Initial condition from a Stage-2 epoch.

        Parameters
        ----------
        at_frequency : float, optional
            Initialise at the epoch when the peak plasma frequency has fallen
            to this value -- the late plume, which is both affordable and the
            epoch the narrowband measurements actually sample.  Pass None to
            use the collisional -> collisionless transition instead, which is
            where the mechanism physically switches on but is usually out of
            computational reach; ``grid()`` will say so.
        """
        if at_frequency is not None:
            from ..emp import resonant_density
            n_target = resonant_density(at_frequency)
            k = int(np.argmin(np.abs(np.log(np.maximum(exp.n_e, 1e-300))
                                     - np.log(n_target))))
        elif exp.transition:
            k = int(exp.transition["index"])
        else:
            raise ValueError("no collisional->collisionless transition in "
                             "this expansion history; pass at_frequency=")

        sr, sz = float(exp.R_r[k]), float(exp.R_z[k])
        kw = dict(
            n_e0=float(exp.n_e[k]),
            T_eV=float(exp.T_eV[k]),
            sigma=(sr, sz, sr),          # y is the drift/expansion axis
            drift_axis=(0.0, 1.0, 0.0),
            v_drift=float(np.hypot(exp.v_r[k], exp.v_z[k])),
            m_ion=exp.material.m_atom,
            Zbar=max(float(exp.Zbar[k]), 0.05),
        )
        kw.update(overrides)
        cfg = cls(**kw)
        cfg._audit()
        return cfg

    # -- self-audit --------------------------------------------------------
    def _audit(self) -> list:
        g = self.grid()
        self.notes = []
        if not g["debye_resolved"]:
            self.notes.append(
                f"Debye length {g['lambda_debye']:.2e} m needs "
                f"{g['extent'] / (g['lambda_debye'] / self.cells_per_debye):.3g} "
                f"cells to span the domain but max_cells is {self.max_cells}; "
                f"actual resolution is {g['cells_per_debye_actual']:.3g} cells "
                "per Debye length. Plain Yee will heat numerically. Treat the "
                "run as qualitative, shrink probe_distance, or use dim=2.")
        if self.gamma_ion - 1.0 < FLOAT32_EPS:
            self.notes.append(
                f"ion Lorentz factor is 1 + {self.gamma_ion - 1.0:.2e}, below "
                f"float32 epsilon ({FLOAT32_EPS:.1e}). PIConGPU builds in "
                "single precision by default, so the ion drift can be lost "
                "entirely. Build with "
                "-c '-DPRECISION_PIC=precision64Bit', or accept that only "
                "the electron drift is represented (which is the dominant "
                "term in the charge separation anyway).")
        if self.gamma_electron - 1.0 < FLOAT32_EPS:
            self.notes.append(
                f"electron Lorentz factor is 1 + "
                f"{self.gamma_electron - 1.0:.2e}, below float32 epsilon: "
                "this run needs a double-precision build to mean anything.")
        cost = self.cost_estimate()
        if cost["hours"] > 48:
            self.notes.append(
                f"estimated {cost['hours']:.0f} GPU-hours. Reduce "
                "probe_distance, use dim=2, or lower particles_per_cell.")
        if cost["gpu_memory_GB"] > 0.8 * self.gpu_memory_GB:
            n_dev = int(np.ceil(cost["gpu_memory_GB"]
                                / (0.8 * self.gpu_memory_GB)))
            self.notes.append(
                f"~{cost['gpu_memory_GB']:.0f} GB of particle data against a "
                f"{self.gpu_memory_GB:.0f} GB device. Either decompose over "
                f"~{n_dev} devices in the .cfg, drop to dim=2, or reduce "
                f"particles_per_cell from {self.particles_per_cell}. "
                "(Set gpu_memory_GB to match your card.)")
        if self.dim == 2:
            self.notes.append(
                "2D3V: fields are those of infinite line charges, so absolute "
                "amplitudes at the probe are not directly comparable with the "
                "3D dipole formula in emp.py. Use it for scalings and for the "
                "cold/warm comparison, not for absolute V/m.")
        return self.notes


# ---------------------------------------------------------------------------
# .param file generation
# ---------------------------------------------------------------------------

_HEADER = """/* Generated by hvi_emp.solvers.picongpu_stage3
 *
 * {what}
 *
 * PIConGPU compiles .param files into the binary: re-run pic-build after
 * any change here.
 */

#pragma once
"""


def write_simulation_param(cfg: PIConGPUConfig, path: str | None = None)->str:
    """Cell size, timestep and base density.

    In PIConGPU releases before 0.8 this file is named ``grid.param``; the
    contents are the same.  ``pic-create`` will tell you which one your tree
    expects.
    """
    g = cfg.grid()
    text = _HEADER.format(what="Cell size, timestep and reference density.")
    text += f"""
namespace picongpu
{{
    namespace SI
    {{
        /** Duration of one timestep
         *
         *  Courant limit for {cfg.dim}D FDTD is {g['dt_courant']:.4e} s;
         *  resolving the plasma period (omega_pe = {g['omega_pe']:.4e} rad/s)
         *  needs {g['dt_plasma']:.4e} s. We take the smaller.
         *  unit: seconds */
        constexpr float_64 DELTA_T_SI = {g['dt']:.8e};

        /** Cell size, cubic.
         *
         *  Debye length is {g['lambda_debye']:.4e} m, so this grid gives
         *  {g['cells_per_debye_actual']:.3g} cells per Debye length
         *  ({'resolved' if g['debye_resolved'] else
             'UNDER-RESOLVED -- expect numerical heating'}).
         *  unit: meter */
        constexpr float_64 CELL_WIDTH_SI  = {g['dx']:.8e};
        constexpr float_64 CELL_HEIGHT_SI = {g['dx']:.8e};
        constexpr float_64 CELL_DEPTH_SI  = {g['dx']:.8e};

        /** Peak electron density of the plume, from Stage 2.
         *  unit: ELEMENTS/m^3 */
        constexpr float_64 BASE_DENSITY_SI = {cfg.n_e0:.8e};
    }} // namespace SI

    constexpr uint32_t TYPICAL_PARTICLES_PER_CELL = {cfg.particles_per_cell}u;
}} // namespace picongpu
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def write_density_param(cfg: PIConGPUConfig, path: str | None = None) -> str:
    """Anisotropic Gaussian plume, plus the probe lattice."""
    g = cfg.grid()
    c = 0.5 * g["extent"]
    sx, sy, sz = cfg.sigma
    # Place the plume centre off-centre along the drift axis so the pulse has
    # room to propagate towards the probe.
    ax = np.asarray(cfg.drift_axis, dtype=float)
    ax = ax / max(np.linalg.norm(ax), 1e-30)
    centre = np.array([c, c, c]) - ax * 0.35 * g["extent"]

    text = _HEADER.format(
        what="Plume density profile (anisotropic Gaussian) and probe lattice.")
    text += f"""
#include "picongpu/particles/densityProfiles/profiles.def"

namespace picongpu
{{
    namespace densityProfiles
    {{
        /** Anisimov ellipsoidal plume from Stage 2.
         *
         *  density = BASE_DENSITY_SI * exp[ -0.5 * (|r - centre| / sigma)^2 ]
         *  with a per-axis sigma, which is exactly the self-similar
         *  ellipsoid the expansion ODE integrates.
         */
        struct PlumeParam
        {{
            static constexpr float_X gasFactor = -0.5;
            static constexpr float_X gasPower = 2.0;

            //! no vacuum reservation: there is no laser in this setup
            static constexpr uint32_t vacuumCellsY = 0;

            //! plume centre, offset upstream of the probe
            //  unit: meter
            static constexpr floatD_64 center_SI = float3_64(
                {centre[0]:.8e}, {centre[1]:.8e}, {centre[2]:.8e}
            ).shrink<simDim>();

            //! 1/e half-widths along x, y, z
            //  unit: meter
            static constexpr floatD_64 sigma_SI = float3_64(
                {sx:.8e}, {sy:.8e}, {sz:.8e}
            ).shrink<simDim>();
        }};

        using Plume = GaussianCloudImpl<PlumeParam>;

        /** Probe lattice: one probe particle every Nth cell.
         *
         *  Probes are neutral and non-interacting; they exist only to record
         *  E and B at their position, which is the simulated equivalent of
         *  the patch antenna in Close et al. (2013).
         */
        using ProbeLattice = EveryNthCellImpl<mCT::UInt32<{max(g['n_cells'] // 32, 1)},
                                                          {max(g['n_cells'] // 32, 1)},
                                                          {max(g['n_cells'] // 32, 1)}>>;
    }} // namespace densityProfiles
}} // namespace picongpu
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def write_particle_param(cfg: PIConGPUConfig, path: str | None = None) -> str:
    """Drift and temperature manipulators, and in-cell start positions."""
    ax = np.asarray(cfg.drift_axis, dtype=float)
    ax = ax / max(np.linalg.norm(ax), 1e-30)
    warn = ""
    if cfg.gamma_ion - 1.0 < FLOAT32_EPS:
        warn = (f"     *\n     * WARNING gamma - 1 = {cfg.gamma_ion - 1.0:.3e} "
                f"is below float32 epsilon\n"
                f"     * ({FLOAT32_EPS:.2e}). Build in double precision or "
                "this drift vanishes:\n"
                "     *   pic-build -c \"-DPRECISION_PIC=precision64Bit\"\n")

    text = _HEADER.format(
        what="Particle drift, temperature and in-cell start position.")
    text += f"""
#include "picongpu/particles/filter/filter.def"
#include "picongpu/particles/manipulators/manipulators.def"
#include "picongpu/particles/startPosition/functors.def"

#include <pmacc/math/operation.hpp>

namespace picongpu
{{
    namespace particles
    {{
        namespace startPosition
        {{
            struct QuietParam
            {{
                using numParticlesPerDimension = mCT::shrinkTo<
                    mCT::Int<{cfg.particles_per_cell // 2 or 1},
                             {cfg.particles_per_cell // 2 or 1}, 1>,
                    simDim>::type;
            }};
            using Quiet = QuietImpl<QuietParam>;

            //! probes sit at the lower-left corner of their cell, one each
            struct OneProbeParameter
            {{
                static constexpr uint32_t numParticlesPerCell = 1u;
                static constexpr auto inCellOffset = float3_X(0., 0., 0.);
            }};
            using OneProbe = OnePositionImpl<OneProbeParameter>;
        }} // namespace startPosition

        constexpr float_X MIN_WEIGHTING = 10.0;

        namespace manipulators
        {{
            /** Ion bulk drift: v = {cfg.v_drift:.4e} m/s
             *  (beta = {cfg.v_drift / C_LIGHT:.4e})
{warn}             */
            struct IonDriftParam
            {{
                static constexpr float_64 gamma = {cfg.gamma_ion:.15f};
                static constexpr auto driftDirection =
                    float3_X({ax[0]:.6f}, {ax[1]:.6f}, {ax[2]:.6f});
            }};
            using AssignIonDrift =
                unary::Drift<IonDriftParam, pmacc::math::operation::Assign>;

            /** Electron bulk drift: {cfg.electron_drift_ratio:.4g} x the ion
             *  drift = {cfg.v_electron_drift:.4e} m/s
             *  (beta = {cfg.v_electron_drift / C_LIGHT:.4e}).
             *
             *  This ratio is sqrt(m_i / Z m_e) -- the Fletcher & Close (2017)
             *  assumption of equal kinetic energy per particle. It is the
             *  hypothesis this simulation exists to test, not a result.
             */
            struct ElectronDriftParam
            {{
                static constexpr float_64 gamma = {cfg.gamma_electron:.15f};
                static constexpr auto driftDirection =
                    float3_X({ax[0]:.6f}, {ax[1]:.6f}, {ax[2]:.6f});
            }};
            using AssignElectronDrift =
                unary::Drift<ElectronDriftParam,
                             pmacc::math::operation::Assign>;

            /** Electron temperature, from thermal_ratio = {cfg.thermal_ratio:g}
             *  ({'cold' if cfg.thermal_ratio < 1 else 'warm'} case:
             *   v_thermal = {cfg.v_thermal_electron:.4e} m/s)
             *  unit: keV
             */
            struct TemperatureParam
            {{
                static constexpr float_64 temperature =
                    {cfg.temperature_keV:.8e};
            }};
            using AddTemperature = unary::Temperature<TemperatureParam>;
        }} // namespace manipulators
    }} // namespace particles
}} // namespace picongpu
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def write_species_definition_param(cfg: PIConGPUConfig,
                                   path: str | None = None) -> str:
    """Electrons, ions and the probe species."""
    A = cfg.m_ion / AMU
    text = _HEADER.format(
        what="Species: plume electrons, plume ions, and field probes.")
    text += f"""
#include "picongpu/particles/Particles.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/DerivedAttributes.def"

#include <pmacc/identifier/alias.hpp>
#include <pmacc/identifier/value_identifier.hpp>
#include <pmacc/meta/String.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>

namespace picongpu
{{
    /* ---- shared flags ------------------------------------------------- */
    using UsedParticleShape = particles::shapes::PCS;    // cubic, low noise
    using UsedField2Particle = FieldToParticleInterpolation<
        UsedParticleShape, AssignedTrilinearInterpolation>;
    using UsedParticlePusher = particles::pusher::Boris;
    using UsedParticleCurrentSolver = currentSolver::Esirkepov<
        UsedParticleShape>;

    value_identifier(float_X, MassRatioIon, {A * AMU / M_ELECTRON:.8e});
    value_identifier(float_X, ChargeRatioIon, {-cfg.Zbar:.8e});

    using ParticleFlagsElectrons = MakeSeq_t<
        particlePusher<UsedParticlePusher>,
        shape<UsedParticleShape>,
        interpolation<UsedField2Particle>,
        current<UsedParticleCurrentSolver>>;

    using ParticleFlagsIons = MakeSeq_t<
        particlePusher<UsedParticlePusher>,
        shape<UsedParticleShape>,
        interpolation<UsedField2Particle>,
        current<UsedParticleCurrentSolver>,
        massRatio<MassRatioIon>,
        chargeRatio<ChargeRatioIon>>;

    /* Probes: no pusher that moves them, no current deposition. They carry
     * probeE and probeB, which is all we read back. */
    using ParticleFlagsProbes = MakeSeq_t<
        particlePusher<particles::pusher::Probe>,
        shape<UsedParticleShape>,
        interpolation<UsedField2Particle>>;

    using ParticleAttributes = MakeSeq_t<position<position_pic>, momentum,
                                         weighting>;

    using PIC_Electrons = Particles<PMACC_CSTRING("e"),
                                    ParticleFlagsElectrons,
                                    ParticleAttributes>;

    using PIC_Ions = Particles<PMACC_CSTRING("i"),
                               ParticleFlagsIons,
                               ParticleAttributes>;

    using Probes = Particles<PMACC_CSTRING("probe"),
                             ParticleFlagsProbes,
                             MakeSeq_t<position<position_pic>, probeB,
                                       probeE>>;

    using VectorAllSpecies = MakeSeq_t<PIC_Electrons, PIC_Ions, Probes>;
}} // namespace picongpu
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def write_species_initialization_param(cfg: PIConGPUConfig,
                                       path: str | None = None) -> str:
    """Init pipeline: quasi-neutral plume, then drifts, then probes."""
    text = _HEADER.format(
        what="Order in which species are created and manipulated.")
    text += """
#include "picongpu/particles/InitFunctors.hpp"

namespace picongpu
{
    namespace particles
    {
        /** The pipeline runs in order.
         *
         *  Electrons are created from the density profile, then ions are
         *  *derived* from them so the two are co-located: the plume starts
         *  quasi-neutral, and every subsequent charge separation is one the
         *  simulation produced rather than one we imposed.
         */
        using InitPipeline = pmacc::mp_list<
            CreateDensity<densityProfiles::Plume, startPosition::Quiet,
                          PIC_Electrons>,
            Derive<PIC_Electrons, PIC_Ions>,
            Manipulate<manipulators::AssignIonDrift, PIC_Ions>,
            Manipulate<manipulators::AssignElectronDrift, PIC_Electrons>,
            Manipulate<manipulators::AddTemperature, PIC_Electrons>,
            CreateDensity<densityProfiles::ProbeLattice,
                          startPosition::OneProbe, Probes>>;
    } // namespace particles
} // namespace picongpu
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def write_file_output_param(cfg: PIConGPUConfig,
                            path: str | None = None) -> str:
    """What goes into the openPMD output."""
    text = _HEADER.format(what="openPMD output selection.")
    text += """
#include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.def"

namespace picongpu
{
    /** Probes carry the answer; the plume species are dumped so the charge
     *  separation can be seen directly. Drop them from FileOutputParticles
     *  if the output volume becomes the problem -- the probes are small.
     */
    using FileOutputParticles = MakeSeq_t<Probes, PIC_Electrons, PIC_Ions>;

    using ChargeDensity_Seq = deriveField::CreateEligible_t<
        VectorAllSpecies, deriveField::derivedAttributes::ChargeDensity>;
    using Density_Seq = deriveField::CreateEligible_t<
        VectorAllSpecies, deriveField::derivedAttributes::Density>;

    using FileOutputFields = MakeSeq_t<FieldE, FieldB, FieldJ,
                                       ChargeDensity_Seq, Density_Seq>;
} // namespace picongpu
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def write_cfg(cfg: PIConGPUConfig, path: str | None = None,
              devices: tuple = (1, 1, 1)) -> str:
    """TBG runtime configuration (no recompile needed to change this)."""
    g = cfg.grid()
    period = max(g["steps"] // max(cfg.n_outputs, 1), 1)
    dx, dy, dz = devices
    if cfg.dim == 2:
        gridsize = f"{g['n_cells']} {g['n_cells']} 1"
    else:
        gridsize = f"{g['n_cells']} {g['n_cells']} {g['n_cells']}"

    text = f"""# PIConGPU runtime configuration
# Generated by hvi_emp.solvers.picongpu_stage3
#
# Runtime options only: changing this file does NOT require pic-build.

#################################
## Section: Required Variables ##
#################################

TBG_wallTime="{max(int(np.ceil(cfg.cost_estimate()['hours'] * 1.5)), 1)}:00:00"

TBG_devices_x={dx}
TBG_devices_y={dy}
TBG_devices_z={dz}

TBG_gridSize="{gridsize}"
TBG_steps="{g['steps']}"

# Absorbing on all sides: the pulse must leave, not wrap around.
TBG_periodic="--periodic 0 0 0"

#################################
## Section: Optional Variables ##
#################################

# The probes are the measurement. Everything else is diagnostics.
TBG_openPMD="--openPMD.period {period} \\
             --openPMD.file simData \\
             --openPMD.ext bp \\
             --openPMD.source 'species_all,fields_all'"

TBG_energy="--fields_energy.period {period} \\
            --e_energy.period {period} --e_energy.filter all \\
            --i_energy.period {period} --i_energy.filter all"

TBG_macroCount="--e_macroParticlesCount.period {period} \\
                --i_macroParticlesCount.period {period}"

TBG_plugins="!TBG_openPMD    \\
             !TBG_energy     \\
             !TBG_macroCount"

#################################
## Section: Program Parameters ##
#################################

TBG_deviceDist="!TBG_devices_x !TBG_devices_y !TBG_devices_z"

TBG_programParams="-d !TBG_deviceDist \\
                   -g !TBG_gridSize   \\
                   -s !TBG_steps      \\
                   !TBG_periodic      \\
                   !TBG_plugins       \\
                   --versionOnce"

TBG_tasks="$(( TBG_devices_x * TBG_devices_y * TBG_devices_z ))"

"$TBG_cfgPath"/submitAction.sh
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


# ---------------------------------------------------------------------------
# Whole input set
# ---------------------------------------------------------------------------

PARAM_FILES = {
    "simulation.param": write_simulation_param,
    "density.param": write_density_param,
    "particle.param": write_particle_param,
    "speciesDefinition.param": write_species_definition_param,
    "speciesInitialization.param": write_species_initialization_param,
    "fileOutput.param": write_file_output_param,
}


def write_input_set(cfg: PIConGPUConfig, directory: str,
                    devices: tuple = (1, 1, 1),
                    gpu: str | None = None) -> dict:
    """Write the .param and .cfg files in PIConGPU's directory layout.

    These are *overlay* files: run ``pic-create`` first to get a complete
    input set with all the defaults, then copy these over the top.  Writing
    only the files that differ from the defaults keeps the diff readable and
    means an upstream change to an unrelated .param does not silently get
    pinned to whatever it looked like when this was generated.
    """
    param_dir = os.path.join(directory, "include", "picongpu", "param")
    cfg_dir = os.path.join(directory, "etc", "picongpu")
    os.makedirs(param_dir, exist_ok=True)
    os.makedirs(cfg_dir, exist_ok=True)

    written = {}
    for name, writer in PARAM_FILES.items():
        p = os.path.join(param_dir, name)
        writer(cfg, p)
        written[name] = p

    ntask = int(np.prod(devices))
    cfg_name = f"{ntask}.cfg"
    written[cfg_name] = os.path.join(cfg_dir, cfg_name)
    write_cfg(cfg, written[cfg_name], devices=devices)

    written["README.txt"] = os.path.join(directory, "README.txt")
    write_run_notes(cfg, written["README.txt"], devices=devices, gpu=gpu)
    return written


def build_command(cfg: PIConGPUConfig, gpu: str | None = None,
                  double_precision: bool | None = None) -> str:
    """The ``pic-build`` line for this configuration."""
    arch = CUDA_ARCH.get(gpu or "", None)
    backend = f"cuda:{arch}" if arch else "cuda"
    if double_precision is None:
        double_precision = cfg.gamma_ion - 1.0 < FLOAT32_EPS
    extra = ' -c "-DPRECISION_PIC=precision64Bit"' if double_precision else ""
    return f'pic-build -b "{backend}"{extra}'


def write_run_notes(cfg: PIConGPUConfig, path: str | None = None,
                    devices: tuple = (1, 1, 1),
                    gpu: str | None = None) -> str:
    """Build, run and interpretation notes."""
    g = cfg.grid()
    cost = cfg.cost_estimate()
    ntask = int(np.prod(devices))
    warn = ("\n".join(f"  ! {w}" for w in cfg.notes) if cfg.notes
            else "  (none)")
    prec = ("double (ion drift needs it)"
            if cfg.gamma_ion - 1.0 < FLOAT32_EPS else "single (default)")

    text = f"""PIConGPU EMP run
================
Generated by hvi_emp.solvers.picongpu_stage3

Install PIConGPU (once)
-----------------------
  git clone https://github.com/ComputationalRadiationPhysics/picongpu.git \\
      $HOME/src/picongpu
  export PICSRC=$HOME/src/picongpu
  export PATH=$PICSRC/bin:$PATH
  export PIC_EXAMPLES=$PICSRC/share/picongpu/examples
  export PIC_BACKEND="cuda"

PIConGPU needs boost, CMake, an MPI, and openPMD-api/ADIOS2 for output. The
project ships a `picongpu.profile` template per system:
https://picongpu.readthedocs.io/en/latest/install/profile.html

Create the input set, then overlay these files
----------------------------------------------
  pic-create $PIC_EXAMPLES/KelvinHelmholtz $HOME/picInputs/hviEMP
  cp -r <this directory>/include $HOME/picInputs/hviEMP/
  cp -r <this directory>/etc     $HOME/picInputs/hviEMP/

KelvinHelmholtz is the closest starting point: two counter-drifting species,
no laser, no moving window. Only the .param files listed below are replaced;
everything else stays at PIConGPU's defaults.

  {chr(10).join('  ' + n for n in sorted(PARAM_FILES))}

Note: in PIConGPU releases before 0.8 `simulation.param` is called
`grid.param`. If pic-build cannot find it, rename it.

Build and run
-------------
  cd $HOME/picInputs/hviEMP
  {build_command(cfg, gpu=gpu)}
  tbg -s bash -c etc/picongpu/{ntask}.cfg \\
      -t etc/picongpu/bash/mpiexec.tpl $SCRATCH/runs/hviEMP_001

Precision: {prec}

Resolution
----------
  Debye length          {g['lambda_debye']:.4e} m
  cell size             {g['dx']:.4e} m
  cells per Debye       {g['cells_per_debye_actual']:.3g}  ({'OK' if g['debye_resolved'] else 'UNDER-RESOLVED'})
  omega_pe              {g['omega_pe']:.4e} rad/s
  timestep              {g['dt']:.4e} s  (Courant {g['dt_courant']:.3e},
                        plasma period {g['dt_plasma']:.3e})
  grid                  {g['n_cells']}^{cfg.dim} over {g['extent']:.4e} m
  steps                 {g['steps']}
  physical time         {g['t_end']:.4e} s

Plasma state handed over from Stage 2
-------------------------------------
  peak n_e              {cfg.n_e0:.4e} m^-3
  T_e                   {cfg.T_eV:.4g} eV
  mean charge Zbar      {cfg.Zbar:.4g}
  ion mass              {cfg.m_ion / AMU:.4g} amu
  plume sigma           {cfg.sigma[0]:.3e}, {cfg.sigma[1]:.3e}, {cfg.sigma[2]:.3e} m
  ion drift             {cfg.v_drift:.4e} m/s
  electron drift        {cfg.v_electron_drift:.4e} m/s
                        ({cfg.electron_drift_ratio:.4g} x ion, the Fletcher &
                         Close assumption under test)
  case                  {'cold' if cfg.thermal_ratio < 1 else 'warm'}
                        (v_thermal / v_drift = {cfg.thermal_ratio:g})

Expected cost
-------------
  {cost['particle_pushes']:.3g} particle pushes
  ~{cost['hours']:.2f} h on one GPU at 1e9 pushes/s
  ~{cost['gpu_memory_GB']:.2f} GB of particle data

Reading the result
------------------
  from hvi_emp.solvers.picongpu_stage3 import load_probe_timeseries
  ts = load_probe_timeseries("$SCRATCH/runs/hviEMP_001/simOutput/simData_%T.bp",
                             position=(x, y, z))
  # -> {{"t", "E", "B", "position"}} comparable with emp.EMPResult

Warnings from the configuration audit
-------------------------------------
{warn}

Caveats
-------
* Probes record the field one timestep behind the last field update
  (PIConGPU pushes particles before solving fields). At {g['dt']:.2e} s that
  is far below any frequency of interest, but it is not zero.
* The drift manipulator assigns one direction to a whole species, so this is
  a directed plume, not a radially expanding shell. For the charge-separation
  mechanism -- which is about electrons outrunning ions along the expansion
  direction -- that is the relevant geometry, but it is not the full
  Anisimov ellipsoid.
* No binary collisions are enabled. At the late-plume densities this bridge
  targets, the plasma is nearly collisionless, which is the regime the
  mechanism needs. If you move to an earlier epoch, switch collisions on.
"""
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


# ---------------------------------------------------------------------------
# Reading output back
# ---------------------------------------------------------------------------

def load_probe_timeseries(pattern: str, position=None,
                          tolerance: float | None = None) -> dict:
    """Read probeE/probeB from PIConGPU's openPMD output.

    Parameters
    ----------
    pattern : str
        openPMD series pattern, e.g.
        ``".../simOutput/simData_%T.bp"``, or a directory containing it.
    position : sequence of 3 floats, optional
        Return the single probe nearest this point [m].  Without it, every
        probe is returned and the arrays gain a leading probe axis.
    tolerance : float, optional
        Fail rather than silently return a distant probe if the nearest one
        is further than this [m].

    Returns
    -------
    dict with ``t`` [s], ``E`` and ``B`` (shape (n_t, 3) for a single probe,
    else (n_probe, n_t, 3)), and ``position``.
    """
    try:
        import openpmd_api as io
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "reading PIConGPU output needs openPMD:\n"
            "    pip install openpmd-api\n"
            "(this is a genuine PyPI package, unlike pywarpx)") from exc

    if os.path.isdir(pattern):
        cands = sorted(glob.glob(os.path.join(pattern, "*.bp"))
                       + glob.glob(os.path.join(pattern, "*.h5")))
        if not cands:
            raise FileNotFoundError(f"no .bp or .h5 series under {pattern}")
        base = os.path.basename(cands[0])
        stem = base.split("_")[0]
        ext = os.path.splitext(base)[1]
        pattern = os.path.join(pattern, f"{stem}_%T{ext}")

    series = io.Series(pattern, io.Access.read_only)

    XYZ = ("x", "y", "z")
    times, Es, Bs, pos = [], [], [], None

    for _, it in series.iterations.items():
        if "probe" not in it.particles:
            raise KeyError(
                "no 'probe' species in the output. Check that "
                "speciesDefinition.param defines Probes and that "
                "fileOutput.param includes it in FileOutputParticles.")
        p = it.particles["probe"]

        # load_chunk() is *lazy*: it hands back a buffer that openPMD only
        # fills on flush().  Everything must be requested first and read
        # after, or the arrays hold whatever was in memory -- which looks
        # like plausible numbers in the wrong order rather than an error.
        E_chunks = [p["probeE"][c].load_chunk() for c in XYZ]
        B_chunks = [p["probeB"][c].load_chunk() for c in XYZ]
        want_pos = pos is None
        if want_pos:
            comps = p["position"]
            offs = p["positionOffset"] if "positionOffset" in p else None
            pos_chunks = [comps[c].load_chunk() for c in XYZ]
            off_chunks = ([offs[c].load_chunk() for c in XYZ]
                          if offs is not None else None)

        series.flush()                       # buffers are valid from here

        E = np.stack([c * (p["probeE"][n].unit_SI or 1.0)
                      for c, n in zip(E_chunks, XYZ)], axis=-1)
        B = np.stack([c * (p["probeB"][n].unit_SI or 1.0)
                      for c, n in zip(B_chunks, XYZ)], axis=-1)
        if want_pos:
            xyz = [c * (comps[n].unit_SI or 1.0)
                   for c, n in zip(pos_chunks, XYZ)]
            if off_chunks is not None:
                xyz = [v + o * (offs[n].unit_SI or 1.0)
                       for v, o, n in zip(xyz, off_chunks, XYZ)]
            pos = np.stack(xyz, axis=-1)

        times.append(it.time * (it.time_unit_SI or 1.0))
        Es.append(E)
        Bs.append(B)

    t = np.asarray(times, dtype=float)
    E = np.asarray(Es, dtype=float)          # (n_t, n_probe, 3)
    B = np.asarray(Bs, dtype=float)
    order = np.argsort(t)
    t, E, B = t[order], E[order], B[order]

    if position is not None:
        d = np.linalg.norm(pos - np.asarray(position, dtype=float), axis=-1)
        i = int(np.argmin(d))
        if tolerance is not None and d[i] > tolerance:
            raise ValueError(
                f"nearest probe is {d[i]:.4g} m from the requested position, "
                f"beyond the {tolerance:.4g} m tolerance. Probe spacing is "
                "set by ProbeLattice in density.param.")
        return {"t": t, "E": E[:, i, :], "B": B[:, i, :],
                "position": pos[i], "distance": float(d[i])}
    return {"t": t, "E": np.moveaxis(E, 0, 1), "B": np.moveaxis(B, 0, 1),
            "position": pos}


def compare_with_reduced_emp(ts: dict, emp_result) -> dict:
    """Peak field and dominant frequency, PIConGPU against reduced Stage 3."""
    E = np.linalg.norm(np.atleast_2d(ts["E"]), axis=-1)
    t = ts["t"]
    peak_pic = float(np.max(E))
    peak_red = float(np.max(np.abs(np.asarray(emp_result.E_field))))

    f_pic = np.nan
    if t.size > 3:
        dt = float(np.median(np.diff(t)))
        if dt > 0:
            spec = np.abs(np.fft.rfft(E - E.mean()))
            freqs = np.fft.rfftfreq(E.size, dt)
            if spec.size > 1:
                f_pic = float(freqs[1 + int(np.argmax(spec[1:]))])

    return {"E_peak_picongpu": peak_pic, "E_peak_reduced": peak_red,
            "ratio": peak_pic / peak_red if peak_red else np.inf,
            "f_peak_picongpu": f_pic}


__all__ = ["PIConGPUConfig", "write_simulation_param", "write_density_param",
           "write_particle_param", "write_species_definition_param",
           "write_species_initialization_param", "write_file_output_param",
           "write_cfg", "write_input_set", "write_run_notes",
           "build_command", "load_probe_timeseries", "compare_with_reduced_emp",
           "PARAM_FILES", "CUDA_ARCH", "FLOAT32_EPS"]


if __name__ == "__main__":                              # pragma: no cover
    from ..pipeline import run_scenario
    sc = run_scenario("Fe", "Al", mass=1e-12, velocity=50e3)
    cfg = PIConGPUConfig.from_expansion(sc.expansion, at_frequency=916e6)
    print(write_run_notes(cfg))
