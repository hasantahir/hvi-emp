"""Stage 2 -- plume expansion into vacuum, ionisation freeze-out, transitions.

Model
-----
**Gas dynamics.**  Adiabatic self-similar expansion of an axisymmetric
ellipsoidal plume, after Anisimov, Bauerle & Luk'yanchuk, J. Appl. Phys. 73,
8337 (1993).  Semi-axes ``(R_r, R_r, R_z)`` with a Gaussian density profile:

    d^2 R_i/dt^2 = P / (rho R_i),      P = n_h k T (1 + Zbar)
    du/dt        = -(P/n_h) d(ln V)/dt,
    u            = 3/2 k T (1 + Zbar) + E_ion(n, T)

The ionisation energy sits *inside* u, so the effective adiabatic index is
computed self-consistently instead of being fixed at 5/3.  This matters: a
recombining plasma returns ionisation energy to the gas and cools much more
slowly than a gamma = 5/3 gas would.  The inversion u -> (T, Zbar) uses the
tabulated Saha solver `ionization.IonisationTable` (~0.5% accurate, ~1e3 x
faster than the direct nested root-find).

**Shell-resolved freeze-out -- and why a single-zone model fails.**  The
recombination time scales as ``tau_rec ~ (alpha3 n_e^2)^-1``, so it varies by
*many* orders of magnitude across the plume's density profile.  The tenuous
outer shells freeze out early, at high charge state; the dense core stays in
Saha equilibrium and recombines almost completely.  A single-zone model
evaluated at the peak density therefore predicts essentially zero surviving
charge -- in flat contradiction with retarding-potential-analyser
measurements.  Here the Gaussian profile is discretised into Lagrangian mass
shells, each of which

  * follows the common temperature history (fast electron thermal conduction),
  * has its own density n_j(t) = w_j n_peak(t),
  * freezes when its own tau_rec exceeds the expansion time,

and the surviving charge is summed over shells.

**Collisional -> collisionless transition.**  nu_ei ~ n_e while
omega_pe ~ n_e^(1/2), so an expanding plasma must cross nu_ei = omega_pe.
Fletcher (NRL/MR/6757--20-10,138, Fig. 2) identifies this crossing as the
trigger for the Close et al. EMP mechanism: electrons at the plume edge become
free to run ahead of the ions, an ambipolar field builds, and the resulting
coherent oscillation radiates.  The crossing is located per shell as well as
for the plume as a whole, because *where* it happens sets the emitted
frequency.

**Validity.**  Self-similarity assumes the density profile *shape* is
preserved -- well satisfied for free expansion into vacuum after a few initial
radii, and the standard treatment for laser-ablation plumes.  It does not
resolve internal shocks, returning rarefactions, or a background gas; see
`stopping_distance` for whether a laboratory chamber is a vacuum for the plume.

References
----------
Anisimov, Bauerle & Luk'yanchuk, J. Appl. Phys. 73, 8337 (1993).
Gurevich, Pariiskaya & Pitaevskii, Sov. Phys. JETP 22, 449 (1966).
Mora, Phys. Rev. Lett. 90, 185002 (2003).
Fletcher & Close, Phys. Plasmas 24, 053102 (2017).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .constants import E_CHARGE, EV, K_B
from .impact import ImpactResult
from .ionization import (collision_frequency_ei, coupling_parameter,
                         debye_length, get_table, plasma_frequency,
                         recombination_time)
from .radiation import cooling_rate
from .materials import Material


# ---------------------------------------------------------------------------
# Gaussian shell discretisation
# ---------------------------------------------------------------------------

#: The plume is a HEMISPHERE, not a sphere, and this is the factor that
#: makes the density reflect it.
#:
#: Stage 1 computes the initial cloud radius as ``r_plume0 = (3V/2pi)^(1/3)``
#: -- explicitly the radius of a *hemisphere* of volume V, because the vapour
#: is released from a crater into the half-space above a solid target. Stage 2
#: then spread that same mass over a *full* 3-D Gaussian, which put ~50% of
#: the plume at z < 0 (inside the target) for the first nanosecond and
#: under-predicted the density everywhere by a factor of two.
#:
#: The cloud stays a half-Gaussian for the whole run, not just while it
#: touches the surface. A radially expanding cloud has no velocity component
#: through the plane containing its centre, so no material ever crosses into
#: the lower half: in the co-moving frame the flat face is preserved and
#: simply recedes with the target. Equivalently, the real half evolves
#: exactly as half of a freely expanding sphere carrying twice the mass.
#:
#: So the correct peak density for the real vapour mass M is that of a full
#: Gaussian carrying 2M. Effect: peak n_e doubles, which moves it toward the
#: ~1e27 m^-3 that hydrocodes report. It has no free parameter and no
#: measurable quantity was tuned to obtain it.
HEMISPHERE_FACTOR = 2.0


def gaussian_shells(n_shell: int = 24, xi_max: float = 3.5
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Mass fractions and density weights of a Gaussian plume.

    For ``n(xi) = n_peak exp(-xi^2/2)`` in scaled ellipsoidal coordinates the
    mass in [xi, xi+dxi] is proportional to ``xi^2 exp(-xi^2/2) dxi``.

    Returns
    -------
    x : mass fraction of each shell (sums to 1)
    w : density weight n_shell / n_peak = exp(-xi^2/2)
    """
    edges = np.linspace(0.0, xi_max, n_shell + 1)
    xi = 0.5 * (edges[:-1] + edges[1:])
    w = np.exp(-0.5 * xi**2)
    dm = xi**2 * w * np.diff(edges)
    x = dm / dm.sum()
    return x, w


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ExpansionResult:
    """Time history of the expanding plume."""
    t: np.ndarray
    R_r: np.ndarray
    R_z: np.ndarray
    v_r: np.ndarray
    v_z: np.ndarray
    z_cm: np.ndarray
    T: np.ndarray
    T_eV: np.ndarray
    Zbar: np.ndarray             # mass-weighted, freeze-out applied
    Zbar_eq: np.ndarray          # local Saha equilibrium at peak density
    n_h: np.ndarray              # peak heavy density
    n_e: np.ndarray              # peak electron density (frozen composition)
    omega_pe: np.ndarray
    nu_ei: np.ndarray
    lambda_De: np.ndarray
    Gamma_coupling: np.ndarray
    Q_free: np.ndarray           # C, surviving free charge vs time
    material: Material
    N_heavy: float
    shells: dict = field(default_factory=dict)
    freeze: dict = field(default_factory=dict)
    transition: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def Q_final(self) -> float:
        """Free positive charge surviving to the end of the run [C]."""
        return float(self.Q_free[-1])

    def summary(self) -> str:
        fz, tr = self.freeze, self.transition
        lines = [
            f"Plume expansion ({self.material.name}), "
            f"N_heavy = {self.N_heavy:.3e}",
            f"  integration window      {self.t[0]:.2e} - {self.t[-1]:.2e} s",
            f"  final semi-axes         R_r {self.R_r[-1]*1e3:.3f} mm, "
            f"R_z {self.R_z[-1]*1e3:.3f} mm",
            f"  asymptotic v_z          {self.v_z[-1]/1e3:.2f} km/s",
            f"  frozen charge Q         {self.Q_final:.3e} C "
            f"(Zbar_eff = {self.Zbar[-1]:.3f})",
        ]
        if fz:
            lines += [
                "  mass-weighted freeze-out:",
                f"    t                     {fz['t']:.3e} s",
                f"    n_e (peak)            {fz['n_e']:.3e} m^-3",
                f"    T_e                   {fz['T_eV']:.3f} eV",
            ]
        if tr:
            lines += [
                "  collisional -> collisionless transition (peak density):",
                f"    t                     {tr['t']:.3e} s",
                f"    plume scale           {tr['R']*1e3:.4f} mm",
                f"    n_e                   {tr['n_e']:.3e} m^-3",
                f"    T_e                   {tr['T_eV']:.3f} eV",
                f"    f_pe                  {tr['f_pe']:.3e} Hz",
                f"    lambda_De             {tr['lambda_De']*1e6:.4f} um",
            ]
        else:
            lines.append("  collisionless transition: not reached in window")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def simulate_expansion(impact: ImpactResult,
                       t_end: float | None = None,
                       n_out: int = 400,
                       n_shell: int = 24,
                       aspect0: float = 0.5,
                       include_bulk_drift: bool = True,
                       radiative_cooling: bool = True,
                       rtol: float = 1e-8,
                       atol: float = 1e-14) -> ExpansionResult:
    """Integrate the plume from the Stage-1 initial condition.

    Parameters
    ----------
    impact : ImpactResult
    t_end : float, optional
        End time [s].  Default 2000 * R0 / v_expansion -- long enough for the
        plume to reach centimetre scale and pass the collisionless transition.
    n_shell : int
        Number of Lagrangian shells in the Gaussian profile.
    aspect0 : float
        Initial R_z / R_r of the plume seed.

        **This parameter is calibrated, not derived, and it matters.** It is
        the only handle in the model on the plume's angular distribution, and
        it moves the peak electron density by ~10x across its plausible
        range.

        aspect0 = 1.0 is a FIXED POINT: with R_r = R_z the two momentum
        equations are identical, the ratio stays exactly 1.000 for all time,
        and the plume is a perfect sphere. In the Anisimov model the
        asymptotic angular distribution is

            N(theta) ~ [k^2 sin^2(theta) + cos^2(theta)]^(-3/2),
            k = Rdot_z / Rdot_r

        so a sphere (k = 1) radiates ISOTROPICALLY -- N is flat at every
        angle. Laboratory measurements of dust-impact plasma plumes report a
        **cosine law**, which is forward-peaked, so k = 1 is excluded by
        experiment rather than merely unlikely.

        Least-squares fitting that cosine out to 80 degrees gives k = 1.33,
        which this model reaches from aspect0 = 0.5. That is the default.

        Be clear about what this is: **one parameter fitted to one measured
        distribution.** It is not derived from the shock solution. In fact
        this framework's own Stage 1 disagrees with it -- `peak_pressure_
        profile` depends on distance alone, so every iso-pressure surface,
        including the vaporisation boundary, is a hemisphere with aspect 1.
        Real plumes are forward-peaked because free-surface rarefaction vents
        material along the normal, and that physics is absent here. aspect0
        is where it gets encoded.

        Set it to 1.0 to recover the isotropic behaviour, or sweep it: it is
        exposed as `scenario.aspect0` in the config tree.
    include_bulk_drift : bool
        Add a centre-of-mass velocity along the surface normal equal to
        ``impact.v_expansion``; this is what carries the plume away from the
        crater rather than merely inflating it in place.
    radiative_cooling : bool
        Include bremsstrahlung and radiative-recombination losses in the
        energy equation (`radiation.cooling_rate`). On by default. For impact
        plumes it changes almost nothing -- the radiative cooling time is ~50x
        the expansion time at formation and diverges from there -- but the
        term is cheap and its absence was a documented gap. Set False to
        recover the strictly adiabatic result.
    """
    mat = impact.material_mix
    if impact.m_vapour <= 0:
        raise ValueError(
            "No vapour produced by this impact -- Stage 2 has nothing to "
            "expand. Increase the impact velocity or check the materials.")

    table = get_table(mat)
    N_heavy = impact.m_vapour / mat.m_atom
    R0 = impact.r_plume0
    Rz0 = aspect0 * R0
    st0 = impact.plasma
    u0 = 1.5 * K_B * st0.T * (1.0 + st0.Zbar) + st0.E_ionisation
    v_cm = impact.v_expansion if include_bulk_drift else 0.0
    if t_end is None:
        t_end = 2000.0 * R0 / max(impact.v_expansion, 1.0)

    x_shell, w_shell = gaussian_shells(n_shell)
    # Peak density of a Gaussian with these semi-axes and N particles.
    peak_norm = HEMISPHERE_FACTOR * (2.0 * np.pi) ** -1.5

    def n_peak(Rr, Rz):
        return N_heavy * peak_norm / (Rr * Rr * Rz)

    def rhs(t, y):
        Rr, Rz, vr, vz, u, _ = y
        Rr, Rz, u = max(Rr, 1e-12), max(Rz, 1e-12), max(u, 1e-30)
        nh = n_peak(Rr, Rz)
        T, Z = table.state(nh, u)
        P = nh * K_B * T * (1.0 + Z)
        rho = nh * mat.m_atom
        dlnV = 2.0 * vr / Rr + vz / Rz
        du = -(P / nh) * dlnV
        if radiative_cooling:
            # Radiated power per heavy particle. Small for impact plumes --
            # ~50x slower than expansion -- but it is cheap and it makes the
            # model correct at parameters where it would not be.
            L = cooling_rate(nh, T * K_B / EV, Z,
                             length=min(Rr, Rz), A=mat.A)["total"]
            du -= L / nh
        return [vr, vz, P / (rho * Rr), P / (rho * Rz), du, v_cm]

    y0 = [R0, Rz0, 0.0, 0.0, u0, 0.0]
    t_eval = np.geomspace(t_end * 1e-7, t_end, n_out)
    sol = solve_ivp(rhs, (0.0, t_end), y0, t_eval=t_eval, method="LSODA",
                    rtol=rtol, atol=atol, max_step=t_end / 100)
    if not sol.success:
        raise RuntimeError(f"Expansion integration failed: {sol.message}")

    t = sol.t
    Rr = np.maximum(sol.y[0], 1e-12)
    Rz = np.maximum(sol.y[1], 1e-12)
    vr, vz, u, zcm = sol.y[2], sol.y[3], np.maximum(sol.y[4], 1e-40), sol.y[5]
    nh_pk = n_peak(Rr, Rz)

    T = np.array([table.temperature(n, uu) for n, uu in zip(nh_pk, u)])
    T = np.maximum(T, 1.0)
    T_eV = T * K_B / EV
    Zbar_eq = np.array([table.Zbar(n, TT) for n, TT in zip(nh_pk, T)])

    # --- shell-resolved freeze-out --------------------------------------
    nsh = len(x_shell)
    Z_sh = np.zeros((len(t), nsh))
    frozen = np.zeros(nsh, dtype=bool)
    Z_frz = np.zeros(nsh)
    t_frz = np.full(nsh, np.nan)
    n_frz = np.zeros(nsh)

    dlnn = np.gradient(np.log(nh_pk), t)
    tau_exp_arr = np.where(np.abs(dlnn) > 0, 1.0 / np.abs(dlnn), np.inf)

    for i in range(len(t)):
        n_sh_i = w_shell * nh_pk[i]
        for j in range(nsh):
            if frozen[j]:
                Z_sh[i, j] = Z_frz[j]
                continue
            Zeq = table.Zbar(n_sh_i[j], T[i])
            Z_sh[i, j] = Zeq
            ne = max(Zeq * n_sh_i[j], 1e-30)
            trec = recombination_time(ne, T_eV[i], max(Zeq, 1.0))
            if i > 0 and trec > tau_exp_arr[i]:
                frozen[j] = True
                Z_frz[j] = Zeq
                t_frz[j] = t[i]
                n_frz[j] = n_sh_i[j]

    Zbar = Z_sh @ x_shell                     # mass-weighted mean charge
    Q_free = E_CHARGE * N_heavy * Zbar
    # Peak electron density uses the innermost shell's (frozen) charge state.
    n_e = Z_sh[:, 0] * nh_pk

    w_pe = np.array([plasma_frequency(x) for x in n_e])
    nu = np.array([collision_frequency_ei(a, b, max(c, 1.0))
                   for a, b, c in zip(n_e, T_eV, Zbar)])
    lam = np.array([debye_length(a, b) for a, b in zip(n_e, T_eV)])
    Gam = np.array([coupling_parameter(a, b) for a, b in zip(n_e, T_eV)])

    freeze_info: dict = {}
    if np.isfinite(t_frz).any():
        # mass-weighted mean freeze time
        good = np.isfinite(t_frz)
        tf = float(np.average(t_frz[good], weights=x_shell[good]))
        k = int(np.argmin(np.abs(t - tf)))
        freeze_info = {
            "t": tf, "n_e": float(n_e[k]), "T_eV": float(T_eV[k]),
            "Zbar": float(Zbar[k]),
            "shell_times": t_frz, "shell_Z": Z_frz,
            "shell_n": n_frz, "shell_mass_fraction": x_shell,
            "shell_density_weight": w_shell,
        }

    transition = _find_transition(t, nu, w_pe, Rr, Rz, n_e, nh_pk, T_eV,
                                  Zbar, lam, vr, vz, v_cm)

    return ExpansionResult(
        t=t, R_r=Rr, R_z=Rz, v_r=vr, v_z=vz + v_cm, z_cm=zcm,
        T=T, T_eV=T_eV, Zbar=Zbar, Zbar_eq=Zbar_eq,
        n_h=nh_pk, n_e=n_e, omega_pe=w_pe, nu_ei=nu, lambda_De=lam,
        Gamma_coupling=Gam, Q_free=Q_free, material=mat, N_heavy=N_heavy,
        shells={"mass_fraction": x_shell, "density_weight": w_shell,
                "Z": Z_sh, "frozen": frozen, "t_freeze": t_frz},
        freeze=freeze_info, transition=transition,
        diagnostics={
            "u": u, "u0": u0, "R0": R0, "v_cm": v_cm, "aspect0": aspect0,
            "tau_exp": tau_exp_arr,
            "energy": _energy_check(vr, vz, u, mat, N_heavy),
        },
    )


def _find_transition(t, nu, w_pe, Rr, Rz, n_e, n_h, T_eV, Zbar, lam,
                     vr, vz, v_cm) -> dict:
    """Locate the first nu_ei = omega_pe crossing (collisional -> kinetic)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(w_pe > 0, nu / np.maximum(w_pe, 1e-300), np.inf)
    idx = np.where(np.isfinite(ratio) & (ratio < 1.0))[0]
    if idx.size == 0:
        return {}
    i = int(idx[0])
    if i == 0:
        t_x = float(t[0])
    else:
        j = i - 1
        lr0, lr1 = np.log(ratio[j]), np.log(max(ratio[i], 1e-300))
        f = lr0 / (lr0 - lr1) if lr0 != lr1 else 1.0
        t_x = float(np.exp(np.log(t[j]) + f * (np.log(t[i]) - np.log(t[j]))))

    def ip(arr):
        return float(np.interp(t_x, t, arr))

    w = ip(w_pe)
    return {
        "t": t_x, "index": i,
        "R": float(np.sqrt(ip(Rr) * ip(Rz))), "R_r": ip(Rr), "R_z": ip(Rz),
        "n_e": ip(n_e), "n_h": ip(n_h), "T_eV": ip(T_eV), "Zbar": ip(Zbar),
        "omega_pe": w, "f_pe": w / (2.0 * np.pi), "nu_ei": ip(nu),
        "lambda_De": ip(lam), "v_r": ip(vr), "v_z": ip(vz) + v_cm,
    }


def _energy_check(vr, vz, u, mat: Material, N: float) -> dict:
    """Relative drift of plume kinetic + internal energy (ODE consistency)."""
    M = N * mat.m_atom
    E = 0.5 * M * (2.0 * vr**2 + vz**2) + N * u
    return {"E_total": E,
            "relative_drift": float((E[-1] - E[0]) / max(abs(E[0]), 1e-300))}


# ---------------------------------------------------------------------------
# Background gas
# ---------------------------------------------------------------------------

def stopping_distance(n_background: float, n_plume0: float, R0: float
                      ) -> float:
    """Distance [m] at which background gas arrests the plume.

    Snowplough estimate: the plume stalls once it has swept up its own mass,
    ``n_plume0 R0^3 = n_background R_stop^3``.

    Use this to check whether a chamber is a vacuum *for the plume*.  Close et
    al. (2013) required < 1e-6 Torr precisely so the expansion stayed
    collisionless; Bianchi et al. (1984) worked at 0.1-6 Torr, where the plume
    is stopped within millimetres -- which is why their ~300 kHz emission is
    usually attributed to a different mechanism (micro-cracking / charge
    relaxation) rather than to coherent plasma oscillation.
    """
    if n_background <= 0:
        return np.inf
    return R0 * (n_plume0 / n_background) ** (1.0 / 3.0)


def torr_to_number_density(p_torr: float, T_K: float = 293.0) -> float:
    """Ideal-gas number density [m^-3] for a pressure in torr."""
    return p_torr * 133.322 / (K_B * T_K)


__all__ = [
    "ExpansionResult", "simulate_expansion", "gaussian_shells",
    "stopping_distance", "torr_to_number_density",
]
