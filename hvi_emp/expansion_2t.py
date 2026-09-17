"""Stage 2, two-temperature with non-equilibrium ionisation.

Why this exists
---------------
The one-temperature model in `expansion.py` computes the ionisation state from
Saha at every step and applies freeze-out afterwards, as a post-process.  Two
measurements say that is the wrong order of operations for this problem:

1. **The charge freezes exactly where electrons and ions decouple.**  For a
   1 pg Fe impact on Al at 50 km/s the bulk of the recombination -- Zbar
   falling from 0.21 to 0.023, which is where essentially all the charge is
   lost -- happens between 12 and 28 ns, and over that same window the ratio
   of the electron-ion equilibration time to the expansion time rises from
   1.06 to 67.  A single temperature is not defensible there.

2. **Three-body recombination heats the electrons.**  Each capture releases
   the binding energy into the electron gas, and the rate goes as T_e^-4.5, so
   the heating strongly suppresses further recombination.  That is a powerful
   negative feedback.  In a one-temperature model the released energy is
   shared with the ions -- which carry ~10^5 times the heat capacity per
   particle at these charge states -- so the electrons stay cold and the
   plasma over-recombines.

The consequence to test is that the frozen charge should be both larger and
*less sensitive to the initial temperature*, which is the same thing as saying
the charge-yield velocity exponent should come down.

The model
---------
Common electron and ion temperatures across the plume (electron thermal
conduction is fast compared with the expansion, which the one-temperature
model already assumes), but they are no longer equal to each other.  Each
Lagrangian mass shell carries its own ionisation state.

    d(3/2 Zbar k T_e)/dt = -(P_e/n_h) dlnV/dt        expansion work
                           - chi dZbar/dt            ionisation energy
                           - Q_ei                    transfer to ions
                           - Lambda_rad/n_h          radiation

    d(3/2 k T_i)/dt      = -(P_i/n_h) dlnV/dt + Q_ei

    dZ_j/dt              = (Z_eq(n_j, T_e) - Z_j) / tau_rec(n_e,j, T_e)

with P_e = Zbar n_h k T_e, P_i = n_h k T_i, and

    Q_ei = 3 (m_e/m_i) nu_ei Zbar k (T_e - T_i)

The ionisation equation is a collisional-radiative *relaxation*: it drives
each shell toward its local Saha value on the recombination timescale, so it
reproduces equilibrium while tau_rec << tau_expansion and freezes smoothly
once tau_rec exceeds it.  That is the same physics the post-hoc freeze-out
approximates, but solved continuously and coupled to T_e -- which is what
makes the recombination-heating feedback available at all.

Limitations
-----------
* One effective ionisation potential per material (`E_ion[0]`), so the energy
  bookkeeping does not resolve stage-by-stage ionisation.
* Electron thermal conduction is assumed infinitely fast (one T_e for the
  whole plume). The shells differ in density and ionisation, not temperature.
* No electron heat flux limiter, no non-Maxwellian tail. The freeze-out
  literature suggests the tail matters at the few-tens-of-percent level.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .constants import E_CHARGE, EV, K_B, M_ELECTRON
from .expansion import (HEMISPHERE_FACTOR, ExpansionResult,
                        _find_transition, gaussian_shells)
from .impact import ImpactResult
from .ionization import (alpha_radiative, alpha_three_body,
                         collision_frequency_ei, coupling_parameter,
                         debye_length, get_table, plasma_frequency,
                         recombination_time)
from .radiation import cooling_rate

#: Mean charge below which the electron energy equation is no longer
#: invertible for dT_e (the inversion divides by Zbar). See `rhs`.
Z_FLOOR = 1e-9


def energy_budget(res: ExpansionResult) -> dict:
    """Per-heavy-particle energy budget of a 2T run, in joules.

    With ``radiative_cooling=False`` the total is a conserved quantity, so
    ``drift`` is a direct test of the model rather than of the physics. Any
    non-zero drift that does *not* shrink when ``rtol`` is tightened is a
    structural error -- that is exactly how the ion-temperature floor was
    caught, at a fixed +1.31% independent of tolerance.

    Note the kinetic term. For a Gaussian plume whose velocity field is
    ``v = (x/R_r, y/R_r, z/R_z) . (Rdot_r, Rdot_r, Rdot_z)``, the mean square
    speed is ``2 Rdot_r^2 + Rdot_z^2`` -- two transverse axes, one axial. The
    factor 2 is not optional; dropping it makes a conserving model look like
    it loses a third of its energy.
    """
    if res.diagnostics.get("model") != "2T":
        raise ValueError("energy_budget expects a 2T ExpansionResult")
    m = res.material.m_atom
    chi = res.diagnostics["chi_eV"] * EV
    Te = res.diagnostics["T_e_eV"] * EV / K_B
    Ti = res.diagnostics["T_i_eV"] * EV / K_B
    v_cm = res.diagnostics.get("v_cm", 0.0)
    U_e = 1.5 * K_B * res.Zbar * Te
    U_i = 1.5 * K_B * Ti
    U_ion = res.Zbar * chi
    # strip the bulk drift: it is transport, not a reservoir the plume can
    # convert from or into.
    KE = 0.5 * m * (2.0 * res.v_r ** 2 + (res.v_z - v_cm) ** 2)
    total = U_e + U_i + U_ion + KE
    return {"U_e": U_e, "U_i": U_i, "U_ion": U_ion, "KE": KE,
            "total": total, "drift": float(total[-1] / total[0] - 1.0)}


def simulate_expansion_2t(impact: ImpactResult,
                          t_end: float | None = None,
                          n_out: int = 400,
                          n_shell: int = 24,
                          aspect0: float = 0.5,
                          include_bulk_drift: bool = True,
                          radiative_cooling: bool = True,
                          recombination_heating: bool = True,
                          equilibration: bool = True,
                          rtol: float = 1e-8,
                          atol: float = 1e-12) -> ExpansionResult:
    """Two-temperature expansion with non-equilibrium ionisation.

    Parameters
    ----------
    recombination_heating : bool
        Return the binding energy to the electron gas when a recombination
        occurs. This is the feedback the model exists to capture; setting it
        False isolates its effect.
    equilibration : bool
        Include electron-ion energy transfer. False forces full decoupling,
        which is useful as a limiting case.

    Returns
    -------
    ExpansionResult, with ``diagnostics["model"] == "2T"`` and the electron
    and ion temperature histories in ``diagnostics["T_e_eV"]`` and
    ``diagnostics["T_i_eV"]``.
    """
    mat = impact.material_mix
    if impact.m_vapour <= 0:
        raise ValueError("No vapour produced by this impact.")

    table = get_table(mat)
    N_heavy = impact.m_vapour / mat.m_atom
    m_i = mat.m_atom
    chi = mat.E_ion[0] if len(mat.E_ion) else 6.0 * EV

    R0 = impact.r_plume0
    Rz0 = aspect0 * R0
    st0 = impact.plasma
    v_cm = impact.v_expansion if include_bulk_drift else 0.0
    if t_end is None:
        t_end = 2000.0 * R0 / max(impact.v_expansion, 1.0)

    x_shell, w_shell = gaussian_shells(n_shell)
    # density weight and mass fraction of each Gaussian shell
    dens_w = np.exp(-0.5 * x_shell ** 2)
    mass_f = w_shell / max(w_shell.sum(), 1e-300)
    # See expansion.HEMISPHERE_FACTOR: the plume is a hemisphere on a
    # plane, so the real mass M occupies a half-Gaussian and the peak
    # density is that of a full Gaussian carrying 2M.
    peak_norm = HEMISPHERE_FACTOR * (2.0 * np.pi) ** -1.5

    def n_peak(Rr, Rz):
        return N_heavy * peak_norm / (Rr * Rr * Rz)

    T0 = max(st0.T, 300.0)
    Z0 = max(st0.Zbar, 1e-12)
    nsh = len(x_shell)

    def unpack(y):
        Rr = max(y[0], 1e-12)
        Rz = max(y[1], 1e-12)
        # y[5], y[6] are ln(T_e), ln(T_i): positivity is structural, so no
        # clamp is needed and none is applied.  A floor here would be an
        # energy source -- it holds the pressure up while the ODE keeps
        # cooling, and the phantom pressure does work on the expansion for
        # the rest of the run.  Measured at +1.3% of total energy.
        return Rr, Rz, y[2], y[3], y[4], np.exp(y[5]), np.exp(y[6]), \
            np.maximum(y[7:], 0.0)

    def rhs(t, y):
        Rr, Rz, vr, vz, zcm, Te, Ti, Z = unpack(y)
        nh = n_peak(Rr, Rz)
        Zbar = float(np.sum(mass_f * Z))
        Te_eV = Te * K_B / EV

        P_i = nh * K_B * Ti
        P_e = Zbar * nh * K_B * Te
        P = P_i + P_e
        rho = nh * m_i
        dlnV = 2.0 * vr / Rr + vz / Rz

        # --- ionisation relaxation, shell by shell ----------------------
        n_j = np.maximum(dens_w * nh, 1e-30)
        Z_eq = np.asarray(table.Zbar(n_j, Te), dtype=float)
        n_e_j = np.maximum(Z * n_j, 1e-30)
        Tc = max(Te_eV, 1e-4)
        Zc = np.maximum(Z, 0.05)
        rate = (alpha_three_body(Tc) * n_e_j ** 2
                + alpha_radiative(Tc, 1.0) * Zc ** 2 * n_e_j)
        dZ = (Z_eq - Z) * np.maximum(rate, 1e-18)
        dZbar = float(np.sum(mass_f * dZ))

        # --- electron energy --------------------------------------------
        # U_e = 3/2 Zbar k Te  =>  3/2 k (Zbar dTe + Te dZbar) = sources
        src_e = -(P_e / nh) * dlnV
        if recombination_heating:
            # recombining (dZbar < 0) releases chi per event into electrons
            src_e -= chi * dZbar
        if radiative_cooling:
            src_e -= cooling_rate(nh, max(Te_eV, 1e-6), max(Zbar, 1e-9),
                                  length=min(Rr, Rz), A=mat.A)["total"] / nh
        Q_ei = 0.0
        if equilibration and Zbar > 0:
            n_e = Zbar * nh
            nu = collision_frequency_ei(max(n_e, 1e-30), max(Te_eV, 1e-4),
                                        max(Zbar, 0.05))
            Q_ei = 3.0 * (M_ELECTRON / m_i) * nu * Zbar * K_B * (Te - Ti)
        src_e -= Q_ei

        # U_e = 3/2 Zbar k Te, so 3/2 k (Zbar dTe + Te dZbar) = src_e.
        # Inverting for dTe divides by Zbar. Below Z_FLOOR the electron gas
        # carries no meaningful energy and the inversion is numerical noise
        # amplified by 1/Zbar, so fall back to the physical limit: a
        # vanishing electron population simply cools adiabatically with the
        # flow. (Not a clamp on the state -- a change of closure.)
        if Zbar > Z_FLOOR:
            dTe = (src_e - 1.5 * K_B * Te * dZbar) / (1.5 * K_B * Zbar)
        else:
            dTe = -(2.0 / 3.0) * Te * dlnV
        dTi = (-(P_i / nh) * dlnV + Q_ei) / (1.5 * K_B)

        out = np.empty_like(y)
        out[0], out[1] = vr, vz
        out[2] = P / (rho * Rr)
        out[3] = P / (rho * Rz)
        out[4] = v_cm
        # evolve ln T, so d(ln T)/dt = (dT/dt)/T
        out[5], out[6] = dTe / Te, dTi / Ti
        out[7:] = dZ
        return out

    y0 = np.concatenate([[R0, Rz0, 0.0, 0.0, 0.0, np.log(T0), np.log(T0)],
                         np.full(nsh, Z0)])
    t_eval = np.geomspace(t_end * 1e-7, t_end, n_out)
    sol = solve_ivp(rhs, (0.0, t_end), y0, t_eval=t_eval, method="LSODA",
                    rtol=rtol, atol=atol, max_step=t_end / 100)
    if not sol.success:
        raise RuntimeError(f"2T expansion failed: {sol.message}")

    t = sol.t
    Rr = np.maximum(sol.y[0], 1e-12)
    Rz = np.maximum(sol.y[1], 1e-12)
    vr, vz, zcm = sol.y[2], sol.y[3], sol.y[4]
    Te = np.exp(sol.y[5])
    Ti = np.exp(sol.y[6])
    Zsh = np.clip(sol.y[7:], 0.0, None)

    nh_pk = n_peak(Rr, Rz)
    Zbar = np.einsum("j,jt->t", mass_f, Zsh)
    n_e = Zbar * nh_pk
    Te_eV = Te * K_B / EV
    Ti_eV = Ti * K_B / EV

    Zeq = np.array([table.Zbar(n, TT) for n, TT in zip(nh_pk, Te)])
    Q_free = E_CHARGE * Zbar * N_heavy

    omega = np.array([plasma_frequency(max(n, 1e-30)) for n in n_e])
    nu = np.array([collision_frequency_ei(max(n, 1e-30), max(T, 1e-4),
                                          max(z, 0.05))
                   for n, T, z in zip(n_e, Te_eV, Zbar)])
    lam = np.array([debye_length(max(n, 1e-30), max(T, 1e-4))
                    for n, T in zip(n_e, Te_eV)])
    gam = np.array([coupling_parameter(max(n, 1e-30), max(T, 1e-4))
                    for n, T in zip(n_e, Te_eV)])

    # Collisional -> collisionless crossing. Delegated to the one-temperature
    # implementation rather than reimplemented: Stage 3 reads keys out of this
    # dict by name, and a private near-duplicate here silently omitted
    # "f_pe", which crashed `simulate_emp` the first time a 2T scenario was
    # run end to end. One definition, one contract.
    transition = _find_transition(t, nu, omega, Rr, Rz, n_e, nh_pk, Te_eV,
                                  Zbar, lam, vr, vz, v_cm)

    # where the ionisation stops changing: the freeze
    freeze = {}
    dlnZ = np.abs(np.gradient(np.log(np.maximum(Zbar, 1e-30)),
                              np.log(np.maximum(t, 1e-30))))
    frozen = np.where(dlnZ < 1e-3)[0]
    if frozen.size:
        i = int(frozen[0])
        freeze = {"t": float(t[i]), "n_e": float(n_e[i]),
                  "T_eV": float(Te_eV[i]), "Zbar": float(Zbar[i]),
                  "shell_Z": Zsh[:, -1].tolist(),
                  "shell_mass_fraction": mass_f.tolist()}

    return ExpansionResult(
        # v_z carries the bulk drift, matching the one-temperature model's
        # contract exactly (expansion.py returns ``vz + v_cm``). Returning
        # the bare expansion rate here would silently mean something
        # different from the same field on the same dataclass.
        t=t, R_r=Rr, R_z=Rz, v_r=vr, v_z=vz + v_cm, z_cm=zcm,
        T=Te, T_eV=Te_eV, Zbar=Zbar, Zbar_eq=Zeq,
        n_h=nh_pk, n_e=n_e, omega_pe=omega, nu_ei=nu,
        lambda_De=lam, Gamma_coupling=gam, Q_free=Q_free,
        material=mat, N_heavy=N_heavy,
        shells={"x": x_shell.tolist(), "mass_fraction": mass_f.tolist(),
                "Z_final": Zsh[:, -1].tolist()},
        freeze=freeze, transition=transition,
        diagnostics={
            "model": "2T",
            "T_e_eV": Te_eV, "T_i_eV": Ti_eV,
            "v_cm": v_cm,
            "recombination_heating": recombination_heating,
            "equilibration": equilibration,
            "radiative_cooling": radiative_cooling,
            "chi_eV": chi / EV,
        })


__all__ = ["simulate_expansion_2t", "energy_budget", "Z_FLOOR"]
