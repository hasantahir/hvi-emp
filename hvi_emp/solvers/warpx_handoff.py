"""WarpX initialised from the hydrocode handoff: the PIC half of the waterfall.

    python scripts/warpx_from_handoff.py full_chain/hydro/handoff/plume

What changed relative to `warpx_stage3`
---------------------------------------
`warpx_stage3` builds its plume from the reduced analytic chain: a Gaussian
wedge, and an electron drift of sqrt(m_i/m_e) times the ion drift imposed by
hand -- the Fletcher & Close (2017) assumption that a PIC run exists to
test. Here every macroparticle descends from an M2C cell:

* positions sampled inside the cell (uniform in x and in r^2, uniform in
  azimuth, so the cylindrical volume is respected);
* electron and ion macroparticles **co-located** in pairs, so the run starts
  exactly quasi-neutral (no sampling-noise fields at t = 0);
* both species carry that cell's single-fluid velocity -- the hydrocode has
  one velocity, and the separation is left for the PIC to produce -- plus a
  Maxwellian at the cell's temperature;
* weights from n_e times the cell volume, so the charge handed over is the
  charge the hydrocode computed, to round-off.

Getting from the handoff to the measured bands
----------------------------------------------
At the collisional -> collisionless transition the plume is dense: in a
millimetre plume the Debye length is nanometres and the plasma frequency is
hundreds of THz. Nothing there oscillates at 315 or 916 MHz; a plasma
frequency of 916 MHz needs n_e = eps0 m_e w^2 / e^2 = 1.0e16 m^-3.

Fletcher & Close (2017), the PIC study the 2021 report builds on, faced the
same gap and started their PIC where the plume resonates with the measured
band: "an impact plasma with a plasma frequency near the RF emission
frequency that was measured in experiments has a peak density of 10^16 m^-3
and a length scale of 40 mm". In between, the plume is collisionless and
quasi-neutral -- the instantaneous-transition assumption already says so --
so it expands ballistically.

`advance="band"` (default) does exactly that to the hydrocode's own plume:
every fluid element moves on its handed-over velocity until the peak
density reaches the resonant density of the highest band, and the EM PIC
starts there. That keeps the hydrocode's spatial and velocity structure,
where Fletcher & Close imposed an analytic profile. `advance="none"` starts
the PIC at the handoff itself, and `plan()` states what that costs.

Neither is free of assumption, and the record says which was used.

The sampler, the advance and the plan are numpy and tested. The generated
PICMI script needs pywarpx and has not been run in this environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from ..constants import AMU, C_LIGHT, E_CHARGE, K_B, M_ELECTRON

__all__ = ["WarpXHandoffConfig", "sample_cloud", "advance_ballistic",
           "cloud_stats", "time_to_peak_density", "n_resonant",
           "make_particles", "plan", "write_warpx_from_handoff"]

EPS0 = 8.8541878128e-12


def n_resonant(f_hz):
    """Electron density whose plasma frequency is f: eps0 m_e w^2 / e^2."""
    return EPS0 * M_ELECTRON * (2.0 * np.pi * np.asarray(f_hz, float)) ** 2 \
        / E_CHARGE ** 2


@dataclass
class WarpXHandoffConfig:
    """Choices for the PIC stage, all recorded in the generated directory."""
    ion_mass_amu: float
    advance: str = "band"              # "band" | "none"
    band_hz: tuple = (315e6, 916e6)    # Close et al.'s antenna bands
    scheme: str = "auto"               # "auto" | "explicit" | "implicit"
    ppc: int = 8                       # macroparticle pairs per plume cell
    max_particles: float = 2.0e7       # both species together
    cells_per_rms: float = 48.0        # structure resolution
    margin_rms: float = 3.0            # room beyond the cloud
    t_max_transits: float = 5.0        # electron transits of the rms size
    n_cycles_lowest_band: float = 3.0  # cycles of the lowest band frequency
    cell_step_budget: float = 2.0e11   # explicit is chosen if it fits this
    probe_rms: tuple = (2.0, 4.0)      # radial probes, in rms sizes
    seed: int = 1
    n_field_dumps: int = 200

    def __post_init__(self):
        if self.advance not in ("band", "none"):
            raise ValueError("advance must be 'band' or 'none'")
        if self.scheme not in ("auto", "explicit", "implicit"):
            raise ValueError("scheme must be 'auto', 'explicit' or "
                             "'implicit'")
        if self.ion_mass_amu <= 0:
            raise ValueError("ion_mass_amu must be positive")
        if self.ppc < 1:
            raise ValueError("ppc must be >= 1")


# ---------------------------------------------------------------------------
# The plume as fluid elements
# ---------------------------------------------------------------------------

def sample_cloud(h, cfg: WarpXHandoffConfig) -> dict:
    """Quasi-neutral fluid elements drawn from the handoff's plume cells.

    Positions in WarpX's RZ convention (x, y transverse; z the axis = M2C x),
    bulk velocity from the cell, the cell's temperature, and the electron
    weight each element carries.
    """
    if h.provenance.get("geometry") != "cylindrical":
        raise NotImplementedError(
            "the WarpX bridge maps M2C's axisymmetric mesh onto RZ. A 3-D "
            "(oblique) handoff needs a 3-D WarpX grid, not written yet -- "
            "the handoff file already carries 3-D data.")
    f = h.fields
    mask = f["plume_mask"].astype(bool)
    ii, jj, kk = np.nonzero(mask)
    if ii.size == 0:
        raise ValueError("handoff has no plume cells")
    xe, ye = f["x_edges"], f["y_edges"]
    ppc = int(max(1, min(cfg.ppc, cfg.max_particles // (2 * ii.size))))
    rng = np.random.default_rng(cfg.seed)

    rep = np.repeat(np.arange(ii.size), ppc)
    ci, cj, ck = ii[rep], jj[rep], kk[rep]
    n = rep.size
    ax = xe[ci] + rng.random(n) * (xe[ci + 1] - xe[ci])
    r2lo, r2hi = ye[cj] ** 2, ye[cj + 1] ** 2
    r = np.sqrt(r2lo + rng.random(n) * (r2hi - r2lo))
    th = rng.random(n) * 2.0 * np.pi
    v = f["velocity"][ci, cj, ck]
    vr = v[:, 1]
    Zi = max(float(h.summary.get("Zbar_mean", 1.0)), 1.0)
    return {
        "x": r * np.cos(th), "y": r * np.sin(th), "z": ax,
        "vx": vr * np.cos(th), "vy": vr * np.sin(th), "vz": v[:, 0],
        "T": f["T_K"][ci, cj, ck].astype(float),
        "w_e": (f["n_e"][ci, cj, ck] * f["cell_volume"][ci, cj, ck]
                / ppc).astype(float),
        "t": float(h.summary["time_s"]),
        "meta": {"ppc_used": ppc, "ppc_requested": cfg.ppc,
                 "n_cells": int(ii.size), "ion_charge_state": Zi,
                 "ion_mass_kg": cfg.ion_mass_amu * AMU,
                 "n_e_peak_handoff": float(f["n_e"][mask].max())},
    }


def advance_ballistic(cloud: dict, dt: float) -> dict:
    """Collisionless, quasi-neutral free expansion of the fluid elements.

    Each element keeps its velocity; positions move on straight lines. This
    is what the instantaneous collisional -> collisionless transition
    implies for the ions, and quasi-neutrality ties the electrons to them
    until the PIC starts. Temperature is carried unchanged, as Fletcher &
    Close carry theirs into the PIC.
    """
    out = dict(cloud)
    for a, v in (("x", "vx"), ("y", "vy"), ("z", "vz")):
        out[a] = cloud[a] + cloud[v] * dt
    out["t"] = cloud["t"] + dt
    return out


def _spread(cloud):
    """Weighted rms of x, y, z about the centroid."""
    w = cloud["w_e"]
    W = w.sum()
    s = []
    for a in ("x", "y", "z"):
        m = (w * cloud[a]).sum() / W
        s.append(np.sqrt(max((w * (cloud[a] - m) ** 2).sum() / W, 1e-300)))
    return np.array(s)


def time_to_peak_density(cloud: dict, n_peak0: float, n_target: float,
                         t_cap: float = 1.0) -> float:
    """Ballistic time for the peak density to fall from n_peak0 to n_target.

    Shape-preserving expansion: n_peak ~ 1/(s_x s_y s_z), with each rms
    width computed exactly from the elements at time t (it includes the
    position-velocity correlation of the hydrocode flow, not just the
    velocity spread). Bisection on a monotone function.
    """
    if n_target >= n_peak0:
        return 0.0
    s0 = np.prod(_spread(cloud))

    def n_at(t):
        return n_peak0 * s0 / np.prod(_spread(advance_ballistic(cloud, t)))

    hi = 1e-9
    while n_at(hi) > n_target:
        hi *= 2.0
        if hi > t_cap:
            raise ValueError(
                f"the plume does not thin to {n_target:.2e} m^-3 within "
                f"{t_cap:g} s of ballistic expansion -- check the handed-over "
                f"velocities")
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if n_at(mid) > n_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cloud_stats(cloud: dict, n_bins: int = 24) -> dict:
    """Centroid, sizes, extents and an equal-volume-ring density map.

    Radial bins are equal in r^2 so every ring has the same volume -- a
    linear binning puts the noisiest, smallest rings on the axis, which is
    exactly where the peak is.
    """
    w = cloud["w_e"]
    W = w.sum()
    r = np.hypot(cloud["x"], cloud["y"])
    z = cloud["z"]
    zc = float((w * z).sum() / W)
    s = _spread(cloud)
    rmax = float(r.max())
    zlo, zhi = float(z.min()), float(z.max())
    r_edges = rmax * np.sqrt(np.linspace(0, 1, n_bins + 1))
    z_edges = np.linspace(zlo, zhi, n_bins + 1)
    H, _, _ = np.histogram2d(r, z, bins=[r_edges, z_edges], weights=w)
    vol = (np.pi * np.diff(r_edges ** 2))[:, None] * np.diff(z_edges)[None, :]
    dens = H / vol
    filled = dens[dens > 0]
    o = np.argsort(filled)
    # electron-weighted 5% quantile from the tenuous side: the plume edge
    cum = np.cumsum((filled * vol.ravel()[(dens > 0).ravel()])[o])
    n_edge = float(filled[o][np.searchsorted(cum, 0.05 * cum[-1])])
    speed = np.sqrt(cloud["vx"] ** 2 + cloud["vy"] ** 2 + cloud["vz"] ** 2)
    o2 = np.argsort(speed)
    c2 = np.cumsum(w[o2])
    return {
        "t": cloud["t"], "centroid_z": zc,
        "rms_z": float(s[2]), "rms_r": float(np.sqrt(s[0] ** 2 + s[1] ** 2)),
        "z_min": zlo, "z_max": zhi, "r_max": rmax,
        "n_peak_binned": float(dens.max()), "n_edge_binned": n_edge,
        "v_p90": float(speed[o2][np.searchsorted(c2, 0.9 * c2[-1])]),
        "T_mean_K": float((w * cloud["T"]).sum() / W),
        "N_e": float(W),
    }


def make_particles(cloud: dict, cfg: WarpXHandoffConfig) -> dict:
    """Co-located electron/ion pairs: bulk velocity plus a Maxwellian."""
    rng = np.random.default_rng(cfg.seed + 1)
    meta = cloud["meta"]
    n = cloud["x"].size

    def species(m, w):
        s = np.sqrt(K_B * cloud["T"] / m)
        u = [cloud[k] + s * rng.standard_normal(n)
             for k in ("vx", "vy", "vz")]
        g = 1.0 / np.sqrt(np.maximum(1.0 - sum(c * c for c in u)
                                     / C_LIGHT ** 2, 1e-12))
        return {"x": cloud["x"], "y": cloud["y"], "z": cloud["z"],
                "ux": (g * u[0]).astype(np.float32),
                "uy": (g * u[1]).astype(np.float32),
                "uz": (g * u[2]).astype(np.float32), "w": w}

    Zi = meta["ion_charge_state"]
    return {"electrons": species(M_ELECTRON, cloud["w_e"]),
            # ion charge Zi e, weight w_e / Zi: each pair is neutral. Where a
            # cell's Z-bar differs from the mean its ions carry the mean q/m
            # -- a stated approximation.
            "ions": species(meta["ion_mass_kg"], cloud["w_e"] / Zi)}


# ---------------------------------------------------------------------------
# Grid, time step and scheme
# ---------------------------------------------------------------------------

def plan(st: dict, cfg: WarpXHandoffConfig, surface_z: float = 0.0) -> dict:
    """Domain, resolution, duration and scheme for the cloud at PIC start."""
    T = st["T_mean_K"]
    n_peak, n_edge = st["n_peak_binned"], st["n_edge_binned"]
    lam_D = float(np.sqrt(EPS0 * K_B * T / (n_peak * E_CHARGE ** 2)))
    wpe_peak = float(np.sqrt(n_peak * E_CHARGE ** 2 / (EPS0 * M_ELECTRON)))
    wpe_edge = float(np.sqrt(n_edge * E_CHARGE ** 2 / (EPS0 * M_ELECTRON)))
    v_te = float(np.sqrt(K_B * T / M_ELECTRON))
    L = max(st["rms_z"], st["rms_r"])
    f_lo = min(cfg.band_hz)
    t_transit = cfg.t_max_transits * L / v_te
    t_band = cfg.n_cycles_lowest_band / f_lo
    t_max = max(t_transit, t_band)

    def layout(dx):
        margin = cfg.margin_rms * L + st["v_p90"] * t_max
        z_lo = st["z_min"] - margin
        r_hi = st["r_max"] + margin
        nz = int(np.ceil((surface_z - z_lo) / dx))
        nr = int(np.ceil(r_hi / dx))
        return z_lo, r_hi, nr, nz

    # explicit: resolve the Debye length at the peak, and the peak omega_pe
    dx_e = min(L / cfg.cells_per_rms, lam_D)
    z_lo_e, r_hi_e, nr_e, nz_e = layout(dx_e)
    dt_e = min(0.9 * dx_e / (C_LIGHT * np.sqrt(2.0)), 0.2 / wpe_peak)
    steps_e = int(np.ceil(t_max / dt_e))
    cost_e = float(nr_e) * nz_e * steps_e

    # implicit: resolve structure and the edge, where the shell radiates
    dx_i = L / cfg.cells_per_rms
    z_lo_i, r_hi_i, nr_i, nz_i = layout(dx_i)
    dt_i = min(0.9 * dx_i / C_LIGHT, 0.2 / wpe_edge)
    steps_i = int(np.ceil(t_max / dt_i))
    cost_i = float(nr_i) * nz_i * steps_i

    scheme = cfg.scheme
    if scheme == "auto":
        scheme = "explicit" if cost_e <= cfg.cell_step_budget else "implicit"
    if scheme == "explicit":
        dx, dt, nr, nz, z_lo, r_hi, steps = (dx_e, dt_e, nr_e, nz_e, z_lo_e,
                                             r_hi_e, steps_e)
    else:
        dx, dt, nr, nz, z_lo, r_hi, steps = (dx_i, dt_i, nr_i, nz_i, z_lo_i,
                                             r_hi_i, steps_i)

    notes = [
        f"t_max = {t_max:.3e} s, set by "
        + (f"{cfg.t_max_transits:g} electron transits of the plume rms "
           f"({t_band:.2e} s would cover {cfg.n_cycles_lowest_band:g} cycles "
           f"of {f_lo / 1e6:.0f} MHz)" if t_transit >= t_band else
           f"{cfg.n_cycles_lowest_band:g} cycles of {f_lo / 1e6:.0f} MHz "
           f"(electron transits need only {t_transit:.2e} s)")
        + ". Lower t_max_transits if the shell dynamics finish sooner."]
    if scheme == "explicit":
        notes.append(f"Explicit EM PIC resolving the peak Debye length "
                     f"({lam_D:.3e} m): {cost_e:.2e} cell-steps, within the "
                     f"{cfg.cell_step_budget:.1e} budget.")
    else:
        notes.append(
            f"Theta-implicit EM: explicit would need {cost_e:.2e} cell-steps "
            f"(budget {cfg.cell_step_budget:.1e}). dx = {dx:.3e} m is "
            f"{dx / lam_D:.3g}x the peak Debye length; omega_pe dt = "
            f"{wpe_edge * dt:.2g} at the edge (resolved) and "
            f"{wpe_peak * dt:.3g} at the peak (stable, damped). If the "
            f"radiating shell sits in under-resolved plasma the EMP is "
            f"numerical -- vary dx first.")
    if cost_i > cfg.cell_step_budget and scheme == "implicit":
        notes.append(f"Even implicit needs {cost_i:.2e} cell-steps "
                     f"({steps} steps). This start epoch is too dense for an "
                     f"EM PIC window; use advance='band'.")
    return {
        "scheme": scheme, "dx": dx, "dt": dt, "nr": nr, "nz": nz,
        "n_steps": steps, "t_max": t_max, "z_lo": z_lo, "z_hi": surface_z,
        "r_hi": r_hi, "lambda_D_peak": lam_D, "omega_pe_peak": wpe_peak,
        "omega_pe_edge": wpe_edge, "v_te": v_te, "L_ref": L,
        "cell_steps_explicit": cost_e, "cell_steps_implicit": cost_i,
        "cell_steps": float(nr) * nz * steps,
        "probe_r": [p * L for p in cfg.probe_rms],
        "probe_z": st["centroid_z"], "notes": notes,
    }


# ---------------------------------------------------------------------------
# The generated run directory
# ---------------------------------------------------------------------------

_SCRIPT = '''#!/usr/bin/env python3
"""WarpX RZ run initialised from an M2C handoff (Fletcher 2021 waterfall).

Generated by hvi_emp.solvers.warpx_handoff. Not executed by the generator.

    mpirun -np <N> python warpx_from_handoff.py

Hydrocode handoff : {source}
    t = {t_h:.4e} s, total charge {Q:.4e} C, T_e {T:.3g} eV,
    {frac:.1%} of free electrons collisionless at handoff
PIC start         : t = {t0:.4e} s  ({advance_note})
    peak n_e {npk:.3e} m^-3, edge n_e {ned:.3e} m^-3, rms {L:.3e} m
Grid  : RZ {nr} x {nz}, dx = {dx:.3e} m, dt = {dt:.3e} s, {steps} steps
        t_max = {tmax:.3e} s
Scheme: {scheme}
{notes}
"""
import numpy as np
from pywarpx import picmi

# ---- choices most likely to need changing on a given WarpX build ---------
SCHEME = "{scheme}"            # "explicit" or "implicit" (theta, Picard)
# Fields: "none" is the axis; "open" is WarpX's absorbing PML; "dirichlet"
# is a conducting wall (PEC) for the EM solver -- the target surface. If
# this WarpX rejects open boundaries in RZ with the Yee solver, switch the
# open entries to its absorbing alternative (absorbing_silver_mueller in a
# native inputs file).
FIELD_LO = ["none", "open"]        # [r_min (axis), z_min]
FIELD_HI = ["open", "dirichlet"]   # [r_max, z_max = target surface]
PART_LO = ["none", "absorbing"]
PART_HI = ["absorbing", "absorbing"]
# ---------------------------------------------------------------------------

grid = picmi.CylindricalGrid(
    number_of_cells=[{nr}, {nz}],
    n_azimuthal_modes=1,
    lower_bound=[0.0, {z_lo:.9e}],
    upper_bound=[{r_hi:.9e}, {z_hi:.9e}],
    lower_boundary_conditions=FIELD_LO,
    upper_boundary_conditions=FIELD_HI,
    lower_boundary_conditions_particles=PART_LO,
    upper_boundary_conditions_particles=PART_HI,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)

electrons = picmi.Species(particle_type="electron", name="electrons")
ions = picmi.Species(name="ions", charge={Zi:.6f} * picmi.constants.q_e,
                     mass={m_i:.9e})

kw = {{}}
if SCHEME == "implicit":
    kw["warpx_evolve_scheme"] = picmi.ThetaImplicitEMEvolveScheme(
        nonlinear_solver=picmi.PicardNonlinearSolver(
            max_iterations=25, relative_tolerance=1.0e-8),
        theta=0.5)

sim = picmi.Simulation(solver=solver, time_step_size={dt:.9e},
                       max_steps={steps}, particle_shape="cubic", **kw)
sim.add_species(electrons, layout=None)
sim.add_species(ions, layout=None)

sim.add_diagnostic(picmi.FieldDiagnostic(
    name="fields", grid=grid, period=max(1, {steps} // {dumps}),
    data_list=["E", "B", "rho", "J"], write_dir="diags",
    warpx_format="openpmd"))

sim.initialize_inputs()
sim.initialize_warpx()

# ---- the handoff: every macroparticle descends from an M2C cell ----------
p = np.load("particles.npz")
for name in ("electrons", "ions"):
    sim.particles.get(name).add_particles(
        x=p[name + "_x"], y=p[name + "_y"], z=p[name + "_z"],
        ux=p[name + "_ux"], uy=p[name + "_uy"], uz=p[name + "_uz"],
        w=p[name + "_w"],
        # every rank loads the same arrays; WarpX keeps one copy in total
        unique_particles=False)

sim.step()
print("done. Probe positions (r, z) [m]:", {probes!r})
print("Read E at a probe: hvi_emp.solvers.warpx_stage3.load_warpx_probe")
'''


def write_warpx_from_handoff(h, outdir: str, cfg: WarpXHandoffConfig) -> dict:
    """Advance (optionally), sample, plan and write one PIC run directory."""
    os.makedirs(outdir, exist_ok=True)
    cloud = sample_cloud(h, cfg)
    meta = dict(cloud["meta"])
    n_target = float(n_resonant(max(cfg.band_hz)))
    # Free streaming leaves out post-transition ambipolar acceleration: the
    # electrons' remaining thermal energy still pushes the ions outward.
    # Compare it with the ions' kinetic energy to bound the error.
    w = cloud["w_e"]
    Zi = meta["ion_charge_state"]
    v2 = cloud["vx"] ** 2 + cloud["vy"] ** 2 + cloud["vz"] ** 2
    e_kin = 0.5 * meta["ion_mass_kg"] * float((w * v2).sum() / w.sum())
    e_th = 1.5 * K_B * float((w * cloud["T"]).sum() / w.sum()) * (1.0 + Zi)
    ratio = e_th / e_kin
    meta["thermal_to_kinetic_per_ion"] = ratio
    meta["max_velocity_underestimate"] = float(np.sqrt(1.0 + ratio))
    if cfg.advance == "band":
        dt_adv = time_to_peak_density(cloud, meta["n_e_peak_handoff"],
                                      n_target)
        cloud = advance_ballistic(cloud, dt_adv)
        advance_note = (
            f"ballistic quasi-neutral expansion for {dt_adv:.3e} s after "
            f"handoff, to a peak density resonant at "
            f"{max(cfg.band_hz) / 1e6:.0f} MHz ({n_target:.2e} m^-3), as "
            f"Fletcher & Close (2017) start their PIC. Free streaming omits "
            f"post-transition ambipolar acceleration: residual thermal energy "
            f"is {ratio:.2f}x the ion kinetic energy, so velocities -- and "
            f"the time to reach the band -- could be off by up to "
            f"{np.sqrt(1.0 + ratio):.2f}x")
    else:
        dt_adv = 0.0
        advance_note = "PIC starts at the hydrocode handoff itself"
    st = cloud_stats(cloud)
    surface = float(h.criteria.get("surface_x", 0.0))
    pl = plan(st, cfg, surface_z=surface)
    parts = make_particles(cloud, cfg)

    flat = {f"{sp}_{k}": v for sp, arrs in parts.items()
            for k, v in arrs.items()}
    np.savez(os.path.join(outdir, "particles.npz"), **flat)

    s = h.summary
    script = _SCRIPT.format(
        source=h.provenance.get("source_file", "?"), t_h=s["time_s"],
        Q=s["total_charge_C"], T=s["T_e_eV_mean"],
        frac=s["collisionless_fraction"], t0=st["t"],
        advance_note=advance_note, npk=st["n_peak_binned"],
        ned=st["n_edge_binned"], L=pl["L_ref"], nr=pl["nr"], nz=pl["nz"],
        dx=pl["dx"], dt=pl["dt"], steps=pl["n_steps"], tmax=pl["t_max"],
        scheme=pl["scheme"], notes="\n".join("    " + n for n in pl["notes"]),
        z_lo=pl["z_lo"], z_hi=pl["z_hi"], r_hi=pl["r_hi"],
        Zi=meta["ion_charge_state"], m_i=meta["ion_mass_kg"],
        dumps=cfg.n_field_dumps,
        probes=[(r, pl["probe_z"]) for r in pl["probe_r"]])
    sp = os.path.join(outdir, "warpx_from_handoff.py")
    with open(sp, "w") as fh:
        fh.write(script)
    os.chmod(sp, 0o755)

    charge = float(parts["electrons"]["w"].sum() * E_CHARGE)
    record = {"plan": pl, "cloud_at_pic_start": st, "particles": meta,
              "advance": {"mode": cfg.advance, "dt_s": dt_adv,
                          "n_target_m3": n_target, "note": advance_note},
              "config": asdict(cfg), "handoff_summary": s,
              "handoff_provenance": h.provenance,
              "handoff_notes": h.notes, "charge_in_particles_C": charge}
    with open(os.path.join(outdir, "plan.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=float)
    return {"script": sp, "particles": os.path.join(outdir, "particles.npz"),
            "plan": pl, "stats": st, "meta": meta, "advance_dt": dt_adv,
            "advance_note": advance_note, "charge_in_particles_C": charge}
