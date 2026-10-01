"""The hydrocode -> PIC handoff: Fletcher's waterfall, as code.

Fletcher (2021), NRL/MR/6757--20-10,138, Section 2:

    "we break simulations of hypervelocity impact plasmas into two parts
    with the first providing initial and boundary conditions for the second
    in a waterfall fashion ... The basic plasma properties of the plume,
    such as the total charge generated, temperature, and expansion velocity,
    are determined by the hydrocode. The PIC takes these conditions..."

and on the join itself:

    "One difficulty is the transition between a collisional and
    collisionless state ... we follow the assumption of Close et al. in
    assuming this transition is instantaneous."

This module implements exactly that, and nothing the report does not
support:

1. **Which material is plume.** Hydrocode cells that are target or
   projectile material, in front of the target surface, moving away from
   it, and carrying free electrons. "The plasma forms from the impact
   (hydrocode) and separates from the target entirely, after which an EMP
   may be produced (PIC)."

2. **When to hand over.** Per cell, the electron plasma frequency against
   the electron-ion Coulomb collision frequency -- the two curves of
   Fletcher's Fig. 2. The handoff frame is the first in which at least
   `collisionless_fraction` of the plume's free electrons sit in cells
   where omega_pe > nu_ei. The transition is then treated as instantaneous,
   per Close et al.; the fraction still collisional at that moment is
   reported, because it is the size of that assumption.

3. **What is handed over.** The report's three scalars -- total charge,
   temperature, expansion velocity -- plus the spatial fields they summarise
   (n_e, n_heavy, Z-bar, T, velocity on the hydrocode mesh). The PIC is
   initialised from the fields; the scalars are for the record and for
   checking.

The handoff file is code-neutral (`.npz` arrays + `.json` metadata, no new
dependency), so WarpX and PIConGPU read the same thing.

What is deliberately NOT here
-----------------------------
* No collisional-to-collisionless *dynamics*. The report calls that future
  work and so does this.
* Electron-neutral collisions are computed as a diagnostic only. Fletcher's
  Fig. 2 uses the Coulomb frequency, and with Z-bar << 1 the neutral term
  can dominate -- when it does, the result says so rather than silently
  changing the criterion.
* No fitting of the plume to an analytic shape. A fit is what the PIC
  bridges did before, from the analytic chain; the point now is to carry
  the hydrocode's own fields.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from .constants import E_CHARGE, EV_PER_K, K_B, M_ELECTRON

__all__ = ["HandoffCriteria", "Handoff", "plasma_frequency",
           "coulomb_log", "nu_ei", "nu_en", "plume_mask", "assess",
           "find_handoff", "extract", "load_handoff", "EPS0"]

EPS0 = 8.8541878128e-12          # F/m


# ---------------------------------------------------------------------------
# Plasma frequencies (SI in, SI out)
# ---------------------------------------------------------------------------

def plasma_frequency(n_e):
    """Electron plasma frequency omega_pe [rad/s]."""
    return np.sqrt(np.asarray(n_e, float) * E_CHARGE ** 2
                   / (EPS0 * M_ELECTRON))


def coulomb_log(n_e, T_eV, Z=1.0, floor: float = 2.0):
    """Electron-ion Coulomb logarithm, NRL Plasma Formulary (2019) p. 34.

        T_e < 10 Z^2 eV:  23 - ln(n_e^1/2 Z T_e^-3/2)
        T_e > 10 Z^2 eV:  24 - ln(n_e^1/2 T_e^-1)

    with n_e in cm^-3. Floored at `floor`: in the dense early plume the
    formula goes below ~2, which means the plasma is strongly coupled and
    no Coulomb logarithm is meaningful -- those cells are deeply collisional
    either way, so the floor only stops a negative log turning into a
    negative collision rate.
    """
    n_cc = np.maximum(np.asarray(n_e, float) * 1e-6, 1e-30)
    T = np.maximum(np.asarray(T_eV, float), 1e-6)
    Z = np.maximum(np.asarray(Z, float), 1.0)
    low = 23.0 - np.log(np.sqrt(n_cc) * Z * T ** -1.5)
    high = 24.0 - np.log(np.sqrt(n_cc) / T)
    lnL = np.where(T < 10.0 * Z * Z, low, high)
    return np.maximum(lnL, floor)


def nu_ei(n_e, T_eV, Zbar):
    """Electron-ion Coulomb collision frequency [1/s].

    NRL electron collision rate, nu_e = 2.91e-6 n_e lnL T_e^-3/2 (cgs n_e,
    eV T_e), written for singly charged ions. For mean charge Z the rate
    goes as sum(n_i Z_i^2) = n_e <Z^2>/<Z>, which is ~Z-bar above 1 and ~1
    below it (a partially ionised plasma's ions are mostly singly charged;
    neutrals do not take part in Coulomb collisions). Hence max(Z-bar, 1).
    """
    n_cc = np.asarray(n_e, float) * 1e-6
    T = np.maximum(np.asarray(T_eV, float), 1e-6)
    Zeff = np.maximum(np.asarray(Zbar, float), 1.0)
    return 2.91e-6 * Zeff * n_cc * coulomb_log(n_e, T, Zbar) * T ** -1.5


def nu_en(n_heavy, Zbar, T_K, sigma_en: float):
    """Electron-neutral momentum-transfer rate [1/s]: diagnostic only.

    n_n sigma v_th,e with n_n = n_heavy (1 - Z-bar) for Z-bar < 1.
    `sigma_en` is an order-of-magnitude constant (~1e-19 m^2 for metal
    vapours); no material-specific value is claimed.
    """
    n_n = np.asarray(n_heavy, float) * np.clip(1.0 - np.asarray(Zbar, float),
                                               0.0, 1.0)
    v_th = np.sqrt(8.0 * K_B * np.maximum(np.asarray(T_K, float), 1.0)
                   / (np.pi * M_ELECTRON))
    return n_n * sigma_en * v_th


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

@dataclass
class HandoffCriteria:
    """Everything the handoff decision depends on, recorded in the file.

    The defaults match the decks this package writes: the target occupies
    x >= 0, its surface is at x = 0, and the projectile arrives along +x,
    so the plume leaves toward -x.
    """
    surface_x: float = 0.0              # m, on the hydrocode's x axis
    outward: float = -1.0               # sign of the outward normal along x
    plume_materials: tuple = (1, 2)     # target, projectile
    min_n_e: float = 1.0e12             # m^-3: below this, numerical dust
    min_outward_speed: float = 0.0      # m/s: must be leaving the surface
    collisionless_fraction: float = 0.5
    sigma_en: float = 1.0e-19           # m^2, electron-neutral diagnostic

    def __post_init__(self):
        if self.outward not in (-1.0, 1.0):
            raise ValueError("outward must be -1 or +1")
        if not 0.0 < self.collisionless_fraction <= 1.0:
            raise ValueError("collisionless_fraction must be in (0, 1]")


# ---------------------------------------------------------------------------
# Per-snapshot analysis
# ---------------------------------------------------------------------------

def _fields(snap):
    """(n_e, n_heavy, Zbar, T_K, vel) from a snapshot, SI, with checks."""
    missing = [k for k in ("electron_density", "temperature", "velocity",
                           "materialid") if k not in snap.fields]
    if missing:
        raise KeyError(
            f"{os.path.basename(snap.path)} lacks {missing}. The deck needs "
            f"ElectronDensity, Temperature, Velocity and MaterialID = On in "
            f"its Output block (decks from m2c_stage1 have them).")
    n_e = np.nan_to_num(np.asarray(snap["electron_density"], float))
    T = np.nan_to_num(np.asarray(snap["temperature"], float))
    vel = np.nan_to_num(np.asarray(snap["velocity"], float))
    if "heavy_particles_density" in snap.fields:
        n_h = np.nan_to_num(np.asarray(snap["heavy_particles_density"], float))
    else:
        n_h = np.full_like(n_e, np.nan)
    if "mean_charge_number" in snap.fields:
        Z = np.nan_to_num(np.asarray(snap["mean_charge_number"], float))
    else:
        Z = np.where(n_h > 0, n_e / np.where(n_h > 0, n_h, 1.0), 0.0)
    return n_e, n_h, Z, T, vel


def plume_mask(snap, crit: HandoffCriteria,
               require_leaving: bool = True) -> np.ndarray:
    """Cells that are plume: impact material, in front, leaving, ionised.

    `require_leaving=False` drops the direction test, which is how the
    charge in material falling back toward the target is measured -- it is
    reported, because excluding it is a choice ("separates from the target
    entirely") and its size should be visible.
    """
    n_e, _n_h, _Z, _T, vel = _fields(snap)
    mat = np.rint(np.asarray(snap["materialid"], float)).astype(int)
    X = np.broadcast_to(snap.x[:, None, None], n_e.shape)
    in_front = (X - crit.surface_x) * crit.outward > 0.0
    keep = np.isin(mat, crit.plume_materials) & in_front & (n_e > crit.min_n_e)
    if require_leaving:
        keep &= vel[..., 0] * crit.outward > crit.min_outward_speed
    return keep


def assess(snap, crit: HandoffCriteria) -> dict:
    """The plume's state in one frame, and how collisionless it is."""
    n_e, n_h, Z, T, vel = _fields(snap)
    mask = plume_mask(snap, crit)
    vol = snap.cell_volumes()
    w = np.where(mask, n_e * vol, 0.0)          # free electrons per cell
    N_e = float(w.sum())
    out = {"time": float(snap.time), "path": snap.path,
           "n_plume_cells": int(mask.sum()), "N_e": N_e,
           "total_charge_C": N_e * E_CHARGE}
    if N_e <= 0.0:
        out.update(collisionless_fraction=0.0, ready=False,
                   reason="no ionised plume in front of the target")
        return out

    T_eV = T * EV_PER_K
    wpe = plasma_frequency(n_e)
    nei = nu_ei(n_e, T_eV, Z)
    nen = nu_en(np.where(np.isfinite(n_h), n_h, 0.0), Z, T, crit.sigma_en)
    free = wpe > nei
    frac = float(w[free].sum() / N_e)
    frac_with_neutrals = float(w[wpe > nei + nen].sum() / N_e)
    out.update({
        "collisionless_fraction": frac,
        "collisionless_fraction_incl_neutrals": frac_with_neutrals,
        "neutral_collisions_matter": bool(
            np.average(nen, weights=w + 1e-300)
            > np.average(nei, weights=w + 1e-300)),
        "omega_pe_median": float(_wmedian(wpe[mask], w[mask])),
        "nu_ei_median": float(_wmedian(nei[mask], w[mask])),
        "ready": frac >= crit.collisionless_fraction,
    })
    return out


def _wmedian(v, w):
    if v.size == 0:
        return float("nan")
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return v[o][np.searchsorted(c, 0.5 * c[-1])]


def find_handoff(series, crit: HandoffCriteria, reader=None) -> tuple:
    """(index of the handoff frame or None, per-frame assessments).

    `series` is `[(time, path)]`, e.g. from `read_m2c_series`. Frames are
    read one at a time, so a long run does not have to fit in memory.
    """
    if reader is None:
        from .solvers.m2c_output import read_m2c_vtr as reader
    rows = []
    for t, path in series:
        snap = reader(path, time=t)
        a = assess(snap, crit)
        rows.append(a)
        if a.get("ready"):
            return len(rows) - 1, rows
    return None, rows


# ---------------------------------------------------------------------------
# The handoff itself
# ---------------------------------------------------------------------------

@dataclass
class Handoff:
    """The plume at the moment it is handed to the PIC code, in SI.

    `summary` holds the report's three quantities (total charge,
    temperature, expansion velocity) and the numbers needed to judge the
    handoff. `fields` holds the hydrocode mesh and the plume on it.
    """
    summary: dict
    fields: dict
    criteria: dict
    provenance: dict
    notes: list = field(default_factory=list)

    def save(self, stem: str) -> tuple:
        """Write `stem.npz` and `stem.json`. Returns both paths."""
        os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".",
                    exist_ok=True)
        npz = stem + ".npz"
        js = stem + ".json"
        np.savez_compressed(npz, **self.fields)
        with open(js, "w") as fh:
            json.dump({"format": "hvi_emp.handoff/1",
                       "summary": self.summary, "criteria": self.criteria,
                       "provenance": self.provenance, "notes": self.notes,
                       "arrays": sorted(self.fields)}, fh, indent=2,
                      default=float)
        return npz, js


def load_handoff(stem: str) -> Handoff:
    """Inverse of `Handoff.save`."""
    stem = stem[:-4] if stem.endswith((".npz", ".json")) else stem
    stem = stem[:-1] if stem.endswith(".") else stem
    with open(stem + ".json") as fh:
        meta = json.load(fh)
    if meta.get("format") != "hvi_emp.handoff/1":
        raise ValueError(f"{stem}.json is not an hvi_emp handoff file")
    with np.load(stem + ".npz") as z:
        fields = {k: z[k] for k in z.files}
    return Handoff(summary=meta["summary"], fields=fields,
                   criteria=meta["criteria"], provenance=meta["provenance"],
                   notes=meta.get("notes", []))


def extract(snap, crit: HandoffCriteria, assessment: dict | None = None,
            allow_synthetic: bool = False) -> Handoff:
    """Build the handoff from one hydrocode frame.

    Refuses a SYNTHETIC frame unless `allow_synthetic`, so a test fixture
    cannot be mistaken for an M2C result further down the chain.
    """
    synthetic = "SYNTHETIC" in os.path.basename(snap.path)
    if synthetic and not allow_synthetic:
        raise ValueError(f"{snap.path} is synthetic test output, not an M2C "
                         f"run. Pass allow_synthetic=True to use it anyway.")
    a = assessment or assess(snap, crit)
    if a["N_e"] <= 0.0:
        raise ValueError(f"no ionised plume at t = {snap.time:.3e} s: "
                         f"{a.get('reason', '')}")

    n_e, n_h, Z, T, vel = _fields(snap)
    mask = plume_mask(snap, crit)
    vol = snap.cell_volumes()
    w = np.where(mask, n_e * vol, 0.0)
    W = w.sum()
    rho = np.asarray(snap.get("density", np.zeros_like(n_e)), float)
    mat = np.rint(np.asarray(snap["materialid"], float)).astype(int)

    def wavg(q):
        return float((q * w).sum() / W)

    speed = np.linalg.norm(vel, axis=-1)
    T_eV = T * EV_PER_K
    # Expansion relative to the plume's own centre of mass: the drift and
    # the spread are different physics and the PIC needs both.
    v_bulk_x = wavg(vel[..., 0])
    X = np.broadcast_to(snap.x[:, None, None], n_e.shape)
    Y = np.broadcast_to(snap.y[None, :, None], n_e.shape)
    x_c = wavg(X)
    s_ax = np.sqrt(max(wavg((X - x_c) ** 2), 0.0))
    if snap.geometry == "cylindrical":
        s_rad = np.sqrt(max(wavg(Y ** 2), 0.0))          # rms radius
        v_rad = wavg(vel[..., 1])                         # mean dr/dt
    else:
        y_c = wavg(Y)
        s_rad = np.sqrt(max(wavg((Y - y_c) ** 2), 0.0))
        v_rad = float("nan")

    front = plume_mask(snap, crit, require_leaving=False)
    N_front = float(np.where(front, n_e * vol, 0.0).sum())

    by_mat = {int(m): float((np.where(mask & (mat == m), rho, 0.0)
                             * vol).sum())
              for m in crit.plume_materials}

    summary = {
        # the report's three quantities
        "total_charge_C": float(W * E_CHARGE),
        "T_e_eV_mean": wavg(T_eV),
        "expansion_speed_mean_m_s": wavg(speed),
        # their context
        "time_s": float(snap.time),
        "N_free_electrons": float(W),
        "Zbar_mean": float(W / max(float((np.where(mask, n_h, 0.0) * vol)
                                         .sum()), 1e-300))
                     if np.isfinite(n_h).all() else wavg(Z),
        "T_e_eV_p10": float(_wquantile(T_eV[mask], w[mask], 0.10)),
        "T_e_eV_p90": float(_wquantile(T_eV[mask], w[mask], 0.90)),
        "expansion_speed_p90_m_s": float(_wquantile(speed[mask], w[mask],
                                                    0.90)),
        "bulk_velocity_axial_m_s": v_bulk_x,
        "mean_radial_velocity_m_s": v_rad,
        "centroid_axial_m": x_c,
        "rms_axial_m": float(s_ax),
        "rms_radial_m": float(s_rad),
        "n_e_peak_m3": float(n_e[mask].max()),
        "plume_mass_kg_by_material": by_mat,
        "collisionless_fraction": a["collisionless_fraction"],
        "collisionless_fraction_incl_neutrals":
            a.get("collisionless_fraction_incl_neutrals"),
        "neutral_collisions_matter": a.get("neutral_collisions_matter"),
        "omega_pe_median_rad_s": a.get("omega_pe_median"),
        "nu_ei_median_s": a.get("nu_ei_median"),
        "n_plume_cells": int(mask.sum()),
        "geometry": snap.geometry,
        # what the direction test left out
        "charge_in_front_any_direction_C": N_front * E_CHARGE,
        "charge_excluded_backflow_C": (N_front - float(W)) * E_CHARGE,
    }

    notes = list(snap.notes)
    still = 1.0 - a["collisionless_fraction"]
    notes.append(
        f"Transition treated as instantaneous (Close et al., as in Fletcher "
        f"2021): {still:.0%} of the plume's free electrons are still in "
        f"cells with nu_ei > omega_pe at handoff, and the PIC will treat "
        f"them as collisionless. That fraction is the size of the "
        f"assumption.")
    back = (N_front - float(W)) / N_front if N_front > 0 else 0.0
    if back > 0.01:
        notes.append(
            f"{back:.1%} of the ionised material in front of the target is "
            f"moving back toward it and is not handed over (the plume is "
            f"what 'separates from the target', Fletcher 2021). Set "
            f"min_outward_speed to a large negative value to include it.")
    if a.get("neutral_collisions_matter"):
        notes.append(
            "Electron-neutral collisions exceed Coulomb collisions on "
            "average here (Z-bar is low). Fletcher's criterion, used for the "
            "handoff, counts Coulomb collisions only; with neutrals included "
            f"the collisionless fraction would be "
            f"{a.get('collisionless_fraction_incl_neutrals', 0):.0%}.")
    if synthetic:
        notes.append("SYNTHETIC input: a prescribed test cloud, not M2C.")

    fields = {
        "x": snap.x, "y": snap.y, "z": snap.z,
        "x_edges": _edges(snap.x), "y_edges": _edges(
            snap.y, lower=0.0 if snap.geometry == "cylindrical" else None),
        "cell_volume": vol,
        "plume_mask": mask.astype(np.uint8),
        "n_e": np.where(mask, n_e, 0.0),
        "n_heavy": np.where(mask, np.nan_to_num(n_h), 0.0),
        "Zbar": np.where(mask, Z, 0.0),
        "T_K": np.where(mask, T, 0.0),
        "velocity": np.where(mask[..., None], vel, 0.0),
        "materialid": mat.astype(np.int16),
    }
    return Handoff(summary=summary, fields=fields, criteria=asdict(crit),
                   provenance={"source_file": os.path.abspath(snap.path),
                               "source_time_s": float(snap.time),
                               "geometry": snap.geometry,
                               "hydrocode": "M2C",
                               "synthetic": synthetic,
                               "n_mpi_pieces": snap.n_pieces,
                               "strategy": "Fletcher (2021) NRL/MR/6757--20-"
                                           "10,138: hydrocode -> PIC "
                                           "waterfall"},
                   notes=notes)


def _edges(c, lower=None):
    from .solvers.m2c_output import cell_edges
    return cell_edges(c, lower=lower)


def _wquantile(v, w, q):
    if v.size == 0:
        return float("nan")
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return v[o][min(np.searchsorted(c, q * c[-1]), v.size - 1)]
