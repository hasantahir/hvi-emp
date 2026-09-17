"""Turn a LAMMPS impact trajectory into the crater -> ionisation -> plasma view.

This is the module that answers "show me the moment the crater makes plasma"
at MD resolution.  It reads a LAMMPS dump, reconstructs the local
thermodynamic state, applies the framework's own Saha solver, and writes both

* per-atom ``.vtp`` -- every atom, coloured by temperature or ionisation, so
  you get the lattice-and-ejecta detail an OVITO render gives you but with
  the plasma physics attached; and
* binned ``.vti`` -- the same state as continuum fields, which is what a
  volume rendering of "the plasma" actually needs.

The thing you must not forget
-----------------------------
**Classical MD cannot ionise.**  An EAM potential has no electrons in it; the
atoms are neutral point masses with an empirical many-body force, and no
trajectory it produces contains an ionisation event.  Fraile et al. (2022),
whose protocol the Stage-1 bridge follows, are simulating *mechanical*
cratering and sputtering, not plasma formation.

So the ionisation shown here is **inferred, not simulated**: MD gives density
and temperature honestly, and Saha then answers "what ionisation state would
matter in local thermodynamic equilibrium at this density and temperature?"
That inference is exactly the closure the reduced Stage 1 already makes -- the
value here is that it is applied cell by cell to a resolved crater rather than
to one bulk average, so you can *see* where the plasma comes from.

Two consequences worth stating before anyone quotes a number:

1. LTE is a real assumption.  At these densities collisional rates are high
   enough that it is defensible, but the shocked region is far from
   equilibrium for the first picosecond.
2. The MD box is nanometres.  Real impact plasma forms in a region a thousand
   times larger.  The scaling laws in `impact.py` are what carry a resolved
   nm-scale answer up to a micrometeoroid; a picture is not a prediction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

if __name__ == "__main__" and __package__ is None:      # pragma: no cover
    raise SystemExit(
        "\nThis file is a module of the `hvi_emp` package, not a script.\n"
        "Run it as a module from the project root:\n\n"
        "    python -m hvi_emp.solvers.lammps_postprocess\n")

from ..constants import AMU, EV, K_B, K_PER_EV
from ..ionization import IonisationTable, get_table
from ..materials import get_material
from ..viz.vti import write_pvd, write_vti, write_vtp

#: Columns we ask LAMMPS for, in the order the deck writes them.
DUMP_COLUMNS = ("id", "type", "x", "y", "z", "vx", "vy", "vz", "c_ke",
                "c_pe", "c_coord")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

@dataclass
class DumpFrame:
    """One timestep from a LAMMPS dump."""
    timestep: int
    box: np.ndarray                    # (3, 2) lo/hi per axis
    columns: tuple
    data: np.ndarray                   # (n_atoms, n_columns)

    def col(self, name: str) -> np.ndarray:
        if name not in self.columns:
            raise KeyError(
                f"column {name!r} not in this dump ({', '.join(self.columns)}). "
                "Regenerate the deck with generate_input_deck(..., "
                "dump_every=N), which asks LAMMPS for the full set.")
        return self.data[:, self.columns.index(name)]

    @property
    def n_atoms(self) -> int:
        return self.data.shape[0]

    def positions(self) -> np.ndarray:
        return np.stack([self.col(c) for c in ("x", "y", "z")], axis=-1)

    def velocities(self) -> np.ndarray:
        return np.stack([self.col(c) for c in ("vx", "vy", "vz")], axis=-1)


def read_lammps_dump(path: str, max_frames: int | None = None):
    """Read a text LAMMPS dump into DumpFrames.

    Handles the standard ``ITEM:`` block structure and whatever column set
    the deck asked for; it does not assume ours.
    """
    frames = []
    with open(path) as fh:
        line = fh.readline()
        while line:
            if not line.startswith("ITEM: TIMESTEP"):
                line = fh.readline()
                continue
            timestep = int(fh.readline().split()[0])
            assert fh.readline().startswith("ITEM: NUMBER OF ATOMS")
            n = int(fh.readline())
            hdr = fh.readline()
            assert hdr.startswith("ITEM: BOX BOUNDS"), hdr
            box = np.array([[float(v) for v in fh.readline().split()[:2]]
                            for _ in range(3)])
            hdr = fh.readline()
            assert hdr.startswith("ITEM: ATOMS"), hdr
            cols = tuple(hdr.split()[2:])
            rows = np.empty((n, len(cols)), dtype=float)
            for i in range(n):
                rows[i] = [float(v) for v in fh.readline().split()]
            frames.append(DumpFrame(timestep, box, cols, rows))
            if max_frames is not None and len(frames) >= max_frames:
                break
            line = fh.readline()
    if not frames:
        raise ValueError(f"no frames parsed from {path}; is it a LAMMPS dump?")
    return frames


# ---------------------------------------------------------------------------
# Local thermodynamic state
# ---------------------------------------------------------------------------

@dataclass
class AtomState:
    """Per-atom and per-cell state inferred from one MD frame."""
    frame: DumpFrame
    cell: float                        # bin size [m]
    origin: np.ndarray
    shape: tuple
    n_number: np.ndarray               # (nx,ny,nz) number density [m^-3]
    T_eV: np.ndarray                   # (nx,ny,nz) temperature [eV]
    Zbar: np.ndarray                   # (nx,ny,nz) mean charge
    n_e: np.ndarray                    # (nx,ny,nz) electron density [m^-3]
    atom_T_eV: np.ndarray              # per-atom, mapped from its cell
    atom_Zbar: np.ndarray
    atom_speed: np.ndarray             # |v| [m/s]
    idx: np.ndarray                    # (n_atoms, 3) cell index per atom
    #: Per-atom electron density [m^-3], mapped from the atom's cell. Carried
    #: per atom so a particle render can be coloured by charge state directly
    #: rather than by a volume behind it.
    atom_n_e: np.ndarray | None = None
    #: Per-atom flags from `attach_fragments`. ``None`` until it is called --
    #: fragment analysis needs a surface position and a cutoff, which are
    #: choices, so it is not done silently inside `analyse_frame`.
    atom_is_ejecta: np.ndarray | None = None
    atom_fragment: np.ndarray | None = None      # -1 for non-ejecta
    atom_fragment_mass: np.ndarray | None = None
    fragments: object = None                     # FragmentSet
    #: Fraction of atoms in cells with Zbar > 0.01. Always read this with
    #: `Zbar.max()`: a high peak over a tiny fraction is not a plasma.
    ionised_fraction: float = 0.0
    notes: list = field(default_factory=list)


def analyse_frame(frame: DumpFrame, material="W", cell: float | None = None,
                  length_unit: float = 1e-10, velocity_unit: float = 1e2,
                  min_atoms_per_cell: int = 4,
                  table: IonisationTable | None = None) -> AtomState:
    """Density, temperature and Saha ionisation, cell by cell.

    Parameters
    ----------
    frame : DumpFrame
    material : str or Material
        Target material, for atomic mass and ionisation potentials.
    cell : float, optional
        Bin size in metres.  Default is 5 lattice-ish spacings (1 nm), which
        holds enough atoms for a meaningful temperature without smearing the
        crater lip.
    length_unit, velocity_unit : float
        LAMMPS ``metal`` units: positions in Angstrom (1e-10 m), velocities in
        Angstrom/ps (1e2 m/s).  Change these if your deck uses other units.
    min_atoms_per_cell : int
        Cells with fewer atoms get no temperature: a "temperature" from two
        particles is noise, not physics.

    Notes
    -----
    Temperature is the *velocity dispersion within a cell*, not a per-atom
    kinetic energy.  A single atom has no temperature; a moving lump of cold
    atoms has large kinetic energy and zero temperature.  Getting this wrong
    makes the whole ejecta plume look hot and ionised when it is merely fast,
    which is the classic way to fabricate plasma that is not there.
    """
    mat = get_material(material) if isinstance(material, str) else material
    pos = frame.positions() * length_unit
    vel = frame.velocities() * velocity_unit
    m_atom = mat.m_atom

    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    if cell is None:
        cell = 1.0e-9
    shape = tuple(max(int(np.ceil((hi[d] - lo[d]) / cell)) + 1, 1)
                  for d in range(3))

    idx = np.floor((pos - lo) / cell).astype(int)
    idx = np.clip(idx, 0, np.array(shape) - 1)
    flat = np.ravel_multi_index(idx.T, shape)
    ncell = int(np.prod(shape))

    count = np.bincount(flat, minlength=ncell).astype(float)
    # mean (bulk) velocity per cell
    vsum = np.stack([np.bincount(flat, weights=vel[:, d], minlength=ncell)
                     for d in range(3)], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vbar = np.where(count[:, None] > 0, vsum / np.maximum(count, 1)[:, None],
                        0.0)
    # thermal energy = dispersion about the cell's own bulk motion
    dv = vel - vbar[flat]
    ke_th = 0.5 * m_atom * np.sum(dv * dv, axis=-1)
    ke_sum = np.bincount(flat, weights=ke_th, minlength=ncell)
    with np.errstate(invalid="ignore", divide="ignore"):
        T_K = np.where(count >= min_atoms_per_cell,
                       (2.0 / 3.0) * ke_sum / np.maximum(count, 1) / K_B, 0.0)

    n_number = count / cell**3
    T_eV = T_K / K_PER_EV

    # Saha, on the same tabulated solver the reduced chain uses
    tab = table if table is not None else get_table(mat)
    # IonisationTable.Zbar takes temperature in KELVIN, not eV. Only
    # evaluate where there is material and enough energy to matter -- below
    # ~0.02 eV Saha returns zero anyway and the work is wasted.
    #
    # This was a Python loop over live cells calling the scalar path once
    # each. On a real MD frame that is tens of thousands of interpreter
    # round-trips; `Zbar` is now vectorised over both arguments, so the
    # whole field is a single call.
    Zbar = np.zeros(ncell)
    live = np.flatnonzero((n_number > 0) & (T_eV > 0.02))
    if live.size:
        Zbar[live] = tab.Zbar(n_number[live], T_K[live])
    n_e = Zbar * n_number

    st = AtomState(
        frame=frame, cell=cell, origin=lo, shape=shape,
        n_number=n_number.reshape(shape), T_eV=T_eV.reshape(shape),
        Zbar=Zbar.reshape(shape), n_e=n_e.reshape(shape),
        atom_T_eV=T_eV[flat], atom_Zbar=Zbar[flat],
        atom_n_e=n_e[flat],
        atom_speed=np.linalg.norm(vel, axis=-1), idx=idx)

    hot = float(T_eV.max()) if T_eV.size else 0.0
    # The ionised MASS fraction, not just the peak. A peak Zbar from one hot
    # cell says almost nothing on its own: what decides whether the plasma
    # matters downstream is how much material is in that state. Reporting
    # only the peak is how an MD frame gets described as "making plasma"
    # when 99.9% of it is cold.
    ion_cells = Zbar > 0.01
    ion_atoms = float(count[ion_cells].sum())
    ion_frac = ion_atoms / max(frame.n_atoms, 1)
    st.ionised_fraction = ion_frac
    st.notes.append(
        f"{frame.n_atoms} atoms, {shape[0]}x{shape[1]}x{shape[2]} cells of "
        f"{cell * 1e9:.2f} nm; peak cell T = {hot:.3f} eV, "
        f"peak Zbar = {Zbar.max():.3f}, ionised mass fraction "
        f"{100.0 * ion_frac:.2f}% ({int(ion_atoms)} atoms in "
        f"{int(ion_cells.sum())} cells)")
    if Zbar.max() < 1e-3:
        st.notes.append(
            "no significant ionisation anywhere in this frame -- at this "
            "velocity that may well be the correct answer.")
    elif ion_frac < 0.2:
        st.notes.append(
            f"only {100.0 * ion_frac:.1f}% of the material is ionised. This "
            f"is the expected picture below the bulk vaporisation threshold: "
            f"a small, hot, localised fraction rather than a bulk plasma. "
            f"Quote the fraction with the peak Zbar, never the peak alone.")
    st.notes.append(
        "IONISATION IS INFERRED, NOT SIMULATED: classical MD has no "
        "electrons. Saha is applied to the MD density and temperature.")
    return st


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def attach_fragments(state: AtomState, material="W",
                     surface: float | None = None,
                     cutoff: float | None = None,
                     cutoff_factor: float | None = None,
                     length_unit: float = 1e-10,
                     velocity_unit: float = 1e2) -> AtomState:
    """Identify ejecta and cluster them, attaching the result to `state`.

    Kept separate from `analyse_frame` because it needs two choices that
    `analyse_frame` has no business guessing: where the original free surface
    was, and what distance counts as "connected". Both change the answer, and
    both are recorded on the returned `FragmentSet`.

    Adds `atom_is_ejecta`, `atom_fragment` (-1 where not ejecta),
    `atom_fragment_mass`, and the `fragments` object itself.
    """
    from .fragments import CUTOFF_FACTOR, analyse_fragments, identify_ejecta

    mat = get_material(material) if isinstance(material, str) else material
    f = state.frame
    pos = f.positions() * length_unit
    vel = f.velocities() * velocity_unit

    is_ej = identify_ejecta(pos, vel, surface=surface)
    frags = analyse_fragments(
        pos, vel, mat.m_atom, mask=is_ej, cutoff=cutoff,
        cutoff_factor=(CUTOFF_FACTOR if cutoff_factor is None
                       else cutoff_factor))

    atom_frag = np.full(f.n_atoms, -1, dtype=np.int32)
    atom_mass = np.zeros(f.n_atoms, dtype=float)
    if frags.index.size:
        atom_frag[frags.index] = frags.labels
        atom_mass[frags.index] = frags.masses[frags.labels]

    state.atom_is_ejecta = is_ej
    state.atom_fragment = atom_frag
    state.atom_fragment_mass = atom_mass
    state.fragments = frags
    state.notes.extend(frags.notes)
    state.notes.append(
        f"{int(is_ej.sum())} of {f.n_atoms} atoms are ejecta "
        f"({100.0 * is_ej.sum() / max(f.n_atoms, 1):.2f}%)")
    return state


def write_frame_vtp(state: AtomState, path: str, compress: bool = True) -> str:
    """Per-atom points with the inferred state attached.

    Open in ParaView, set representation to Points or Point Gaussian, and
    colour by ``T_eV`` or ``Zbar`` to see where the crater turns into plasma.
    """
    f = state.frame
    pos = f.positions().astype(np.float32)
    scalars = {
        "T_eV": state.atom_T_eV.astype(np.float32),
        "Zbar": state.atom_Zbar.astype(np.float32),
        "speed": state.atom_speed.astype(np.float32),
        "type": f.col("type").astype(np.float32),
    }
    if state.atom_n_e is not None:
        scalars["electron_density"] = state.atom_n_e.astype(np.float32)
        # log10 for the same reason the volume fields carry one: n_e spans
        # far too many decades for a linear colour map.
        scalars["log10_electron_density"] = np.log10(
            np.maximum(state.atom_n_e, 1e12)).astype(np.float32)
    if state.atom_is_ejecta is not None:
        scalars["is_ejecta"] = state.atom_is_ejecta.astype(np.float32)
    if state.atom_fragment is not None:
        scalars["fragment_id"] = state.atom_fragment.astype(np.float32)
    if state.atom_fragment_mass is not None:
        scalars["fragment_mass"] = state.atom_fragment_mass.astype(np.float32)
        # fragment size in atoms reads better on a colour bar than kilograms
        mat_m = state.atom_fragment_mass
        scalars["fragment_atoms"] = np.where(
            mat_m > 0, mat_m / max(mat_m[mat_m > 0].min(), 1e-300)
            if np.any(mat_m > 0) else 0.0, 0.0).astype(np.float32)
    for extra, name in (("c_ke", "ke"), ("c_pe", "pe"), ("c_coord", "coord")):
        if extra in f.columns:
            scalars[name] = f.col(extra).astype(np.float32)
    return write_vtp(path, pos, scalars, compress=compress)


def write_frame_vti(state: AtomState, path: str, compress: bool = True) -> str:
    """The same state binned to a grid, for volume rendering."""
    fields = {
        "number_density": state.n_number.astype(np.float32),
        "temperature_eV": state.T_eV.astype(np.float32),
        "ionisation": state.Zbar.astype(np.float32),
        "electron_density": state.n_e.astype(np.float32),
    }
    c = state.cell
    return write_vti(path, fields, spacing=(c, c, c),
                     origin=tuple(float(v) for v in state.origin),
                     compress=compress)


def convert_dump(dump_path: str, directory: str, material="W",
                 cell: float | None = None, name: str = "md",
                 timestep_fs: float = 0.1, max_frames: int | None = None,
                 atoms: bool = True, grid: bool = True,
                 fragments: bool = False, surface: float | None = None,
                 cutoff: float | None = None,
                 verbose: bool = True) -> dict:
    """LAMMPS dump -> ParaView series of atoms and/or binned plasma fields.

    Returns a dict of the written paths plus the per-frame peak temperature
    and ionisation, so a glance at the return value tells you whether the run
    made plasma at all.
    """
    os.makedirs(directory, exist_ok=True)
    frames = read_lammps_dump(dump_path, max_frames=max_frames)
    mat = get_material(material) if isinstance(material, str) else material
    table = get_table(mat)

    vtp_files, vti_files, times, peaks = [], [], [], []
    for i, fr in enumerate(frames):
        st = analyse_frame(fr, material=mat, cell=cell, table=table)
        if fragments:
            attach_fragments(st, material=mat, surface=surface, cutoff=cutoff)
        t = fr.timestep * timestep_fs * 1e-15
        times.append(t)
        rec = {"timestep": fr.timestep, "t": t,
               "T_eV_max": float(st.T_eV.max()),
               "Zbar_max": float(st.Zbar.max()),
               "n_e_max": float(st.n_e.max()),
               # always alongside the peak: a high Zbar over 1% of the mass
               # is not a plasma, and the peak alone invites that reading
               "ionised_fraction": float(st.ionised_fraction)}
        if fragments and st.fragments is not None:
            rec.update(n_fragments=st.fragments.n_fragments,
                       n_ejecta=int(st.atom_is_ejecta.sum()),
                       ejecta_mass=float(st.fragments.masses.sum()))
        peaks.append(rec)
        if atoms:
            vtp_files.append(write_frame_vtp(
                st, os.path.join(directory, f"{name}_atoms_{i:04d}.vtp")))
        if grid:
            vti_files.append(write_frame_vti(
                st, os.path.join(directory, f"{name}_grid_{i:04d}.vti")))
        if verbose and (i == 0 or i == len(frames) - 1):
            print(f"  frame {i:3d} t={t * 1e12:8.3f} ps  "
                  f"T_max={st.T_eV.max():7.3f} eV  "
                  f"Zbar_max={st.Zbar.max():6.3f}")

    out = {"frames": len(frames), "times": times, "peaks": peaks}
    note = ("IONISATION IS INFERRED, NOT SIMULATED.\n"
            "Classical MD has no electrons; Saha is applied to the MD "
            "density and temperature\nunder an LTE assumption. See "
            "hvi_emp/solvers/lammps_postprocess.py.")
    if atoms:
        out["atoms_pvd"] = write_pvd(
            os.path.join(directory, f"{name}_atoms.pvd"), vtp_files, times,
            comment=note)
        out["atoms"] = vtp_files
    if grid:
        out["grid_pvd"] = write_pvd(
            os.path.join(directory, f"{name}_grid.pvd"), vti_files, times,
            comment=note)
        out["grid"] = vti_files

    if verbose:
        Tmax = max(p["T_eV_max"] for p in peaks)
        Zmax = max(p["Zbar_max"] for p in peaks)
        print(f"\n  peak over the run: T = {Tmax:.3f} eV, Zbar = {Zmax:.4f}")
        if Zmax < 1e-3:
            print("  -> no ionisation. Correct for MD-accessible velocities; "
                  "see the module docstring.")
        print(f"  open {out.get('atoms_pvd', out.get('grid_pvd'))} in ParaView")
    return out


__all__ = ["DumpFrame", "AtomState", "read_lammps_dump", "analyse_frame",
           "write_frame_vtp", "write_frame_vti", "convert_dump",
           "DUMP_COLUMNS"]
