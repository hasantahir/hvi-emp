"""Read M2C's field output: `results/solution.pvd` and its `.vtr` snapshots.

This is the first half of the waterfall's join. Until it existed, M2C wrote
`electron_density` and `mean_charge_number` every output interval -- the deck
even says "these two are what hvi_emp Stage 3 consumes" -- and nothing read
them. Every PIC bridge was initialised from the analytic chain instead.

Format, from source rather than from inspection
-----------------------------------------------
M2C writes each snapshot through PETSc (`Output::WriteSolutionSnapshot`,
`PetscViewerVTKOpen` + `VecView`). The bytes are produced by PETSc's
`DMDAVTKWriteAll_VTR` (src/dm/impls/da/grvtk.c):

* `<VTKFile type="RectilinearGrid" ... header_type="UInt64">`
* **one `<Piece>` per MPI rank**, each with its own `Extent`, its own
  `Xcoord`/`Ycoord`/`Zcoord` and its own copy of every field -- so a
  64-rank run has 64 pieces that must be stitched together;
* every array `format="appended"`, preceded in the blob by an 8-byte count
  of the bytes that follow, data `Float64`, x fastest;
* vector fields (`velocity`) as one array with `NumberOfComponents="3"`.

The points are DMDA nodes, which in M2C's finite-volume scheme are **cell
centres**. Cell volumes are rebuilt from those centres.

Two M2C quirks this tolerates:

* `electron_density` is written twice (the block is duplicated in
  Output.cpp); the first copy is kept.
* if a DMDA has named fields PETSc writes `name.field`; the suffix is
  stripped.

Units: everything returned is SI. M2C's mm-g-s-K-A values are converted with
the same table the deck writer uses, so the two directions cannot drift.

Geometry: an M2C `Cylindrical` mesh has X as the symmetry axis and Y as the
radius (Y0 = 0) with one cell in Z. `geometry="auto"` infers that from
nz == 1.
"""

from __future__ import annotations

import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .m2c_stage1 import m2c_to_si

__all__ = ["M2CSnapshot", "read_m2c_vtr", "read_m2c_series",
           "cell_edges", "FIELD_UNITS"]

#: M2C output name -> unit kind for `m2c_to_si`, or None if dimensionless.
FIELD_UNITS = {
    "density": "density",
    "velocity": "velocity",
    "pressure": "pressure",
    "temperature": "temperature",
    "delta_temperature": "temperature",
    "internal_energy": "energy_per_mass",
    "delta_internal_energy": "energy_per_mass",
    "materialid": None,
    "mean_charge_number": None,
    "electron_density": "number_density",
    "heavy_particles_density": "number_density",
}

_NP = {"Float64": "<f8", "Float32": "<f4", "Int32": "<i4", "Int64": "<i8",
       "UInt8": "<u1", "UInt32": "<u4", "UInt64": "<u8"}


def cell_edges(centres: np.ndarray, lower: float | None = None) -> np.ndarray:
    """Cell boundaries from cell centres, for a possibly graded mesh.

    Interior boundaries are midpoints; the two end cells are taken
    symmetric about their centres. `lower` pins the first boundary (the
    axis, r = 0, for an axisymmetric mesh) when the centres alone would put
    it slightly off.
    """
    c = np.asarray(centres, float)
    if c.size == 1:
        # One cell: no spacing to infer from. The caller supplies the width
        # for a 2-D slab, so return a unit-width placeholder around it.
        return np.array([c[0] - 0.5, c[0] + 0.5])
    e = np.empty(c.size + 1)
    e[1:-1] = 0.5 * (c[1:] + c[:-1])
    e[0] = c[0] - (e[1] - c[0])
    e[-1] = c[-1] + (c[-1] - e[-2])
    if lower is not None:
        e[0] = lower
    return e


@dataclass
class M2CSnapshot:
    """One M2C output frame, in SI, on the global (stitched) mesh."""
    path: str
    time: float
    x: np.ndarray                 # cell centres [m]; axis for cylindrical
    y: np.ndarray                 # cell centres [m]; radius for cylindrical
    z: np.ndarray
    fields: dict                  # name -> (nx, ny, nz) or (nx, ny, nz, 3)
    geometry: str                 # "cylindrical" or "cartesian"
    n_pieces: int = 1
    notes: list = field(default_factory=list)

    @property
    def shape(self) -> tuple:
        return (self.x.size, self.y.size, self.z.size)

    def cell_volumes(self, dz: float | None = None) -> np.ndarray:
        """Volume of every cell [m^3], shaped like a scalar field.

        Cylindrical: 2 pi r dr dx integrated exactly, pi (r_out^2 - r_in^2)
        dx, with the first radial edge pinned to the axis. Cartesian 3-D:
        dx dy dz. `dz` only matters for a Cartesian 2-D slab.
        """
        ex = np.diff(cell_edges(self.x))
        if self.geometry == "cylindrical":
            ey = cell_edges(self.y, lower=0.0)
            ey = np.clip(ey, 0.0, None)
            ring = np.pi * (ey[1:] ** 2 - ey[:-1] ** 2)
            vol = ex[:, None] * ring[None, :]
            return np.repeat(vol[:, :, None], self.z.size, axis=2)
        dy = np.diff(cell_edges(self.y))
        if self.z.size > 1:
            dzz = np.diff(cell_edges(self.z))
        else:
            dzz = np.array([dz if dz is not None else 1.0])
        return ex[:, None, None] * dy[None, :, None] * dzz[None, None, :]

    def __getitem__(self, name):
        return self.fields[name]

    def get(self, name, default=None):
        return self.fields.get(name, default)


def _split(raw: bytes) -> tuple:
    """(XML header text, byte offset of the first appended byte)."""
    i = raw.find(b"<AppendedData")
    if i < 0:
        raise ValueError("no <AppendedData> block: not a PETSc/M2C .vtr "
                         "file, or it was written inline/ASCII")
    j = raw.index(b"_", i) + 1
    head = raw[:i].decode("utf-8", "replace")
    # PETSc closes </RectilinearGrid> before <AppendedData>; only the
    # document element is left open at the split.
    if "</RectilinearGrid>" not in head:
        head += "</RectilinearGrid>"
    return head + "</VTKFile>", j


def _read_block(raw: bytes, base: int, offset: int, dtype: str,
                count: int, header: str) -> np.ndarray:
    """One appended array: 4- or 8-byte length prefix, then `count` values.

    PETSc writes the prefix as a BYTE count (UInt64 for header_type
    UInt64). Older writers have used an element count; both are accepted
    and anything else is an error rather than a silent misread.
    """
    hsize = 8 if header == "UInt64" else 4
    pos = base + offset
    (n,) = struct.unpack_from("<Q" if hsize == 8 else "<I", raw, pos)
    item = np.dtype(dtype).itemsize
    if n == count * item:
        nvals = count
    elif n == count:
        nvals = count
    else:
        raise ValueError(f"appended block at offset {offset} declares {n} "
                         f"(bytes or values) but the piece needs {count} "
                         f"values of {item} bytes")
    return np.frombuffer(raw, dtype=dtype, count=nvals, offset=pos + hsize)


def _canonical(name: str) -> str:
    """'density.0' -> 'density' (PETSc's named-DMDA-field suffix)."""
    return re.sub(r"\.\w+$", "", name) if "." in name else name


def read_m2c_vtr(path: str, time: float = float("nan"),
                 geometry: str = "auto", si: bool = True) -> M2CSnapshot:
    """Read one M2C/PETSc `.vtr` snapshot, stitching all MPI pieces."""
    with open(path, "rb") as fh:
        raw = fh.read()
    text, base = _split(raw)
    root = ET.fromstring(text)
    if root.get("type") != "RectilinearGrid":
        raise ValueError(f"{os.path.basename(path)}: VTKFile type "
                         f"{root.get('type')!r}, expected RectilinearGrid")
    header = root.get("header_type", "UInt32")
    if root.get("byte_order", "LittleEndian") != "LittleEndian":
        raise ValueError("big-endian .vtr not supported")
    grid = root.find("RectilinearGrid")
    whole = [int(v) for v in grid.get("WholeExtent").split()]
    nx, ny, nz = (whole[1] - whole[0] + 1, whole[3] - whole[2] + 1,
                  whole[5] - whole[4] + 1)

    X = np.full(nx, np.nan)
    Y = np.full(ny, np.nan)
    Z = np.full(nz, np.nan)
    fields: dict = {}
    notes: list = []
    pieces = grid.findall("Piece")
    if not pieces:
        raise ValueError(f"{os.path.basename(path)}: no <Piece> elements")

    for piece in pieces:
        ext = [int(v) for v in piece.get("Extent").split()]
        i0, i1, j0, j1, k0, k1 = ext
        px, py, pz = i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1
        coords = piece.find("Coordinates").findall("DataArray")
        for arr, (lo, n, dst) in zip(coords, ((i0, px, X), (j0, py, Y),
                                             (k0, pz, Z))):
            vals = _read_block(raw, base, int(arr.get("offset")),
                               _NP[arr.get("type")], n, header)
            dst[lo:lo + n] = vals

        seen_here: set = set()
        for arr in piece.find("PointData").findall("DataArray"):
            name = _canonical(arr.get("Name"))
            if name in seen_here:
                # Output.cpp writes electron_density twice. Identical data.
                if name not in [n.split(":")[0] for n in notes]:
                    notes.append(f"{name}: duplicate array in the file, "
                                 f"first copy used")
                continue
            seen_here.add(name)
            nc = int(arr.get("NumberOfComponents", "1"))
            vals = _read_block(raw, base, int(arr.get("offset")),
                               _NP[arr.get("type")], px * py * pz * nc,
                               header)
            # x fastest: (k, j, i[, c]) in memory -> (i, j, k[, c])
            if nc == 1:
                block = vals.reshape(pz, py, px).transpose(2, 1, 0)
                tgt = fields.setdefault(name, np.full((nx, ny, nz), np.nan))
            else:
                block = vals.reshape(pz, py, px, nc).transpose(2, 1, 0, 3)
                tgt = fields.setdefault(name,
                                        np.full((nx, ny, nz, nc), np.nan))
            tgt[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = block

    for lab, a in (("x", X), ("y", Y), ("z", Z)):
        if np.isnan(a).any():
            raise ValueError(f"{os.path.basename(path)}: pieces do not cover "
                             f"the whole {lab} extent -- truncated write?")
    for name, a in fields.items():
        if np.isnan(a).any():
            raise ValueError(f"{os.path.basename(path)}: field {name!r} has "
                             f"uncovered cells -- a piece is missing")

    if geometry == "auto":
        geometry = "cylindrical" if nz == 1 and Y.min() >= -1e-12 else \
                   "cartesian"
    if geometry not in ("cylindrical", "cartesian"):
        raise ValueError(f"geometry must be cylindrical/cartesian/auto, "
                         f"got {geometry!r}")

    if si:
        X, Y, Z = (m2c_to_si(a, "length") for a in (X, Y, Z))
        for name in list(fields):
            kind = FIELD_UNITS.get(name)
            if kind is not None:
                fields[name] = m2c_to_si(fields[name], kind)

    return M2CSnapshot(path=path, time=time, x=X, y=Y, z=Z, fields=fields,
                       geometry=geometry, n_pieces=len(pieces), notes=notes)


def read_m2c_series(results_dir: str, pvd: str = "solution.pvd") -> list:
    """`[(time [s], path)]` from M2C's collection file, in time order.

    M2C rewrites the closing tags of the .pvd after every snapshot, so an
    interrupted run can leave it without them. That case is recovered by
    reading the DataSet lines directly rather than failing the XML parse.
    """
    p = os.path.join(results_dir, pvd)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"no {pvd} in {results_dir}. M2C writes it under the deck's "
            f"Output Prefix (results/ for decks from this package).")
    with open(p, "r", errors="replace") as fh:
        txt = fh.read()
    rows = re.findall(r'<DataSet\s+timestep="([^"]+)"\s+file="([^"]+)"', txt)
    out = []
    for t, f in rows:
        fp = f if os.path.isabs(f) else os.path.join(results_dir, f)
        out.append((float(t), fp))
    out.sort(key=lambda r: r[0])
    return out
