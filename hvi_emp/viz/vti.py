"""VTK ImageData (`.vti`) writer and the impact/plume/smoke volume sampler.

File format
-----------
VTK XML ImageData, version 1.0, little-endian, ``header_type="UInt64"``, with
raw appended binary payloads.  Each array in the appended section is preceded
by an 8-byte unsigned length in bytes.  This is the compact, fast-loading
variant; ParaView and VisIt read it natively.

Physics provenance
------------------
The reduced chain solves the *plume*.  It does not solve the solid target, so
the crater and the shock front are drawn from the model's own scaling laws
rather than from a continuum solution.  Every field carries a provenance tag
(`FIELD_PROVENANCE`), which is also written into the `.pvd` as a comment, so
an image cannot quietly imply more than the model delivers.

    solved       integrated by the Stage-2 ODE
    parametric   the model's own closed-form law, evaluated for display
    illustrative geometry only; no dynamics behind it
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np

from ..constants import EV, K_B

#: What each output field actually is. See the module docstring.
FIELD_PROVENANCE = {
    "density": "solved (plume) + parametric (condensate)",
    "plasma_density": "solved -- Stage-2 self-similar expansion",
    "log10_plasma_density": "solved -- log10 of plasma_density, floored; "
                            "colour by THIS, not the raw density",
    "log10_electron_density": "solved -- log10 of electron_density, floored",
    "smoke_density": "parametric -- two-phase condensate, Stage-1 inventory "
                     "expanded at the condensate sound speed",
    "temperature_eV": "solved -- Stage-2 energy equation",
    "electron_density": "solved -- Stage-2 with shell-resolved freeze-out",
    "ionisation": "solved -- Saha with freeze-out",
    "pressure": "solved -- plume; parametric in the target",
    "velocity": "solved -- self-similar plus centre-of-mass drift",
    "shock_pressure": "parametric -- P_ic (r/r_ic)^-n decay law (THEORY 2.6)",
    "material": "illustrative -- 0 vacuum, 1 target, 2 crater, 3 plume, "
                "4 condensate, 5 projectile. The crater is a SPHERICAL CAP "
                "by construction and the plume a GAUSSIAN ELLIPSOID; neither "
                "shape is solved",
    "projectile": "illustrative -- rigid sphere before contact",
    "projectile_fraction": "solved as a SINGLE NUMBER, drawn as a UNIFORM "
                           "field -- the reduced chain mixes projectile and "
                           "target into one homogeneous plume and has no "
                           "spatial mixing structure at all",
}

_VTK_TYPE = {np.dtype("float32"): "Float32", np.dtype("float64"): "Float64",
             np.dtype("uint8"): "UInt8", np.dtype("int32"): "Int32"}


#: Half-width of the co-moving box, in plume Gaussian sigmas.
#:
#: The plume is a Gaussian, so its *visible* extent is several sigma, not
#: one: the 1e-2 contour sits at 3.03 sigma and the 1e-3 contour at 3.72.
#: A box sized at 1.3 sigma -- which is what the lab-frame factor gives --
#: is therefore overflowed by the very part of the plume you can see, and
#: the render fills the frame edge to edge from the second frame on. 4.0
#: holds the 1e-3 contour with a little margin left for it to grow into.
PLUME_BOX_SIGMA = 4.0


def _note_once(notes: list, text: str) -> None:
    """Append `text` unless it is already there.

    `_window_times` is called from several places (sizing, frame times, the
    run notes), so a note raised inside it would otherwise appear three
    times in one report and look like three separate problems.
    """
    if text not in notes:
        notes.append(text)


def _log10_floor(a, floor: float):
    """``log10(max(a, floor))`` -- a log field a linear colour map can show.

    Rendering a plume by its raw density does not work, and the reason is
    worth stating once. The density is a Gaussian, so inside a single frame
    it runs from its peak down through ~36 decades before it underflows
    float32. Across the series the peak itself falls ~2 more. A linear
    colour bar over that range puts everything except a handful of central
    voxels in the bottom colour, which renders as a flat blob of the "zero"
    colour with a faint halo -- regardless of how much structure the
    solution actually has.

    Flooring first is what keeps this bounded: without it, log10 of the
    Gaussian tail reaches -300 and the colour range is just as useless in
    the other direction.
    """
    return np.log10(np.maximum(np.asarray(a, dtype=float), floor))


# ---------------------------------------------------------------------------
# Low-level writers
# ---------------------------------------------------------------------------

def _compress_blocks(payload: bytes, block_size: int = 1 << 15):
    """zlib-compress a payload in VTK's blocked format.

    VTK's appended-data header for a compressed array is
    ``[nblocks][uncompressed block size][last block size][csize_1..csize_n]``
    (all UInt64 when header_type="UInt64"), followed by the compressed
    blocks.  Getting this wrong produces files that load as garbage rather
    than failing, which is why the test suite round-trips it.
    """
    import zlib

    n = len(payload)
    if n == 0:
        return struct.pack("<QQQ", 0, block_size, 0)
    blocks = [payload[i:i + block_size] for i in range(0, n, block_size)]
    comp = [zlib.compress(b, 6) for b in blocks]
    last = len(blocks[-1])
    header = struct.pack("<QQQ", len(blocks), block_size,
                         0 if last == block_size else last)
    header += b"".join(struct.pack("<Q", len(c)) for c in comp)
    return header + b"".join(comp)


def write_vti(path: str, fields: dict, spacing, origin=(0.0, 0.0, 0.0),
              point_data: bool = True, compress: bool = True) -> str:
    """Write one `.vti` file.

    Parameters
    ----------
    path : str
    fields : dict[str, ndarray]
        Arrays shaped ``(nx, ny, nz)`` for scalars or ``(nx, ny, nz, 3)`` for
        vectors.  All must share the same spatial shape.  Written in VTK's
        Fortran (x-fastest) order.
    spacing : (dx, dy, dz)
    origin : (x0, y0, z0)
    point_data : bool
        True writes point data (extent = shape-1), False cell data
        (extent = shape).
    compress : bool
        zlib-compress the payload.  These fields are mostly zero and smooth,
        so this typically saves 5-20x -- the difference between a usable
        series and one that fills a disk.
    """
    if not fields:
        raise ValueError("no fields to write")
    shapes = {v.shape[:3] for v in fields.values()}
    if len(shapes) != 1:
        raise ValueError(f"fields have inconsistent shapes: {shapes}")
    nx, ny, nz = shapes.pop()

    ext = (nx - 1, ny - 1, nz - 1) if point_data else (nx, ny, nz)
    whole = f"0 {ext[0]} 0 {ext[1]} 0 {ext[2]}"
    tag = "PointData" if point_data else "CellData"

    # --- assemble the appended payload ---------------------------------
    blobs, decls, offset = [], [], 0
    scalar_name = None
    for name, arr in fields.items():
        a = np.asarray(arr)
        if a.dtype not in _VTK_TYPE:
            a = a.astype(np.float32)
        ncomp = 1 if a.ndim == 3 else a.shape[3]
        if ncomp == 1 and scalar_name is None:
            scalar_name = name
        # VTK expects x fastest; our arrays are (x, y, z) C-order, so
        # transpose to (z, y, x) and ravel C-order == x fastest.
        if ncomp == 1:
            flat = np.ascontiguousarray(a.transpose(2, 1, 0)).ravel()
        else:
            flat = np.ascontiguousarray(a.transpose(2, 1, 0, 3)).ravel()
        payload = flat.tobytes()
        blobs.append(_compress_blocks(payload) if compress
                     else struct.pack("<Q", len(payload)) + payload)
        decls.append(
            f'        <DataArray type="{_VTK_TYPE[a.dtype]}" Name="{name}" '
            f'NumberOfComponents="{ncomp}" format="appended" '
            f'offset="{offset}"/>')
        offset += len(blobs[-1])

    sx, sy, sz = spacing
    ox, oy, oz = origin
    header = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64"'
        + (' compressor="vtkZLibDataCompressor"' if compress else '') + '>\n'
        f'  <ImageData WholeExtent="{whole}" Origin="{ox:.9g} {oy:.9g} '
        f'{oz:.9g}" Spacing="{sx:.9g} {sy:.9g} {sz:.9g}">\n'
        f'    <Piece Extent="{whole}">\n'
        f'      <{tag}' + (f' Scalars="{scalar_name}"' if scalar_name else "")
        + '>\n' + "\n".join(decls) + f'\n      </{tag}>\n'
        '    </Piece>\n'
        '  </ImageData>\n'
        '  <AppendedData encoding="raw">\n   _')
    footer = '\n  </AppendedData>\n</VTKFile>\n'

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for b in blobs:
            fh.write(b)
        fh.write(footer.encode("ascii"))
    return path


def write_vtp(path: str, points, scalars: dict | None = None,
              compress: bool = True) -> str:
    """Write per-atom points as VTK XML PolyData (`.vtp`).

    Same appended-binary machinery as `write_vti`, so MD trajectories and
    continuum fields land in ParaView through one code path. Each atom becomes
    a vertex cell, which is what lets ParaView render them as points/spheres
    and colour by any attached scalar.

    Parameters
    ----------
    points : (n, 3) array
        Positions in metres.
    scalars : dict[str, (n,) array]
        Per-atom values -- temperature, ionisation, speed, whatever you want
        to colour by.
    """
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (n, 3), got {pts.shape}")
    n = pts.shape[0]
    scalars = scalars or {}
    for k, v in scalars.items():
        if np.asarray(v).shape[0] != n:
            raise ValueError(
                f"scalar {k!r} has {np.asarray(v).shape[0]} values for "
                f"{n} points")

    def blob(arr):
        payload = np.ascontiguousarray(arr).tobytes()
        return (_compress_blocks(payload) if compress
                else struct.pack("<Q", len(payload)) + payload)

    blobs, decls, offset = [], [], 0

    def add(kind, name, arr, ncomp=1):
        nonlocal offset
        a = np.asarray(arr)
        if a.dtype not in _VTK_TYPE:
            a = a.astype(np.float32)
        b = blob(a)
        blobs.append(b)
        decls.append((kind,
                      f'<DataArray type="{_VTK_TYPE[a.dtype]}" Name="{name}" '
                      f'NumberOfComponents="{ncomp}" format="appended" '
                      f'offset="{offset}"/>'))
        offset += len(b)

    add("points", "Points", pts, 3)
    for name, arr in scalars.items():
        add("point_data", name, arr)
    # one vertex cell per atom: connectivity 0..n-1, offsets 1..n
    add("verts_conn", "connectivity", np.arange(n, dtype=np.int32))
    add("verts_off", "offsets", (np.arange(n, dtype=np.int32) + 1))

    def d(kind):
        return "\n        ".join(x for k, x in decls if k == kind)

    first_scalar = next(iter(scalars), None)
    header = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64"'
        + (' compressor="vtkZLibDataCompressor"' if compress else '') + '>\n'
        '  <PolyData>\n'
        f'    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" '
        'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n'
        '      <Points>\n        ' + d("points") + '\n      </Points>\n'
        '      <Verts>\n        ' + d("verts_conn") + '\n        '
        + d("verts_off") + '\n      </Verts>\n'
        + ('      <PointData'
           + (f' Scalars="{first_scalar}"' if first_scalar else '')
           + '>\n        ' + d("point_data") + '\n      </PointData>\n'
           if scalars else '')
        + '    </Piece>\n'
        '  </PolyData>\n'
        '  <AppendedData encoding="raw">\n   _')
    footer = '\n  </AppendedData>\n</VTKFile>\n'

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for b in blobs:
            fh.write(b)
        fh.write(footer.encode("ascii"))
    return path


def write_pvd(path: str, files, times, comment: str | None = None) -> str:
    """Write a ParaView `.pvd` collection tying the frames to physical times."""
    lines = ['<?xml version="1.0"?>']
    if comment:
        safe = comment.replace("--", "- -")
        lines.append(f"<!--\n{safe}\n-->")
    lines += ['<VTKFile type="Collection" version="0.1" '
              'byte_order="LittleEndian">', "  <Collection>"]
    for f, t in zip(files, times):
        lines.append(f'    <DataSet timestep="{t:.9g}" group="" part="0" '
                     f'file="{os.path.basename(f)}"/>')
    lines += ["  </Collection>", "</VTKFile>", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Sampling the framework state onto a uniform grid
# ---------------------------------------------------------------------------

@dataclass
class ImpactVolume:
    """Samples a `Scenario` onto a uniform grid, frame by frame.

    Geometry: the target surface is the plane z = 0, the target occupies
    z < 0, the projectile arrives along -z and the plume expands into z > 0.
    The grid is centred on the impact point in x and y.

    Parameters
    ----------
    scenario : Scenario
    shape : (nx, ny, nz)
    domain : float, optional
        Half-width in x and y [m].  Default: sized so the plume fills the box
        at the final frame.
    z_below : float, optional
        Depth of target included [m].  Default: one crater radius.
    smoke_velocity_fraction : float
        The condensate leaves at this fraction of the plasma expansion speed.
        It is the marginally-vaporised material, which arrives at the boiling
        point with no superheat, so it is slower and cooler; 0.35 is a
        rendering choice, not a solved quantity.
    """
    scenario: object
    shape: tuple = (128, 128, 128)
    domain: float | None = None
    z_below: float | None = None
    smoke_velocity_fraction: float = 0.35
    crater_growth_exponent: float = 0.5
    notes: list = field(default_factory=list)
    domain_mode: str = "window"
    #: For domain_mode="window": the sampled window ends when the plume has
    #: expanded to this fraction of its final radius, and begins when it is
    #: `window_start` of the window's end radius. Defaults give a plume that
    #: grows from ~8% to ~77% of the box, i.e. a visible expansion.
    window_end: float = 1.0
    window_start: float = 0.10
    #: Minimum plume radius, in cells, in the first sampled frame. The window
    #: start is advanced until this is met, so the animation never opens on a
    #: sub-voxel plume.
    min_start_voxels: float = 2.0
    #: Minimum on-screen growth of the plume across the sampled window.
    #: Takes priority over min_start_voxels when the grid cannot give both.
    min_growth: float = 4.0
    #: ``"lab"`` keeps the impact point at the origin: you see the crater and
    #: the plume flying away downrange, which is the true geometry.
    #: ``"plume"`` follows the plume's centre of mass, so the box has to hold
    #: only the plume's *size* rather than its whole trajectory.
    #:
    #: This matters far more than it sounds. The plume drifts at ~v/2 (25 km/s
    #: for a 50 km/s impact) while expanding at only ~2.5 km/s, so over a
    #: 14 microsecond window it travels 0.25 m and grows to 0.035 m. A lab-frame
    #: box must be ~0.5 m across to contain the trajectory, and the object
    #: inside it is 14x smaller: measured on the reference case, the plume
    #: occupies **1.9% of the voxels** at the last frame and 0.003% at the
    #: first. That is what makes a rendered plume look like a small ball that
    #: barely changes -- almost the whole grid is empty space it flies
    #: through.
    #:
    #: In ``"plume"`` mode the target, crater and shock are not drawn: they
    #: are hundreds of plume radii behind and would be meaningless.
    frame_of_reference: str = "lab"
    #: Floor for the log10 density fields [kg/m^3] and [m^-3]. Everything
    #: below becomes the floor value, which bounds the colour range to
    #: something a display can resolve instead of the ~36 decades a Gaussian
    #: tail reaches before it underflows.
    log_floor: float = 1e-12
    log_floor_ne: float = 1e12

    def __post_init__(self):
        sc = self.scenario
        if sc.expansion is None:
            raise ValueError(
                "this scenario produced no plume (below the vaporisation "
                "threshold), so there is nothing to visualise")
        if self.frame_of_reference not in ("lab", "plume"):
            raise ValueError(
                f"frame_of_reference must be 'lab' or 'plume', got "
                f"{self.frame_of_reference!r}")
        exp = sc.expansion
        self.t_end = float(exp.t[-1])
        # Two different extents, and using the wrong one is the whole bug.
        #   R_final    -- how far the plume GETS (size + drift): what a
        #                 lab-frame box must contain.
        #   R_size     -- how big the plume IS: what a co-moving box needs.
        self.R_final = float(max(exp.R_r[-1], exp.R_z[-1]) + exp.z_cm[-1])
        self.R_size = float(max(exp.R_r[-1], exp.R_z[-1]))
        # radius the box is sized to in `window` mode
        self.R_window = self.R_final
        if self.domain_mode == "window":
            self.R_window = self._window_times()[2]
        self.crater_radius = self._crater_radius_final()
        # Mass fraction of the vapour that came from the projectile. One
        # number for the whole plume -- see `projectile_fraction` in `frame`.
        self._w_projectile = float(
            self.scenario.impact.diagnostics.get("w_projectile", 0.0))
        # Whether the caller chose the box themselves. `domain_at` must then
        # leave it alone: before this flag existed, `domain_mode="fixed"`
        # returned 1.3*R_final regardless, so an explicit `domain=` was
        # silently discarded on the first `frame()` call and every
        # impact-scale render came back plume-sized.
        self._domain_is_explicit = self.domain is not None
        self._z_below_is_explicit = self.z_below is not None
        if self.domain is None:
            self.domain = 1.3 * (self.R_size
                                 if self.frame_of_reference == "plume"
                                 else self.R_final)
        if self.z_below is None:
            # Co-moving: no target in view, so make the box symmetric about
            # the plume. Lab: just enough depth to show the crater.
            self.z_below = (self.domain if self.frame_of_reference == "plume"
                            else min(2.0 * self.crater_radius,
                                     0.5 * self.domain))
        if self.domain_mode == "adaptive":
            # Say this loudly. `adaptive` resizes the box to the plume every
            # frame, so the plume occupies the SAME FRACTION OF THE IMAGE at
            # every timestep -- measured at R/box = 0.068 from the first
            # frame to the last while the plume itself grew 3.5x. The
            # animation is then genuinely unchanging, which reads as "the
            # simulation is not doing anything" when in fact only the
            # camera is following it.
            #
            # It is still the right mode for inspecting the plume's internal
            # structure at every scale, which is why it is kept.
            _note_once(
                self.notes,
                "domain_mode='adaptive' rescales the box with the plume, so "
                "the plume's apparent size is CONSTANT in every frame and "
                "the animation shows no expansion. Use 'window' (the "
                "default) to see it grow.")
        self._build_grid(self.domain, self.z_below)

    def _build_grid(self, domain: float, z_below: float) -> None:
        nx, ny, nz = self.shape
        self.domain, self.z_below = domain, z_below
        self.spacing = (2 * domain / (nx - 1), 2 * domain / (ny - 1),
                        (domain + z_below) / (nz - 1))
        self.origin = (-domain, -domain, -z_below)
        x = np.linspace(-domain, domain, nx)
        y = np.linspace(-domain, domain, ny)
        z = np.linspace(-z_below, domain, nz)
        # Coordinates as broadcast views, not five dense copies.
        #
        # A dense meshgrid stores X, Y and Z at full resolution and then Rxy
        # and Rsph on top: 5 x nx x ny x nz doubles. At 256^3 that is 671 MB
        # of coordinates before a single field is sampled, and every field
        # expression streams all of it through cache.
        #
        # Four of the five do not need to be materialised. X varies only
        # along x, Y only along y, Z only along z, and Rxy is independent of
        # z entirely. Computing them sparse and then `broadcast_to` the full
        # shape gives arrays that *report* shape (nx, ny, nz) and index like
        # dense ones, while sharing the small buffer -- `broadcast_to`
        # returns a view, so this costs nothing.
        #
        # Only Rsph genuinely varies in all three. 671 MB -> 134 MB, and the
        # sampler stops being memory-bandwidth-bound.
        #
        # The views are read-only, which is correct here (nothing writes to a
        # coordinate array) and would catch it loudly if anything tried. The
        # public shape of self.X/Y/Z/Rxy is deliberately unchanged: callers
        # and tests index them as full 3-D arrays, and quietly changing that
        # would be the same class of contract break as returning a different
        # meaning for the same field.
        self.grid_shape = (nx, ny, nz)
        Xs, Ys, Zs = np.meshgrid(x, y, z, indexing="ij", sparse=True)
        full = self.grid_shape
        self.X = np.broadcast_to(Xs, full)
        self.Y = np.broadcast_to(Ys, full)
        self.Z = np.broadcast_to(Zs, full)
        self.Rxy = np.broadcast_to(np.hypot(Xs, Ys), full)
        self.Rsph = np.sqrt(Xs**2 + Ys**2 + Zs**2)

    def _window_times(self) -> tuple:
        """(t_start, t_end, R_end) for the `window` mode.

        The whole expansion spans ~5 decades in radius. No fixed box can show
        that, which is why the old adaptive default silently turned an
        expansion into a fixed-size ball. Instead pick a sub-range of the
        history in which the plume grows by ~10x, and size the box to the end
        of it.
        """
        exp = self.scenario.expansion
        # Use the same extent measure the box is sized by: the plume's own
        # radius PLUS the centre-of-mass drift. Sizing on radius alone let the
        # plume translate out of a correctly-sized box at late times, so only
        # its tail was sampled and the sub-voxel fallback fired -- the movie
        # went from a resolved ball to a single bright voxel.
        R = (np.maximum(np.asarray(exp.R_r, float), np.asarray(exp.R_z, float))
             + np.asarray(exp.z_cm, float))
        t = np.asarray(exp.t, float)
        R_end = float(self.window_end * R[-1])
        R_beg = float(self.window_start * R_end)
        # R(t) is monotonic for a freely expanding plume, so searchsorted is
        # safe and exact at the sample points.
        i0 = int(np.searchsorted(R, R_beg))
        i1 = int(np.searchsorted(R, R_end))
        i0 = min(max(i0, 0), len(t) - 2)
        i1 = min(max(i1, i0 + 1), len(t) - 1)

        # window_start is only a starting guess. What actually matters is that
        # the plume is resolved in the FIRST frame -- otherwise the animation
        # opens on the single-voxel fallback and appears to pop into
        # existence. The plume's *radius* is what has to span cells, which is
        # smaller than the extent measure above whenever the cloud drifts, so
        # advance the start until the radius covers `min_start_voxels` cells.
        radius = np.sqrt(np.asarray(exp.R_r, float)
                         * np.asarray(exp.R_z, float))
        dx = 2.0 * (1.3 * R[i1]) / min(self.shape)
        need = self.min_start_voxels * dx
        i_start = i0
        while i0 < i1 - 1 and radius[i0] < need:
            i0 += 1
        # ...but never at the cost of the expansion itself. On a coarse grid
        # the two requirements conflict: satisfying the first frame can push
        # the start so late that the plume already fills the box. Growth is
        # the point of the animation, so it wins, and we say the grid is too
        # coarse rather than silently returning a static ball.
        if radius[i1] < self.min_growth * max(radius[i0], 1e-300):
            i0 = i_start
            _note_once(
                self.notes,
                f"opening frame has a {radius[i0] / dx:.1f}-cell plume. The "
                "cloud drifts several radii downrange, so a box that keeps "
                "the impact site in view is necessarily much larger than the "
                "plume early on. Growth is preserved (that is the point of "
                "the animation); the first frame or two will be small. Raise "
                "`shape`, or set window_start higher to open later.")
        return float(t[i0]), float(t[i1]), float(R[i1])

    def _plume_size_at(self, t: float) -> float:
        """Plume radius at `t` [m] -- size only, no centre-of-mass drift.

        The co-moving box is sized from the *window end* rather than from
        `t`, so the box stays fixed and the plume visibly grows inside it.
        A box that resized every frame would hold the plume at a constant
        apparent size and hide the very thing being animated.
        """
        exp = self.scenario.expansion
        if self.domain_mode == "window":
            t_ref = self._window_times()[1]
        else:
            t_ref = float(exp.t[-1])
        tt = np.clip(t_ref, exp.t[0], exp.t[-1])
        return float(max(np.interp(tt, exp.t, exp.R_r),
                         np.interp(tt, exp.t, exp.R_z)))

    def domain_at(self, t: float) -> tuple:
        """Domain half-width and target depth for this frame.

        Three modes, and the difference is the difference between an
        animation that shows expansion and one that does not:

        ``window`` (default)
            Fixed box, sized to the plume at the *end of the sampled window*
            rather than at the end of the whole history.  Combined with
            ``times(mode="window")`` the plume grows from a small fraction of
            the box to most of it, which is what "watch the plume expand"
            means.  Use this for animations.

        ``adaptive``
            The box tracks the plume, so all five decades of expansion are
            visible in one series -- but the plume then occupies a *constant*
            fraction of the box (~7%), so on screen it is a ball of fixed
            size and the expansion is invisible.  It is a zoom-out, not an
            expansion.  Useful for inspecting any single epoch; misleading as
            a movie.

        ``fixed``
            Box sized to the final plume over the whole history.  Physically
            honest and useless to look at: the plume is under one voxel
            across for ~90% of the frames.

        ParaView handles varying Origin/Spacing across a .pvd without
        complaint, which is why ``adaptive`` works at all.
        """
        # In the co-moving frame the box travels with the plume, so it needs
        # to hold the plume's SIZE, not its trajectory. Sizing on the
        # trajectory is what left the plume occupying under 2% of the voxels
        # and rendering as a small ball in a mostly-empty box.
        # An explicitly requested box wins over every mode below. This is
        # what makes the "impact" scene possible at all.
        if self._domain_is_explicit:
            return self.domain, (self.z_below if self._z_below_is_explicit
                                 else min(2.0 * self.crater_radius,
                                          0.65 * self.domain))
        if self.frame_of_reference == "plume":
            half = PLUME_BOX_SIGMA * self._plume_size_at(t)
            return half, half
        if self.domain_mode == "fixed":
            return 1.3 * self.R_final, min(2.0 * self.crater_radius,
                                           0.65 * self.R_final)
        if self.domain_mode == "window":
            half = 1.3 * self.R_window
            return half, min(2.0 * self.crater_radius, 0.65 * half)
        exp = self.scenario.expansion
        tt = np.clip(max(t, exp.t[0]), exp.t[0], exp.t[-1])
        R = float(max(np.interp(tt, exp.t, exp.R_r),
                      np.interp(tt, exp.t, exp.R_z))
                  + np.interp(tt, exp.t, exp.z_cm))
        # never smaller than the crater / projectile, so the impact site
        # stays resolved in the earliest frames
        R = max(R, 3.0 * self.crater_radius,
                4.0 * self.scenario.impact.projectile.radius)
        if t <= 0.0:
            # before contact there is no plume; frame the projectile and the
            # impact site instead, or the sphere is smaller than a cell
            R = max(6.0 * self.scenario.impact.projectile.radius,
                    1.5 * self.crater_radius)
        z_below = min(2.0 * self.crater_radius, 0.65 * R)
        return 1.6 * R, z_below

    # -- geometry helpers -------------------------------------------------
    def _crater_radius_final(self) -> float:
        """Cour-Palais penetration correlation; illustrative only."""
        imp = self.scenario.impact
        rho_p = imp.projectile.material.rho0
        rho_t, c_t, BH = imp.target.rho0, 5100.0, 30.0
        d = 2.0 * imp.projectile.radius
        P_over_d = (5.24 * (d * 100.0) ** (1 / 19) * BH ** -0.25
                    * (rho_p / rho_t) ** 0.5
                    * (imp.projectile.v_normal / c_t) ** (2 / 3))
        return float(P_over_d * d)          # radius ~ penetration depth

    def crater_radius_at(self, t: float) -> float:
        if t <= 0:
            return 0.0
        t_grow = self.crater_radius / 5100.0
        f = min(1.0, (t / t_grow) ** self.crater_growth_exponent)
        return f * self.crater_radius

    # -- field construction ----------------------------------------------
    def _plume_fields(self, t: float) -> dict:
        """Gaussian ellipsoid from the Stage-2 solution, interpolated to t."""
        exp = self.scenario.expansion
        tt = np.clip(t, exp.t[0], exp.t[-1])

        def ip(arr):
            return float(np.interp(tt, exp.t, arr))

        Rr, Rz = max(ip(exp.R_r), 1e-9), max(ip(exp.R_z), 1e-9)
        # In the co-moving frame the box travels with the plume, so the
        # plume sits at the box centre for every frame and the animation
        # shows expansion rather than translation. `zc_lab` is kept because
        # the target surface is still a real boundary and it has to be
        # placed correctly in whichever frame we are drawing.
        zc_lab = ip(exp.z_cm)
        zc = 0.0 if self.frame_of_reference == "plume" else zc_lab
        n_pk = ip(exp.n_h)
        T_eV = ip(exp.T_eV)
        Zbar = ip(exp.Zbar)
        vr, vz = ip(exp.v_r), ip(exp.v_z)
        m_atom = exp.material.m_atom

        g = np.exp(-0.5 * ((self.Rxy / Rr) ** 2
                           + ((self.Z - zc) / Rz) ** 2))
        # Only above the target surface. The plume is a hemisphere released
        # from a plane and it stays one: a radially expanding cloud has no
        # velocity through the plane containing its centre, so nothing ever
        # crosses into the lower half.
        #
        # In the co-moving frame the same surface is at z = -z_cm, because
        # the box has travelled z_cm with the plume. Clipping at z = 0 there
        # would delete the plume's trailing half; not clipping at all would
        # draw material behind the target surface at early times. Once
        # z_cm >> R_z the clip is a no-op, which is why the plume looks
        # symmetric at late times.
        z_surface = 0.0 if self.frame_of_reference == "lab" else -zc_lab
        g = np.where(self.Z > z_surface, g, 0.0)

        # A plume smaller than one cell falls between grid points, so the
        # best-sampled value is far out on the Gaussian tail -- it underflows
        # float32 and renders as "no plasma at all" rather than "too small to
        # see". If the peak sampled value is below 0.1% of the true peak the
        # grid simply does not resolve the plume, so deposit it in the nearest
        # voxel above the surface: wrong shape, right existence. Same fallback
        # the sub-cell projectile uses.
        if g.max() < 1e-3 and n_pk > 0.0:
            above = self.Z > 0.0
            if above.any():
                d = ((self.Rxy) ** 2 + (self.Z - max(zc, 0.0)) ** 2)
                d = np.where(above, d, np.inf)
                g = np.zeros_like(g)
                g.flat[int(np.argmin(d))] = 1.0
        n_h = n_pk * g
        rho = n_h * m_atom
        n_e = n_h * Zbar
        T = T_eV * np.where(g > 1e-6, 1.0, 0.0)
        P = n_h * (1.0 + Zbar) * K_B * (T_eV * EV / K_B)

        # self-similar velocity: v = (r/R) * Rdot, plus the c.o.m. drift
        with np.errstate(invalid="ignore", divide="ignore"):
            vx = np.where(Rr > 0, self.X / Rr * vr, 0.0)
            vy = np.where(Rr > 0, self.Y / Rr * vr, 0.0)
            vzf = np.where(Rz > 0, (self.Z - zc) / Rz * vz, 0.0) + vz
        mask = g > 1e-6
        vel = np.stack([vx * mask, vy * mask, vzf * mask], axis=-1)
        return {"rho": rho, "n_e": n_e, "T_eV": T, "Zbar": Zbar * g,
                "P": P, "vel": vel, "g": g}

    def _smoke_field(self, t: float) -> np.ndarray:
        """Condensate/dust: the two-phase inventory, slower and cooler.

        Parametric.  The two-phase material leaves at the boiling point with
        no superheat, so it expands at a fraction of the plasma speed; that
        fraction is `smoke_velocity_fraction`, a rendering choice.
        """
        imp = self.scenario.impact
        m_smoke = float(imp.diagnostics.get("m_twophase", 0.0))
        if m_smoke <= 0 or t <= 0:
            return np.zeros(self.grid_shape)
        f = self.smoke_velocity_fraction
        exp = self.scenario.expansion
        tt = np.clip(t, exp.t[0], exp.t[-1])
        Rr = max(float(np.interp(tt, exp.t, exp.R_r)) * f, 1e-9)
        Rz = max(float(np.interp(tt, exp.t, exp.R_z)) * f, 1e-9)
        zc = float(np.interp(tt, exp.t, exp.z_cm)) * f
        g = np.exp(-0.5 * ((self.Rxy / Rr) ** 2 + ((self.Z - zc) / Rz) ** 2))
        g = np.where(self.Z > 0.0, g, 0.0)
        vol = (2.0 * np.pi) ** 1.5 * Rr * Rr * Rz
        return (m_smoke / vol) * g

    def _target_and_shock(self, t: float):
        """Target solid, crater cavity and the parametric shock front."""
        imp = self.scenario.impact
        solid = (self.Z < 0.0)
        r_c = self.crater_radius_at(t)
        crater = solid & (self.Rsph < r_c)
        solid_frac = np.where(solid & ~crater, 1.0, 0.0)

        # shock front: hemispherical, at Us*t, with the model's decay law
        shock = np.zeros(self.grid_shape)
        if t > 0:
            Us = float(imp.state.target["Us"]) if imp.state else 6000.0
            r_ic = max(imp.r_ic, 1e-9)
            r_front = Us * t
            band = solid & (self.Rsph <= r_front) & (self.Rsph >= r_ic)
            n_decay = float(imp.diagnostics.get("n_decay", 2.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                shock = np.where(
                    band, imp.P_ic * (self.Rsph / r_ic) ** (-n_decay), 0.0)
            core = solid & (self.Rsph < r_ic) & (self.Rsph <= r_front)
            shock = np.where(core, imp.P_ic, shock)
        return solid_frac, crater, np.nan_to_num(shock)

    def _projectile(self, t: float) -> np.ndarray:
        """Rigid sphere approaching the surface; illustrative, pre-impact only."""
        imp = self.scenario.impact
        if t >= 0:
            return np.zeros(self.grid_shape)
        r = imp.projectile.radius
        zc = -imp.projectile.v_normal * t + r      # t<0 -> above the surface
        d2 = self.Rxy**2 + (self.Z - zc) ** 2
        out = (d2 < r * r).astype(np.float32)
        if out.max() == 0.0:
            # sub-cell projectile: mark the nearest voxel so the approach is
            # visible at any resolution rather than silently vanishing
            out.flat[int(np.argmin(d2))] = 1.0
        return out

    def frame(self, t: float) -> dict:
        """All fields at time `t` (t < 0 is before contact)."""
        d, zb = self.domain_at(t)
        if abs(d - self.domain) > 1e-12 or abs(zb - self.z_below) > 1e-12:
            self._build_grid(d, zb)
        pl = self._plume_fields(t) if t > 0 else None
        smoke = self._smoke_field(t)
        solid_frac, crater, shock = self._target_and_shock(t)
        proj = self._projectile(t)

        zeros = np.zeros(self.grid_shape, dtype=np.float32)
        rho_p = pl["rho"] if pl else zeros
        rho_total = rho_p + smoke + solid_frac * self.scenario.impact.target.rho0

        material = np.zeros(self.grid_shape, dtype=np.uint8)
        material[solid_frac > 0.5] = 1
        material[crater] = 2
        if pl is not None:
            material[pl["g"] > 1e-3] = 3
        material[(smoke > 0) & (material == 0)] = 4
        material[proj > 0.5] = 5

        if self.frame_of_reference == "plume":
            # The target is hundreds of plume radii behind the co-moving box.
            # Drawing it at the box origin would put a slab of aluminium
            # through the middle of the plume.
            rho_total = rho_p
            material = np.zeros(self.grid_shape, dtype=np.uint8)
            if pl is not None:
                material[pl["g"] > 1e-3] = 3
            shock = zeros
            proj = zeros

        f = {
            "density": rho_total.astype(np.float32),
            "plasma_density": rho_p.astype(np.float32),
            # log10 of the density, floored. A Gaussian plume spans ~36
            # decades inside one frame and its peak falls ~2 decades across
            # the series, so a LINEAR colour map shows a uniform blob at the
            # bottom of the range and nothing else -- which is exactly what a
            # default ParaView render of `plasma_density` looks like.
            # Colour by this field instead and a linear map becomes a log
            # map, with the structure visible and the range stable in time.
            "log10_plasma_density": _log10_floor(
                rho_p, self.log_floor).astype(np.float32),
            "log10_electron_density": _log10_floor(
                pl["n_e"] if pl else zeros, self.log_floor_ne).astype(
                    np.float32),
            "smoke_density": smoke.astype(np.float32),
            "temperature_eV": (pl["T_eV"] if pl else zeros).astype(np.float32),
            "electron_density": (pl["n_e"] if pl else zeros).astype(np.float32),
            "ionisation": (pl["Zbar"] if pl else zeros).astype(np.float32),
            "pressure": ((pl["P"] if pl else zeros) + shock).astype(np.float32),
            "shock_pressure": shock.astype(np.float32),
            "velocity": (pl["vel"] if pl else
                         np.zeros(self.grid_shape + (3,))).astype(np.float32),
            "material": material,
            "projectile": proj.astype(np.float32),
            # Uniform BY CONSTRUCTION, and named so that is unmissable.
            #
            # Stage 1 computes how much of the vapour came from the
            # projectile and how much from the target (w_projectile here),
            # but it mixes them into ONE effective material with one
            # temperature and one charge state. There is no "projectile
            # material here, target material there" in this model, so this
            # field is a constant wherever there is plume.
            #
            # Rendering it is still worth doing: a viewer who sees a flat
            # colour learns the model's structure immediately, whereas a
            # viewer shown no field at all is left to infer it from the
            # shape -- and may conclude the uniformity is a result.
            "projectile_fraction": (
                np.where(rho_p > 0.0, self._w_projectile, 0.0)
                .astype(np.float32)),
        }
        return f

    def shock_transit_time(self) -> float:
        """When the shock front leaves the box [s].

        Past this the front is outside the domain and `shock_pressure` shows
        only the residual decay-law field, which is constant in time and reads
        as a static halo. It is the natural end of an impact-scale animation.
        """
        imp = self.scenario.impact
        Us = float(imp.state.target["Us"]) if imp.state else 6000.0
        return float(np.hypot(self.domain, self.z_below) / max(Us, 1.0))

    def crater_growth_time(self) -> float:
        """When the crater stops growing [s] (see `crater_radius_at`)."""
        return float(self.crater_radius / 5100.0)

    def plume_departure_time(self) -> float:
        """When the plume's centre of mass leaves the box [s].

        **This is usually the binding constraint on an impact-scale
        animation, and it is much earlier than intuition suggests.** The plume
        drifts downrange at roughly half the impact speed while expanding at
        only a few km/s -- measured at 25.0 against 3.1 km/s for the reference
        case, a ratio of 8. So it does not sit over the crater growing; it
        *leaves*, and a box sized to show a 74 um crater is emptied in
        nanoseconds.

        Running an animation past this point renders 80% of its frames with
        nothing in the domain, which reads as "the simulation stopped".
        """
        exp = self.scenario.expansion
        if exp is None:                             # pragma: no cover
            return float("inf")
        t = np.asarray(exp.t, dtype=float)
        z = np.asarray(exp.z_cm, dtype=float)
        beyond = np.flatnonzero(z > self.domain)
        if beyond.size == 0:
            return float(t[-1])
        return float(t[beyond[0]])

    def times(self, n_frames: int = 60, n_pre: int | None = None,
              mode: str | None = None,
              t_end: float | None = None) -> np.ndarray:
        """Frame times: a few before contact, then through the expansion.

        Strictly increasing and starting before t = 0, so ParaView's timeline
        runs forward through approach, impact and expansion.

        `t_end` truncates the post-impact range. Without it the series always
        runs to the end of the Stage-2 expansion, which for an impact-scale
        box means most frames are spent after the shock has left the domain
        and the crater has stopped growing -- a long static tail.
        """
        if n_frames < 2:
            raise ValueError("n_frames must be >= 2")
        exp = self.scenario.expansion
        t0, t1 = float(exp.t[0]), float(exp.t[-1])
        if t_end is not None:
            if t_end <= 0:
                raise ValueError("t_end must be positive")
            t1 = min(t1, float(t_end))

        # scale the pre-impact frames with the request; always leave >= 1
        # post-impact frame even for a 2-frame series
        if n_pre is None:
            n_pre = max(1, min(5, n_frames // 5))
        n_pre = int(np.clip(n_pre, 1, n_frames - 1))
        n_post = n_frames - n_pre

        if mode == "window" or (mode is None and self.domain_mode == "window"):
            # Sample only the sub-range the fixed box was sized for, and do it
            # *linearly* in time: the late plume expands at constant speed, so
            # linear sampling gives a ball growing at a constant rate on
            # screen. Log sampling over five decades is what made the old
            # default look static.
            w0, w1, _ = self._window_times()
            w1 = min(w1, t1)                      # honour t_end here too
            w0 = max(w0, t1 * 1e-9)
            if not w0 < w1:
                # t_end landed before or inside the window: fall back to a
                # plain linear ramp rather than emitting a degenerate range.
                post = np.linspace(t1 / n_post, t1, n_post)
            else:
                post = np.linspace(w0, w1, n_post)
        elif mode == "log":
            post = np.geomspace(max(t0, t1 * 1e-6), t1, n_post)
        else:
            post = np.linspace(t1 / n_post, t1, n_post)

        # approach: from -approach up to (but not including) 0, ascending
        approach = (4.0 * self.scenario.impact.projectile.radius
                    / max(self.scenario.impact.projectile.v_normal, 1.0))
        pre = np.linspace(-approach, 0.0, n_pre, endpoint=False)
        t = np.concatenate([pre, post])
        assert np.all(np.diff(t) > 0), "frame times must increase"
        if t_end is not None:
            assert t[-1] <= t1 * (1.0 + 1e-12), (
                f"t_end={t_end:g} requested but series ends at {t[-1]:g}")
        return t


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

#: A compact default: the fields most people actually colour by. Pass
#: ``fields="all"`` for everything in FIELD_PROVENANCE.
#: `log10_plasma_density` is included and is the one to colour by: the raw
#: density spans far too many decades for a linear colour map, which is why a
#: default render of `plasma_density` is a flat blob.
DEFAULT_FIELDS = ("density", "plasma_density", "log10_plasma_density",
                  "smoke_density", "temperature_eV", "electron_density",
                  "log10_electron_density", "material")


# ---------------------------------------------------------------------------
# Scenes: one domain cannot show both the crater and the plume
# ---------------------------------------------------------------------------

#: Scene presets.
#:
#: **Why this exists.** The default domain is sized for the plume, and on that
#: domain the crater is *0.013 cells across* — a seventy-eighth of one voxel —
#: and the projectile is 1/1800th. They are not small on screen, they are
#: absent. The shock front does not reach even one cell until ~1 microsecond,
#: by which time the crater has long since stopped growing. Any render of the
#: default domain is therefore a picture of the plume and nothing else, which
#: is exactly why it looks like a featureless ball.
#:
#: The numbers, for the reference case (Fe, 1 pg, 50 km/s -> Al, 128^3):
#:
#: ===========  ==============  =============  ==============
#: scene        cell size       crater         shock resolved
#: ===========  ==============  =============  ==============
#: ``impact``   3.5 um          21 cells       from 0.6 ns
#: ``plume``    5.7 mm          0.013 cells    from 960 ns
#: ===========  ==============  =============  ==============
#:
#: Four orders of magnitude separate them. They are two different films of
#: two different events, and trying to make one serve both is what produced
#: the static blob. Render both and cut between them.
SCENES = {
    "impact": {
        "what": "projectile, shock, crater growth and plasma birth",
        "when": "first ~100 ns",
        "domain_radii": 3.0,          # in final crater radii
        "z_below_frac": 0.8,          # of the domain half-width
        "time_mode": "log",
        "frame_of_reference": "lab",
        "colour_by": "log10_plasma_density",
    },
    "plume": {
        "what": "the plasma plume expanding and going collisionless",
        "when": "~1 to 100 microseconds",
        "domain_radii": None,         # let ImpactVolume size it for the plume
        "z_below_frac": None,
        "time_mode": None,            # the window sampler
        "frame_of_reference": "lab",
        "colour_by": "log10_plasma_density",
    },
    "plume_comoving": {
        "what": "the plume only, with the camera riding its centre of mass",
        "when": "~1 to 100 microseconds",
        "domain_radii": None,
        "z_below_frac": None,
        "time_mode": None,
        "frame_of_reference": "plume",
        "colour_by": "log10_plasma_density",
    },
}


def scene_kwargs(scenario, scene: str = "impact") -> dict:
    """`ImpactVolume` keyword arguments for a named scene.

    Returns a dict ready to splat into `ImpactVolume` or `write_scene`.
    `scene_kwargs(sc, "impact")["domain"]` is the number that makes the crater
    visible.
    """
    if scene not in SCENES:
        raise KeyError(f"unknown scene {scene!r}; have {sorted(SCENES)}")
    spec = SCENES[scene]
    kw: dict = {"frame_of_reference": spec["frame_of_reference"]}
    if spec["domain_radii"] is not None:
        # Size the box on the crater, not the plume. crater_radius is
        # computed in ImpactVolume.__post_init__, so build a throwaway
        # (8^3, negligible) to ask it.
        probe = ImpactVolume(scenario, shape=(8, 8, 8))
        R = probe.crater_radius
        kw["domain"] = spec["domain_radii"] * R
        kw["z_below"] = spec["domain_radii"] * R * spec["z_below_frac"]
        kw["domain_mode"] = "fixed"
    return kw


#: Fields worth writing for an impact-scale scene. `plume` scenes do not need
#: the target fields, and vice versa, but the cost of carrying both is small
#: next to the confusion of a missing array when ParaView tries to colour by
#: it.
IMPACT_FIELDS = ("log10_plasma_density", "plasma_density", "temperature_eV",
                 "ionisation", "shock_pressure", "material", "projectile",
                 "projectile_fraction", "density")

#: The minimum that renders: one array to colour by, one to draw the target
#: surface from. Worth knowing about at high resolution -- you can only
#: volume-render one array at a time anyway.
MINIMAL_FIELDS = ("log10_plasma_density", "material")

#: Peak resident memory per grid cell while sampling one frame, in bytes.
#:
#: **Measured, not estimated.** `ImpactVolume.frame` builds a stack of float64
#: intermediates, so peak RSS is far above the float32 output. Measured on the
#: reference case across 96^3 to 224^3 it settles at ~175 B/cell -- about 44x
#: a single float32 field:
#:
#: ====== ============ ==============
#: grid   peak RSS     bytes per cell
#: ====== ============ ==============
#: 96^3   0.25 GB      283
#: 128^3  0.47 GB      226
#: 160^3  0.84 GB      206
#: 192^3  1.29 GB      182
#: 224^3  1.98 GB      176
#: ====== ============ ==============
#:
#: The trend is still falling slowly, so this over-estimates a little at
#: large N, which is the direction an OOM guard should err in.
BYTES_PER_CELL_PEAK = 175.0

#: Grid presets. The limit is host RAM during *generation*, not the GPU: a
#: volume renderer uploads one array (512^3 float32 = 0.54 GB, comfortable on
#: any modern card), while the sampler needs ~44x that.
QUALITY = {
    "draft": {"n": 96,
              "for": "is the camera pointing the right way?"},
    "talk": {"n": 256,
             "for": "projector resolution; the default worth using"},
    "high": {"n": 384,
             "for": "close-ups of the crater rim"},
    "workstation": {"n": 512,
                    "for": "a 64 GB+ machine; near the point of diminishing "
                           "returns for a 1920x1080 frame"},
    "extreme": {"n": 768,
                "for": "needs ~128 GB; only worth it for a still, not a "
                       "series"},
}

# Derived, not typed: a hand-written RAM column drifts from the estimator the
# moment BYTES_PER_CELL_PEAK is re-measured.
for _spec in QUALITY.values():
    _spec["ram_GB"] = _spec["n"] ** 3 * BYTES_PER_CELL_PEAK / 1e9
del _spec


def estimate_memory_GB(shape, n_fields: int = len(IMPACT_FIELDS)) -> dict:
    """Peak host RAM to sample one frame, and the bytes it will hold.

    Returns ``{"cells", "peak_GB", "output_GB", "vram_GB"}``. `vram_GB` is
    one float32 array -- what a volume renderer actually uploads -- which is
    invariably the smallest number of the three and is why the GPU is not the
    constraint here.
    """
    cells = float(shape[0]) * float(shape[1]) * float(shape[2])
    return {
        "cells": cells,
        "peak_GB": cells * BYTES_PER_CELL_PEAK / 1e9,
        "output_GB": cells * 4.0 * n_fields / 1e9,
        "vram_GB": cells * 4.0 / 1e9,
    }


def _available_memory_GB() -> float | None:
    """Host RAM available now [GB], or None if it cannot be determined."""
    try:                                     # Linux
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1e6
    except OSError:
        pass
    try:                                     # POSIX fallback
        return (os.sysconf("SC_AVPHYS_PAGES")
                * os.sysconf("SC_PAGE_SIZE") / 1e9)
    except (ValueError, OSError, AttributeError):
        return None


def write_scene(scenario, directory: str, scene: str = "impact",
                n_frames: int = 60, shape: tuple | None = None,
                fields=None, name: str | None = None,
                quality: str | None = None, allow_oversize: bool = False,
                verbose: bool = True, **overrides) -> dict:
    """Write one named scene: domain, time window and fields all consistent.

    This is the entry point to prefer over `write_impact_series`, which sizes
    everything for the plume and therefore cannot show a crater -- see
    `SCENES` for the arithmetic.

    Parameters
    ----------
    scene : {"impact", "plume", "plume_comoving"}
    shape : (nx, ny, nz) or None
        ``None`` takes the grid from `quality`.
    quality : str or None
        A key of `QUALITY` -- "draft", "talk", "high", "workstation",
        "extreme". Ignored when `shape` is given. Defaults to "talk".
    allow_oversize : bool
        Generate even when the estimate exceeds available RAM. The estimate
        is ~175 B/cell (measured), and being wrong about it means the process
        is killed partway through a series, so the default is to refuse.
    overrides
        Passed through to `ImpactVolume`, so `domain=`, `z_below=` and the
        rest can still be set by hand.

    Returns
    -------
    The dict from `write_impact_series`, plus ``scene``, ``t_end``,
    ``cells_across_crater`` -- the number this whole mechanism exists to keep
    above 1 -- and ``memory``.
    """
    spec = SCENES[scene] if scene in SCENES else None
    if spec is None:
        raise KeyError(f"unknown scene {scene!r}; have {sorted(SCENES)}")

    if shape is None:
        q = quality or "talk"
        if q not in QUALITY:
            raise KeyError(f"unknown quality {q!r}; have {sorted(QUALITY)}")
        n = QUALITY[q]["n"]
        shape = (n, n, n)
    want_fields = IMPACT_FIELDS if fields is None else tuple(fields)
    mem = estimate_memory_GB(shape, len(want_fields))
    avail = _available_memory_GB()
    if (avail is not None and mem["peak_GB"] > avail and not allow_oversize):
        raise MemoryError(
            f"sampling {shape[0]}x{shape[1]}x{shape[2]} needs about "
            f"{mem['peak_GB']:.1f} GB peak and only {avail:.1f} GB is "
            f"available.\n"
            f"  The estimate is {BYTES_PER_CELL_PEAK:.0f} B/cell, measured "
            f"on this sampler.\n"
            f"  Options: a smaller grid (quality='talk' is "
            f"{QUALITY['talk']['n']}^3, {QUALITY['talk']['ram_GB']:.1f} GB), "
            f"fewer fields (MINIMAL_FIELDS), or allow_oversize=True if you "
            f"believe the estimate is pessimistic here.")

    kw = scene_kwargs(scenario, scene)
    kw.update(overrides)
    vol = ImpactVolume(scenario, shape=shape, **kw)

    t_end = None
    if scene == "impact":
        # Long enough for the shock to cross and the crater to finish, but
        # NOT past the moment the plume leaves the box. The first version of
        # this ran to 3x the shock transit, which for the reference case was
        # 43.6 ns against a plume departure at 8.9 ns -- so 80% of the frames
        # had an empty domain, which looks exactly like a simulation that has
        # stopped.
        t_shock = 3.0 * max(vol.shock_transit_time(), vol.crater_growth_time())
        t_depart = vol.plume_departure_time()
        t_end = min(t_shock, t_depart)
        if t_depart < t_shock:
            _note_once(vol.notes,
                       f"scene ends at {t_end:.3e} s, when the plume's centre "
                       f"of mass leaves the box -- not at the {t_shock:.3e} s "
                       f"the shock and crater would need. The plume drifts "
                       f"downrange ~8x faster than it expands, so an "
                       f"impact-scale box empties long before the crater "
                       f"story finishes. Use scene='plume' to follow it.")

    times = vol.times(n_frames, mode=spec["time_mode"], t_end=t_end)
    cells = vol.crater_radius / vol.spacing[0]
    if scene == "impact" and cells < 1.0:        # pragma: no cover - guard
        _note_once(vol.notes,
                   f"crater is {cells:.3g} cells across even in the impact "
                   f"scene; raise `shape` or lower `domain`")

    if verbose:
        print(f"scene '{scene}': {spec['what']}")
        print(f"  grid {shape[0]}x{shape[1]}x{shape[2]} "
              f"({mem['cells']/1e6:.1f} Mcells), peak RAM "
              f"~{mem['peak_GB']:.1f} GB, one array in VRAM "
              f"{mem['vram_GB']*1e3:.0f} MB")
        print(f"  domain +/-{vol.domain*1e3:.4g} mm, cell "
              f"{vol.spacing[0]*1e6:.3g} um, crater {cells:.1f} cells across")
        print(f"  t = {times[0]:.3e} to {times[-1]:.3e} s, "
              f"{len(times)} frames")

    res = _write_series_at(scenario, vol, times, directory,
                           name=name or scene, fields=want_fields,
                           verbose=verbose)
    res.update(scene=scene, t_end=times[-1], cells_across_crater=float(cells),
               colour_by=spec["colour_by"], memory=mem)
    return res


def write_impact_series(scenario, directory: str, n_frames: int = 60,
                        shape: tuple = (128, 128, 128),
                        name: str = "impact", time_mode: str | None = None,
                        fields=DEFAULT_FIELDS, compress: bool = True,
                        verbose: bool = True, **vol_kw) -> dict:
    """Write the whole time series plus a `.pvd` collection.

    Parameters
    ----------
    scenario : Scenario
        From `hvi_emp.run_scenario`.
    directory : str
    n_frames : int
    shape : (nx, ny, nz)
        Grid resolution. 128^3 is ~8 MB per scalar field per frame in
        Float32; the writer reports total size.
    time_mode : {"log", "linear"}
        Log spacing is strongly preferred: the plume evolves over five
        decades of time, so linear frames show the last decade only.

    Returns
    -------
    dict with ``pvd``, ``files``, ``times``, ``bytes``.
    """
    vol = ImpactVolume(scenario, shape=shape, **vol_kw)
    times = vol.times(n_frames, mode=time_mode)
    return _write_series_at(scenario, vol, times, directory, name=name,
                            fields=fields, compress=compress, verbose=verbose)


def _write_series_at(scenario, vol, times, directory: str, name: str,
                     fields=DEFAULT_FIELDS, compress: bool = True,
                     verbose: bool = True) -> dict:
    """Write an already-configured volume at already-chosen times.

    Split out so `write_scene` and `write_impact_series` cannot drift apart:
    the only thing that differs between them is how `vol` and `times` were
    arrived at.
    """
    os.makedirs(directory, exist_ok=True)
    shape = vol.shape

    want = (None if fields in ("all", None)
            else tuple(fields))
    files, total = [], 0
    for i, t in enumerate(times):
        frame = vol.frame(float(t))
        if want is not None:
            missing = set(want) - set(frame)
            if missing:
                raise KeyError(f"unknown field(s): {sorted(missing)}; "
                               f"available: {sorted(frame)}")
            frame = {k: frame[k] for k in want}
        p = os.path.join(directory, f"{name}_{i:04d}.vti")
        write_vti(p, frame, spacing=vol.spacing, origin=vol.origin,
                  compress=compress)
        total += os.path.getsize(p)
        files.append(p)
        if verbose and (i % max(1, len(times) // 10) == 0
                        or i == len(times) - 1):
            print(f"  frame {i + 1:>4}/{len(times)}  t = {t: .3e} s")

    imp = scenario.impact
    written = sorted(want) if want is not None else sorted(vol.frame(
        float(times[-1])))
    comment = (
        f"HVI-EMP impact visualisation\n"
        f"{imp.projectile.material.name} {imp.projectile.mass:.3e} kg at "
        f"{imp.projectile.velocity/1e3:.1f} km/s -> {imp.target.name}\n"
        f"domain: +/-{vol.domain*1e3:.4g} mm in x,y; "
        f"-{vol.z_below*1e3:.4g} to +{vol.domain*1e3:.4g} mm in z\n"
        f"cell {vol.spacing[0]*1e6:.4g} um; crater is "
        f"{vol.crater_radius/vol.spacing[0]:.2f} cells across\n"
        f"grid {shape[0]}x{shape[1]}x{shape[2]}, {len(times)} frames, "
        f"t = {times[0]:.2e} to {times[-1]:.2e} s\n"
        f"domain mode: {vol.domain_mode}"
        + (" (the box grows with the plume; Origin/Spacing vary per frame)"
           if vol.domain_mode == "adaptive" else "") + "\n\n"
        "FIELD PROVENANCE (see docs/VISUALISATION.md):\n"
        + "\n".join(f"  {k:<18} {FIELD_PROVENANCE[k]}"
                     for k in written if k in FIELD_PROVENANCE))
    pvd = write_pvd(os.path.join(directory, f"{name}.pvd"), files, times,
                    comment=comment)

    if verbose:
        print(f"\nwrote {len(files)} .vti + {os.path.basename(pvd)} "
              f"({total/1e6:.1f} MB) to {directory}")
        print(f"  paraview {pvd}")
    return {"pvd": pvd, "files": files, "times": times, "bytes": total,
            "volume": vol, "fields": written}


__all__ = ["write_vti", "write_pvd", "write_vtp", "write_impact_series",
           "QUALITY", "MINIMAL_FIELDS", "estimate_memory_GB",
           "BYTES_PER_CELL_PEAK",
           "write_scene", "scene_kwargs", "SCENES", "IMPACT_FIELDS",
           "ImpactVolume", "FIELD_PROVENANCE", "DEFAULT_FIELDS"]
