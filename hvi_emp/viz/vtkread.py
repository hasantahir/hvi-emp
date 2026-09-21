"""Read back the `.vti` / `.vtp` / `.pvd` files this package writes.

`vti.py` writes VTK XML with an appended, zlib-blocked binary payload. Until
now nothing could read it back without ParaView, which meant the only way to
find out what was in a frame was to render it -- and a black render does not
distinguish "the data is empty", "the colour field is flat" and "the camera
is wrong".

This is the inverse of `write_vti` / `write_vtp` / `write_pvd`, in numpy
only. It is deliberately narrow: it parses the files *this package* writes,
not VTK XML in general. No `<AppendedData encoding="base64">`, no parallel
`.pvti`, no multi-piece. Anything outside that raises rather than guessing.

The point of it is `field_stats`: the numbers that say whether a frame can
possibly produce a picture, before any renderer is involved.
"""

from __future__ import annotations

import os
import struct
import xml.etree.ElementTree as ET
import zlib

import numpy as np

__all__ = ["read_vtk_xml", "read_pvd", "field_stats", "pick_field",
           "opacity_plan", "FLOOR_TOL"]

_NP_TYPE = {
    "Float32": np.float32, "Float64": np.float64,
    "Int8": np.int8, "UInt8": np.uint8,
    "Int32": np.int32, "UInt32": np.uint32,
    "Int64": np.int64, "UInt64": np.uint64,
}

#: A value within this fraction of the array minimum counts as "floor" --
#: vacuum, unset, or the log-floor `write_vti` clamps to. Fields whose signal
#: occupies a vanishing fraction above the floor cannot volume-render with a
#: default opacity ramp, however correct the camera is.
FLOOR_TOL = 1e-6


def _split_appended(path: str):
    """(header text, whole file, offset of byte 0 of the appended blob)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    i = raw.find(b"<AppendedData")
    if i < 0:
        return raw.decode("utf-8", "replace"), raw, None
    j = raw.index(b"_", i) + 1
    return raw[:i].decode("utf-8", "replace"), raw, j


def _decode(da, raw, base, compressed):
    """One <DataArray format="appended"> into a flat numpy array."""
    fmt = da.get("format", "appended")
    if fmt != "appended":
        raise NotImplementedError(
            f"DataArray {da.get('Name')!r} is format={fmt!r}; this reader "
            f"handles only the appended-raw form that vti.py writes")
    dtype = _NP_TYPE.get(da.get("type"))
    if dtype is None:
        raise NotImplementedError(f"unhandled VTK type {da.get('type')!r}")
    off = base + int(da.get("offset", 0))

    if not compressed:
        (nbytes,) = struct.unpack_from("<Q", raw, off)
        count = int(nbytes) // np.dtype(dtype).itemsize
        return np.frombuffer(raw, dtype, count, off + 8)

    # [nblocks][block size][last block size][csize_1..csize_n] then blocks.
    nblocks, _bs, _last = struct.unpack_from("<QQQ", raw, off)
    p = off + 24
    sizes = struct.unpack_from(f"<{int(nblocks)}Q", raw, p)
    p += 8 * int(nblocks)
    chunks = []
    for s in sizes:
        chunks.append(zlib.decompress(raw[p:p + int(s)]))
        p += int(s)
    return np.frombuffer(b"".join(chunks), dtype)


def read_vtk_xml(path: str) -> dict:
    """Parse one `.vti` or `.vtp`.

    Returns a dict with `type`, the geometry description, and `arrays`
    mapping name -> ndarray. ImageData arrays come back shaped ``(nx, ny,
    nz)`` (or ``(nx, ny, nz, ncomp)``), undoing the transpose `write_vti`
    applies, so indices mean the same thing on both sides.
    """
    text, raw, base = _split_appended(path)
    compressed = "vtkZLibDataCompressor" in text
    if not text.rstrip().endswith("</VTKFile>"):
        text = text + "</VTKFile>"
    root = ET.fromstring(text)
    if root.get("header_type", "UInt64") != "UInt64":
        raise NotImplementedError("only header_type=UInt64 is handled")

    grid = root[0]
    piece = grid[0]
    out = {"path": path, "type": root.get("type"), "arrays": {},
           "components": {}}

    if out["type"] == "ImageData":
        ext = [int(v) for v in grid.get("WholeExtent").split()]
        dims = (ext[1] - ext[0] + 1, ext[3] - ext[2] + 1, ext[5] - ext[4] + 1)
        out["extent"] = ext
        out["dims"] = dims
        out["origin"] = [float(v) for v in grid.get("Origin", "0 0 0").split()]
        out["spacing"] = [float(v)
                          for v in grid.get("Spacing", "1 1 1").split()]
        o, s = out["origin"], out["spacing"]
        out["bounds"] = [o[0], o[0] + s[0] * (dims[0] - 1),
                         o[1], o[1] + s[1] * (dims[1] - 1),
                         o[2], o[2] + s[2] * (dims[2] - 1)]
    elif out["type"] == "PolyData":
        dims = None
        out["npoints"] = int(piece.get("NumberOfPoints", 0))
    else:
        raise NotImplementedError(f"unhandled VTKFile type {out['type']!r}")

    for tag in ("PointData", "CellData", "Points"):
        node = piece.find(tag)
        if node is None:
            continue
        for da in node.findall("DataArray"):
            flat = _decode(da, raw, base, compressed)
            ncomp = int(da.get("NumberOfComponents", 1))
            # The coordinates always land under "points", whatever the file
            # calls them -- write_vtp names that array "Points", and a
            # capital P would leave callers looking for coordinates under a
            # key that is not there while `pick_field` tried to colour by
            # them.
            name = "points" if tag == "Points" else (da.get("Name") or tag)
            if out["type"] == "ImageData" and tag in ("PointData", "CellData"):
                nx, ny, nz = dims if tag == "PointData" else [d + 1
                                                              for d in dims]
                want = nx * ny * nz * ncomp
                if flat.size != want:
                    raise ValueError(
                        f"{os.path.basename(path)}:{name} has {flat.size} "
                        f"values, expected {want} for {nx}x{ny}x{nz}x{ncomp}")
                if ncomp == 1:
                    arr = flat.reshape(nz, ny, nx).transpose(2, 1, 0)
                else:
                    arr = flat.reshape(nz, ny, nx, ncomp).transpose(2, 1, 0, 3)
            else:
                arr = flat.reshape(-1, ncomp) if ncomp > 1 else flat
            out["arrays"][name] = arr
            out["components"][name] = ncomp

    if out["type"] == "PolyData":
        pts = out["arrays"].get("points")
        if pts is not None and pts.size:
            out["bounds"] = [float(pts[:, 0].min()), float(pts[:, 0].max()),
                             float(pts[:, 1].min()), float(pts[:, 1].max()),
                             float(pts[:, 2].min()), float(pts[:, 2].max())]
    return out


def read_pvd(path: str) -> list:
    """`[(time, absolute frame path), ...]` in file order."""
    root = ET.parse(path).getroot()
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for ds in root.iter("DataSet"):
        f = ds.get("file")
        if not f:
            continue
        out.append((float(ds.get("timestep", 0.0)),
                    f if os.path.isabs(f) else os.path.join(base, f)))
    return out


def field_stats(arr) -> dict:
    """Can this array produce a picture?

    `signal_fraction` is the one that matters. A volume renderer with the
    default opacity ramp draws what is above the floor; when that is 1e-4 of
    the voxels, the frame is black no matter where the camera is, and the
    honest answer is to change the colour field or the transfer function
    rather than the camera.
    """
    a = np.asarray(arr, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return {"n": int(a.size), "finite": 0, "constant": True,
                "min": float("nan"), "max": float("nan"),
                "signal_fraction": 0.0, "signal_count": 0}
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo
    if span <= 0:
        return {"n": int(a.size), "finite": int(finite.size), "constant": True,
                "min": lo, "max": hi, "signal_fraction": 0.0,
                "signal_count": 0, "nonfinite": int(a.size - finite.size)}
    above = finite > lo + FLOOR_TOL * span
    n_sig = int(above.sum())
    sig = finite[above]
    return {
        "n": int(a.size), "finite": int(finite.size), "constant": False,
        "min": lo, "max": hi, "span": span,
        "signal_count": n_sig,
        "signal_fraction": n_sig / float(finite.size),
        # Percentiles of the signal only. Stretching a colour map over
        # [min, max] when the floor is 99.99% of the volume wastes the whole
        # map on vacuum; these are the numbers a useful map is built from.
        "p50": float(np.percentile(sig, 50)) if n_sig else lo,
        "p99": float(np.percentile(sig, 99)) if n_sig else lo,
        "nonfinite": int(a.size - finite.size),
    }


def pick_field(arrays, prefer=(), exclude=("points",)) -> tuple:
    """(name, stats) for the field most likely to render into something.

    Preference order is honoured only among fields that actually vary: a
    named default that happens to be constant in this frame loses to one
    that is not, because a constant field is a black picture by definition.
    """
    scored = []
    for name, arr in arrays.items():
        if name in exclude:
            continue
        a = np.asarray(arr)
        if a.ndim > 3 or (a.ndim == 2 and a.shape[-1] > 1):
            continue                      # vectors: no scalar map for them
        st = field_stats(a)
        if st["constant"] or not st["signal_count"]:
            continue
        rank = prefer.index(name) if name in prefer else len(prefer)
        # Among varying fields, prefer the caller's order; then the one with
        # the most voxels carrying signal.
        scored.append((rank, -st["signal_fraction"], name, st))
    if not scored:
        return None, None
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2], scored[0][3]


def opacity_plan(pvd_path: str, field: str | None = None,
                 max_frames: int = 12, prefer=()) -> dict | None:
    """Volume-rendering colour limits and opacity ramp, from the real data.

    Two hardcoded numbers used to make these scenes render black, and both
    were invisible in the log because ParaView did exactly as it was told:

    **The opacity knees were fractions of [min, max].** With the ramp rising
    only above 0.45 of the range, and 99.99% of a plume volume sitting at
    the -12 log floor, the entire signal landed in the transparent part of
    the curve. The fix is to put the knees at percentiles of the *signal*,
    which needs the actual values, which is what this reads.

    **`ScalarOpacityUnitDistance` was 1.0e-5 m for every scene.** VTK
    accumulates opacity as ``1 - (1-a)**(sample/unit)``, so a unit distance
    three orders of magnitude off the cell size does not dim the result, it
    annihilates or saturates it. The impact scene has ~6e-7 m cells, giving
    an exponent of 0.06 and an effective opacity near zero: black. It must
    be derived from the spacing.

    Returns None when the data cannot be read, so callers can fall back to
    the old fixed ramp rather than failing to write a scene at all.
    """
    try:
        frames = read_pvd(pvd_path)
    except Exception:                                       # noqa: BLE001
        return None
    if not frames:
        return None

    step = max(1, len(frames) // max(max_frames, 1))
    sample = frames[::step][:max_frames]
    if frames[-1] not in sample:
        sample.append(frames[-1])

    # Choose the field by a vote over the sampled frames, never from the
    # first one. Frame 0 is the frame where nothing has happened yet: in
    # every scene here its log density is *exactly* constant, so a
    # first-frame choice silently falls through to whatever else varies
    # (`material`, which is geometry, not physics) and the whole plan is
    # then built for the wrong array.
    loaded, spacing, votes = [], None, {}
    for _t, path in sample:
        try:
            d = read_vtk_xml(path)
        except Exception:                                   # noqa: BLE001
            continue
        if d["type"] != "ImageData":
            return None
        spacing = spacing or d.get("spacing")
        loaded.append(d)
        if field is None:
            name, _ = pick_field(d["arrays"], prefer=tuple(prefer))
            if name:
                votes[name] = votes.get(name, 0) + 1
    chosen = field or (max(votes, key=votes.get) if votes else None)
    if chosen is None:
        return None

    sig_all, stats = [], []
    for d in loaded:
        if chosen not in d["arrays"]:
            continue
        a = np.asarray(d["arrays"][chosen], float).ravel()
        fin = a[np.isfinite(a)]
        if fin.size == 0:
            continue
        st = field_stats(fin)
        stats.append(st)
        if st["constant"]:
            continue
        lo = st["min"]
        sig = fin[fin > lo + FLOOR_TOL * st["span"]]
        if sig.size:
            # Subsample: percentiles do not need every voxel of every frame,
            # and this runs at scene-writing time on a login node.
            sig_all.append(sig if sig.size <= 200_000 else
                           np.random.default_rng(0).choice(sig, 200_000,
                                                           replace=False))
    if not sig_all:
        return None

    sig = np.concatenate(sig_all)
    p50, p90, hi = (float(np.percentile(sig, 50)),
                    float(np.percentile(sig, 90)), float(sig.max()))
    lo = float(min(s["min"] for s in stats if not s["constant"]))
    # Start the map at the signal median, not the floor: below it is vacuum,
    # and a colour map spent on vacuum is a black frame with a legend.
    c_lo = p50 if p50 > lo else lo
    if hi <= c_lo:
        hi = c_lo + max(abs(c_lo) * 1e-6, 1e-12)

    span = hi - c_lo
    points = [c_lo, 0.0, 0.5, 0.0,
              c_lo + 0.25 * span, 0.08, 0.5, 0.0,
              p90 if c_lo < p90 < hi else c_lo + 0.6 * span, 0.35, 0.5, 0.0,
              hi, 0.9, 0.5, 0.0]

    unit = float(np.mean([abs(s) for s in (spacing or [1.0])
                          if s]) or 1.0)
    n_const = sum(1 for s in stats if s["constant"])
    return {
        "field": chosen, "lo": c_lo, "hi": hi, "floor": lo,
        "p50": p50, "p90": p90, "points": points,
        "unit_distance": unit,
        "signal_fraction": float(np.mean([s["signal_fraction"]
                                          for s in stats])),
        "constant_frames": n_const, "frames_sampled": len(stats),
    }
