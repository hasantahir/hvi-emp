"""Tests for the `.vti` writer and the impact volume sampler.

The writer is validated by an **independent parser** defined here, not by a
VTK library: the bytes are what ParaView reads, so the bytes are what gets
checked. A malformed compressed block loads as garbage rather than failing,
which is exactly the sort of bug a round-trip catches and eyeballing does not.
"""

from __future__ import annotations

import re
import struct
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp import run_scenario
from hvi_emp.viz import (FIELD_PROVENANCE, ImpactVolume, write_impact_series,
                         write_pvd, write_vti)
from hvi_emp.viz.vti import DEFAULT_FIELDS

warnings.simplefilter("ignore")

_NP = {"Float32": np.float32, "Float64": np.float64,
       "UInt8": np.uint8, "Int32": np.int32}


def read_vti(path):
    """Independent .vti reader -- deliberately not using any VTK library."""
    raw = Path(path).read_bytes()
    marker = raw.index(b"<AppendedData")
    cut = raw.index(b"_", marker) + 1
    head = raw[:cut].decode("ascii")
    data = raw[cut:]

    assert 'header_type="UInt64"' in head
    assert 'byte_order="LittleEndian"' in head
    ext = [int(v) for v in
           re.search(r'WholeExtent="([\d\s]+)"', head).group(1).split()]
    nx, ny, nz = ext[1] + 1, ext[3] + 1, ext[5] + 1
    spacing = [float(v) for v in
               re.search(r'Spacing="([^"]+)"', head).group(1).split()]
    origin = [float(v) for v in
              re.search(r'Origin="([^"]+)"', head).group(1).split()]
    compressed = 'compressor="vtkZLibDataCompressor"' in head

    fields = {}
    for typ, name, ncomp, off in re.findall(
            r'<DataArray type="(\w+)" Name="([^"]+)" '
            r'NumberOfComponents="(\d+)" format="appended" offset="(\d+)"/>',
            head):
        ncomp, off = int(ncomp), int(off)
        if compressed:
            nb, _bs, _last = struct.unpack_from("<QQQ", data, off)
            sizes = struct.unpack_from(f"<{nb}Q", data, off + 24)
            pos, buf = off + 24 + 8 * nb, b""
            for cs in sizes:
                buf += zlib.decompress(data[pos:pos + cs])
                pos += cs
        else:
            n = struct.unpack_from("<Q", data, off)[0]
            buf = data[off + 8: off + 8 + n]
        arr = np.frombuffer(buf, dtype=_NP[typ])
        assert arr.size == nx * ny * nz * ncomp, name
        arr = (arr.reshape(nz, ny, nx).transpose(2, 1, 0) if ncomp == 1
               else arr.reshape(nz, ny, nx, ncomp).transpose(2, 1, 0, 3))
        fields[name] = arr
    return {"shape": (nx, ny, nz), "spacing": spacing, "origin": origin,
            "compressed": compressed, "fields": fields}


# ===========================================================================
# Writer round-trips
# ===========================================================================

@pytest.mark.parametrize("compress", [False, True])
def test_vti_roundtrip_exact(tmp_path, compress):
    """Scalars, vectors and uint8 must survive byte-for-byte, in VTK's
    x-fastest ordering."""
    rng = np.random.default_rng(0)
    nx, ny, nz = 13, 7, 5
    a = (rng.random((nx, ny, nz)) * 1e6).astype(np.float32)
    v = np.zeros((nx, ny, nz, 3), np.float32)
    v[..., 0], v[..., 2] = a, -a
    u = (a % 7).astype(np.uint8)

    p = tmp_path / "t.vti"
    write_vti(str(p), {"s": a, "vec": v, "mat": u},
              spacing=(0.1, 0.2, 0.3), origin=(-1.0, -2.0, -3.0),
              compress=compress)
    r = read_vti(p)

    assert r["shape"] == (nx, ny, nz)
    assert np.allclose(r["spacing"], [0.1, 0.2, 0.3])
    assert np.allclose(r["origin"], [-1.0, -2.0, -3.0])
    assert r["compressed"] is compress
    assert np.array_equal(r["fields"]["s"], a)
    assert np.array_equal(r["fields"]["vec"], v)
    assert np.array_equal(r["fields"]["mat"], u)


def test_vti_ordering_is_x_fastest(tmp_path):
    """The single most likely silent bug: axes transposed on write."""
    a = np.zeros((4, 3, 2), np.float32)
    a[1, 0, 0] = 1.0          # a specific, asymmetric voxel
    p = tmp_path / "o.vti"
    write_vti(str(p), {"s": a}, spacing=(1, 1, 1))
    got = read_vti(p)["fields"]["s"]
    assert got[1, 0, 0] == 1.0
    assert got.sum() == 1.0


def test_vti_compression_shrinks_smooth_data(tmp_path):
    a = np.zeros((32, 32, 32), np.float32)
    a[8:24, 8:24, 8:24] = 1.0        # mostly zero, like the real fields
    p1, p2 = tmp_path / "u.vti", tmp_path / "c.vti"
    write_vti(str(p1), {"s": a}, spacing=(1, 1, 1), compress=False)
    write_vti(str(p2), {"s": a}, spacing=(1, 1, 1), compress=True)
    assert p2.stat().st_size < 0.25 * p1.stat().st_size
    assert np.array_equal(read_vti(p2)["fields"]["s"], a)


def test_vti_rejects_inconsistent_shapes(tmp_path):
    with pytest.raises(ValueError, match="inconsistent"):
        write_vti(str(tmp_path / "x.vti"),
                  {"a": np.zeros((4, 4, 4)), "b": np.zeros((4, 4, 5))},
                  spacing=(1, 1, 1))


def test_pvd_carries_physical_times(tmp_path):
    files = [str(tmp_path / f"f_{i}.vti") for i in range(3)]
    times = [-1e-9, 1e-9, 1e-6]
    p = write_pvd(str(tmp_path / "s.pvd"), files, times, comment="note")
    txt = Path(p).read_text()
    for t in times:
        assert f'timestep="{t:.9g}"' in txt
    assert "f_0.vti" in txt and "note" in txt
    assert 'type="Collection"' in txt


# ===========================================================================
# Impact volume sampler
# ===========================================================================

@pytest.fixture(scope="module")
def scenario():
    return run_scenario("Fe", "Al", mass=1e-9, velocity=30e3)


def test_volume_geometry_conventions(scenario):
    """Target below z=0, plume above, impact at the origin in x,y."""
    vol = ImpactVolume(scenario, shape=(24, 24, 24))
    f = vol.frame(1e-7)
    mat = f["material"]
    below = vol.Z < 0
    # solid target only below the surface
    assert (mat[below] == 1).any()
    assert not (mat[~below] == 1).any()
    # plume only above it
    plume = f["plasma_density"]
    assert plume[below].max() == 0.0
    assert plume[~below].max() > 0.0


def test_volume_plume_expands_and_thins(scenario):
    """The physics the animation is supposed to show."""
    vol = ImpactVolume(scenario, shape=(24, 24, 24))   # adaptive default
    early = vol.frame(1e-8)
    late = vol.frame(1e-5)
    assert late["plasma_density"].max() < early["plasma_density"].max()
    assert late["temperature_eV"].max() <= early["temperature_eV"].max()


def test_adaptive_domain_keeps_the_plume_visible(scenario):
    """Regression: with a fixed box sized to the final metre-scale plume, the
    early frames contained no plume at all -- the whole animation was empty
    until the last few frames."""
    vol = ImpactVolume(scenario, shape=(32, 32, 32), domain_mode="adaptive")
    for t in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
        f = vol.frame(t)
        assert f["plasma_density"].max() > 0.0, f"no plume at t={t:g}"
    # and the box must actually grow
    d_early, _ = vol.domain_at(1e-9)
    d_late, _ = vol.domain_at(1e-5)
    assert d_late > 10 * d_early


def test_projectile_only_before_contact(scenario):
    vol = ImpactVolume(scenario, shape=(24, 24, 24))
    assert vol.frame(-1e-9)["projectile"].max() == 1.0
    assert vol.frame(+1e-9)["projectile"].max() == 0.0


def test_crater_grows_monotonically(scenario):
    vol = ImpactVolume(scenario, shape=(16, 16, 16))
    r = [vol.crater_radius_at(t) for t in (0.0, 1e-9, 1e-8, 1e-7, 1e-5)]
    assert r[0] == 0.0
    assert all(b >= a for a, b in zip(r, r[1:]))
    assert r[-1] == pytest.approx(vol.crater_radius, rel=1e-9)


def test_smoke_is_slower_than_the_plasma(scenario):
    """The condensate must trail the plasma -- that is its whole point."""
    vol = ImpactVolume(scenario, shape=(32, 32, 32))
    f = vol.frame(1e-6)
    if f["smoke_density"].max() > 0:
        # extent along z of each component
        zs = vol.Z[f["smoke_density"] > f["smoke_density"].max() * 1e-3]
        zp = vol.Z[f["plasma_density"] > f["plasma_density"].max() * 1e-3]
        assert zs.max() < zp.max()


def test_every_field_has_a_provenance_tag(scenario):
    vol = ImpactVolume(scenario, shape=(16, 16, 16))
    assert set(vol.frame(1e-7)) == set(FIELD_PROVENANCE)


# ===========================================================================
# Series driver
# ===========================================================================

def test_write_impact_series(tmp_path, scenario):
    res = write_impact_series(scenario, str(tmp_path / "s"), n_frames=8,
                              shape=(16, 16, 16), verbose=False)
    assert len(res["files"]) == 8
    assert Path(res["pvd"]).exists()
    assert res["times"][0] < 0 < res["times"][-1]     # pre-impact frames
    assert np.all(np.diff(res["times"]) > 0)          # monotone

    r = read_vti(res["files"][-1])
    assert set(r["fields"]) == set(DEFAULT_FIELDS)
    pvd = Path(res["pvd"]).read_text()
    assert "PROVENANCE" in pvd                        # honesty survives
    for f in res["files"]:
        assert Path(f).name in pvd


def test_field_selection_and_unknown_field(tmp_path, scenario):
    res = write_impact_series(scenario, str(tmp_path / "a"), n_frames=3,
                              shape=(12, 12, 12), fields=("density",),
                              verbose=False)
    assert set(read_vti(res["files"][0])["fields"]) == {"density"}

    res_all = write_impact_series(scenario, str(tmp_path / "b"), n_frames=3,
                                  shape=(12, 12, 12), fields="all",
                                  verbose=False)
    assert set(read_vti(res_all["files"][0])["fields"]) == set(FIELD_PROVENANCE)

    with pytest.raises(KeyError, match="unknown field"):
        write_impact_series(scenario, str(tmp_path / "c"), n_frames=2,
                            shape=(8, 8, 8), fields=("nope",), verbose=False)


def test_series_refuses_when_there_is_no_plume(tmp_path):
    sub = run_scenario("Fe", "Al", mass=1e-12, velocity=5e3)
    assert sub.expansion is None
    with pytest.raises(ValueError, match="no plume"):
        ImpactVolume(sub)


if __name__ == "__main__":       # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))


# ===========================================================================
# The animation has to show the expansion
# ===========================================================================

def _plume_extent_in_voxels(vol, t):
    """Half-max radius of the plume, in voxels, for one frame."""
    half, _ = vol.domain_at(t)
    n = vol.shape[0]
    exp = vol.scenario.expansion
    k = int(np.argmin(np.abs(np.asarray(exp.t) - t)))
    R = float(np.sqrt(exp.R_r[k] * exp.R_z[k]))
    return R / (2.0 * half / n)


def test_window_mode_actually_shows_the_plume_expanding(scenario):
    """The default must render an expansion, not a fixed-size ball.

    The previous default (`adaptive`) grew the box in lockstep with the
    plume, so the plume occupied a *constant* ~7% of the box in every frame.
    ParaView then faithfully showed a ball of fixed size while the physical
    radius grew by 4000x -- the animation was a zoom-out wearing an
    expansion's clothes.
    """
    from hvi_emp.viz.vti import ImpactVolume

    vol = ImpactVolume(scenario, shape=(96, 96, 96))
    assert vol.domain_mode == "window", "window must be the default"
    ts = [t for t in vol.times(24) if t > 0]
    v0 = _plume_extent_in_voxels(vol, ts[0])
    v1 = _plume_extent_in_voxels(vol, ts[-1])

    assert v1 / v0 >= vol.min_growth, (
        f"plume only grows {v1 / v0:.1f}x on screen ({v0:.1f} -> {v1:.1f} "
        "voxels); that will not read as an expansion")
    # The plume also drifts several radii downrange, so a box that keeps the
    # impact site in view is much larger than the plume itself; the final
    # radius in cells scales with `shape`. What must hold at any resolution is
    # that the final frame is resolved, not sub-voxel.
    assert v1 > 2.0, (
        f"final plume is {v1:.1f} cells in radius -- raise `shape` "
        "(192 is a good workstation setting) or window_start")
    # and it must still fit
    half, _ = vol.domain_at(ts[-1])
    exp = scenario.expansion
    k = int(np.argmin(np.abs(np.asarray(exp.t) - ts[-1])))
    R = float(np.sqrt(exp.R_r[k] * exp.R_z[k]))
    assert R < half, "final plume overflows the box"


def test_window_mode_keeps_the_cell_size_constant(scenario):
    """A fixed box is what makes growth visible; spacing must not drift."""
    from hvi_emp.viz.vti import ImpactVolume

    vol = ImpactVolume(scenario, shape=(64, 64, 64), domain_mode="window")
    halves = [vol.domain_at(t)[0] for t in vol.times(16) if t > 0]
    assert max(halves) == pytest.approx(min(halves), rel=1e-12)


def test_adaptive_mode_is_a_zoom_not_an_expansion(scenario):
    """Documents the trap, so nobody reinstates it as the default by accident."""
    from hvi_emp.viz.vti import ImpactVolume

    vol = ImpactVolume(scenario, shape=(96, 96, 96), domain_mode="adaptive")
    ts = [t for t in vol.times(24) if t > 0]
    sizes = [_plume_extent_in_voxels(vol, t) for t in ts]
    # constant to within a factor of ~1.5 -- that is the whole problem
    assert max(sizes) / min(sizes) < 2.0
    # while the physical radius changes by orders of magnitude
    exp = scenario.expansion
    R = np.sqrt(np.asarray(exp.R_r) * np.asarray(exp.R_z))
    assert R[-1] / R[1] > 100.0


def test_series_written_in_window_mode_grows_on_disk(tmp_path, scenario):
    """End-to-end: read the frames back and count plume voxels."""
    from hvi_emp.viz.vti import write_impact_series

    out = write_impact_series(scenario, str(tmp_path), n_frames=10,
                              shape=(96, 96, 96), domain_mode="window",
                              verbose=False)
    counts, spacings = [], []
    for path in sorted(out["files"]):
        meta = read_vti(path)                     # the independent parser
        a = meta["fields"]["plasma_density"]
        if a.max() <= 0:
            continue
        counts.append(int((a > 0.5 * a.max()).sum()))
        spacings.append(meta["spacing"][0])

    assert len(counts) >= 3
    assert counts[-1] > 4 * counts[0], (
        f"plume voxel count barely moved: {counts}")
    assert max(spacings) == pytest.approx(min(spacings), rel=1e-9), (
        "cell size must be constant in window mode")


# ---------------------------------------------------------------------------
# Why a rendered plume looked like a static blob
# ---------------------------------------------------------------------------

def test_comoving_box_holds_the_plume_not_its_trajectory(scenario):
    """The lab-frame box is sized by drift, which dwarfs the plume.

    A 50 km/s impact sends the plume downrange at ~25 km/s while it expands
    at ~2.5 km/s, so over the sampled window it travels ~0.25 m and grows to
    ~0.035 m. A box that keeps the impact site in view is therefore ~10x
    larger than the object in it, and the plume occupies under 2% of the
    voxels -- which renders as a small ball in a mostly empty box.
    """
    lab = ImpactVolume(scenario, shape=(48, 48, 48),
                       frame_of_reference="lab")
    plume = ImpactVolume(scenario, shape=(48, 48, 48),
                         frame_of_reference="plume")
    t = plume.times(n_frames=6)[-1]
    # The co-moving box only has to hold the plume, so it is several times
    # smaller. The margin is not tighter than this because the default
    # aspect0=0.5 makes the plume taller than it is wide (R_z/R_r -> 1.37),
    # and the box is sized by the larger semi-axis.
    assert plume.domain_at(t)[0] < 0.4 * lab.domain_at(t)[0]

    def fill(vol):
        d = vol.frame(float(t))["plasma_density"].ravel()
        m = d.max()
        return int((d > 1e-3 * m).sum()) / d.size if m > 0 else 0.0

    assert fill(plume) > 8 * fill(lab)


def test_comoving_plume_grows_within_a_fixed_box(scenario):
    """The point of the animation: the plume must visibly expand.

    The box is sized once from the window end, so growth is on screen. A box
    that resized every frame would hold the plume at constant apparent size
    and hide exactly what is being animated.
    """
    vol = ImpactVolume(scenario, shape=(48, 48, 48),
                       frame_of_reference="plume")
    ts = vol.times(n_frames=6)
    widths = [vol.domain_at(float(t))[0] for t in ts]
    assert max(widths) == pytest.approx(min(widths), rel=1e-9), \
        "co-moving box must be fixed, or growth is hidden"

    fills = []
    for t in ts[2:]:
        d = vol.frame(float(t))["plasma_density"].ravel()
        m = d.max()
        fills.append(int((d > 1e-3 * m).sum()) / d.size if m > 0 else 0.0)
    assert fills[-1] > 4 * fills[0], "plume does not visibly grow"
    assert fills[-1] < 0.95, "plume overflows the box"


def test_comoving_frame_keeps_the_whole_plume(scenario):
    """The surface clip must not apply when z=0 is the plume's middle.

    In the lab frame the plume is cut at the target surface. Reusing that
    clip in the co-moving frame would delete its lower half.
    """
    vol = ImpactVolume(scenario, shape=(48, 48, 48),
                       frame_of_reference="plume")
    d = vol.frame(float(vol.times(n_frames=6)[-1]))["plasma_density"]
    nz = d.shape[2]
    below = d[:, :, :nz // 2].sum()
    above = d[:, :, nz // 2:].sum()
    assert below > 0, "lower half of the plume was clipped away"
    assert below == pytest.approx(above, rel=0.25)


def test_comoving_frame_draws_no_target(scenario):
    """The target is hundreds of plume radii behind the co-moving box."""
    vol = ImpactVolume(scenario, shape=(48, 48, 48),
                       frame_of_reference="plume")
    f = vol.frame(float(vol.times(n_frames=6)[-1]))
    assert not (f["material"] == 1).any(), "solid target drawn in plume frame"
    assert f["shock_pressure"].max() == 0.0


def test_log_density_field_is_bounded_and_matches_the_raw_field(scenario):
    """A linear colour map on log10 is a log map on the density.

    The raw field spans ~36 decades inside one frame, so a linear colour bar
    over it shows a uniform blob. This is the field to colour by.
    """
    vol = ImpactVolume(scenario, shape=(48, 48, 48))
    f = vol.frame(float(vol.times(n_frames=6)[-1]))
    raw, lg = f["plasma_density"], f["log10_plasma_density"]
    assert lg.min() == pytest.approx(np.log10(vol.log_floor), abs=1e-5)
    assert np.isfinite(lg).all(), "log field must never be -inf or NaN"
    live = raw > vol.log_floor
    assert np.allclose(lg[live], np.log10(raw[live].astype(float)), atol=1e-5)
    # bounded to something a colour bar can resolve
    assert lg.max() - lg.min() < 20


def test_default_fields_include_the_log_density(scenario):
    from hvi_emp.viz.vti import DEFAULT_FIELDS, FIELD_PROVENANCE
    assert "log10_plasma_density" in DEFAULT_FIELDS
    for name in DEFAULT_FIELDS:
        assert name in FIELD_PROVENANCE, f"{name} has no provenance tag"


def test_bad_frame_of_reference_is_rejected(scenario):
    with pytest.raises(ValueError, match="lab.*plume|plume.*lab"):
        ImpactVolume(scenario, shape=(16, 16, 16),
                     frame_of_reference="comoving")


def test_adaptive_mode_holds_apparent_size_constant(scenario):
    """`adaptive` is why an animation can look completely static.

    It resizes the box to the plume every frame, so the plume occupies the
    same fraction of the image throughout: measured R/box = 0.068 from the
    first frame to the last while the plume itself grew 3.5x. That reads as
    "the simulation is not running" when only the camera is following it.

    This is pinned rather than fixed because the mode has a legitimate use
    (inspecting internal structure at every scale). What is fixed is that it
    is no longer the default anywhere, and that it announces itself.
    """
    vol = ImpactVolume(scenario, shape=(32, 32, 32), domain_mode="adaptive")
    ts = vol.times(n_frames=6)
    e = scenario.expansion
    ratios = []
    for t in ts[2:]:
        half = vol.domain_at(float(t))[0]
        R = float(max(np.interp(t, e.t, e.R_r), np.interp(t, e.t, e.R_z)))
        ratios.append(R / half)
    assert max(ratios) == pytest.approx(min(ratios), rel=0.05), \
        "adaptive should hold apparent size fixed; behaviour changed"
    assert any("apparent size is CONSTANT" in n for n in vol.notes), \
        "adaptive mode must warn that it hides the expansion"


def test_window_mode_actually_shows_growth(scenario):
    """The default must do the opposite: apparent size grows on screen."""
    vol = ImpactVolume(scenario, shape=(32, 32, 32), domain_mode="window")
    assert vol.domain_mode == "window", "window must be the class default"
    ts = vol.times(n_frames=6)
    e = scenario.expansion
    ratios = []
    for t in ts[2:]:
        half = vol.domain_at(float(t))[0]
        R = float(max(np.interp(t, e.t, e.R_r), np.interp(t, e.t, e.R_z)))
        ratios.append(R / half)
    assert ratios[-1] > 3.0 * ratios[0], \
        f"plume grew only {ratios[-1]/ratios[0]:.1f}x on screen"
    assert not any("apparent size is CONSTANT" in n for n in vol.notes)


def test_example_does_not_override_the_good_defaults():
    """Regression: the example passed domain_mode='adaptive' explicitly.

    The class default was fixed to 'window' long before, but the example --
    which is what people actually run -- overrode it, so the fix never
    reached anyone. A default is only fixed if the callers stop overriding
    it.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "examples" / "08_vti_visualisation.py").read_text()
    assert 'default="adaptive"' not in src, \
        "the example is overriding the default back to adaptive"
    assert 'default="window"' in src
