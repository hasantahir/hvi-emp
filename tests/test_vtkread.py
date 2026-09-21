"""Read back what we write, and pin the two constants that caused black frames.

The regressions here are not hypothetical. Every one of them is a bug that
shipped and produced an all-black animation that looked, in the log, exactly
like a successful render.
"""

import numpy as np
import pytest

from hvi_emp.viz.vti import write_pvd, write_vti, write_vtp
from hvi_emp.viz.vtkread import (FLOOR_TOL, field_stats, opacity_plan,
                                 pick_field, read_pvd, read_vtk_xml)


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("compress", [True, False])
def test_vti_round_trip_values_and_orientation(tmp_path, compress):
    """Values come back at the same (x, y, z) index they went in at.

    `write_vti` transposes to VTK's x-fastest order. If the reader does not
    undo exactly that, arrays come back looking plausible and permuted,
    which is worse than failing.
    """
    nx, ny, nz = 5, 6, 7
    a = np.arange(nx * ny * nz, dtype=np.float32).reshape(nx, ny, nz)
    p = tmp_path / "f.vti"
    write_vti(str(p), {"a": a}, spacing=(0.1, 0.2, 0.3),
              origin=(-1.0, -2.0, -3.0), compress=compress)

    d = read_vtk_xml(str(p))
    assert d["type"] == "ImageData"
    assert d["dims"] == (nx, ny, nz)
    np.testing.assert_allclose(d["arrays"]["a"], a)
    np.testing.assert_allclose(d["spacing"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(d["origin"], [-1.0, -2.0, -3.0])
    # A distinctive corner: catches a transpose that a symmetric array hides.
    assert d["arrays"]["a"][1, 2, 3] == a[1, 2, 3]


def test_vti_multiple_fields_and_dtypes(tmp_path):
    nx, ny, nz = 4, 4, 4
    f32 = np.random.default_rng(0).random((nx, ny, nz)).astype(np.float32)
    f64 = np.random.default_rng(1).random((nx, ny, nz))
    p = tmp_path / "m.vti"
    write_vti(str(p), {"a": f32, "b": f64}, spacing=(1, 1, 1))
    d = read_vtk_xml(str(p))
    np.testing.assert_allclose(d["arrays"]["a"], f32)
    np.testing.assert_allclose(d["arrays"]["b"], f64)


def test_vti_large_enough_to_span_blocks(tmp_path):
    """More than one zlib block, so the blocked header is exercised."""
    a = np.random.default_rng(2).random((40, 40, 40)).astype(np.float32)
    assert a.nbytes > (1 << 15)
    p = tmp_path / "big.vti"
    write_vti(str(p), {"a": a}, spacing=(1, 1, 1), compress=True)
    np.testing.assert_allclose(read_vtk_xml(str(p))["arrays"]["a"], a)


def test_vtp_round_trip(tmp_path):
    pts = np.random.default_rng(3).random((50, 3))
    T = np.linspace(0, 1, 50)
    p = tmp_path / "p.vtp"
    write_vtp(str(p), pts, {"T_eV": T})
    d = read_vtk_xml(str(p))
    assert d["type"] == "PolyData"
    assert d["npoints"] == 50
    np.testing.assert_allclose(d["arrays"]["points"], pts, rtol=1e-6)
    np.testing.assert_allclose(d["arrays"]["T_eV"], T, rtol=1e-6)


def test_pvd_round_trip(tmp_path):
    for i in range(3):
        write_vti(str(tmp_path / f"s_{i:04d}.vti"),
                  {"a": np.zeros((3, 3, 3), np.float32)}, spacing=(1, 1, 1))
    times = [0.0, 1.5e-6, 3.0e-6]
    write_pvd(str(tmp_path / "s.pvd"),
              [f"s_{i:04d}.vti" for i in range(3)], times)
    got = read_pvd(str(tmp_path / "s.pvd"))
    assert [t for t, _ in got] == times
    assert all(str(tmp_path) in f for _, f in got)


def test_shape_mismatch_is_loud(tmp_path):
    """A truncated payload must raise, not silently reshape."""
    p = tmp_path / "t.vti"
    write_vti(str(p), {"a": np.zeros((4, 4, 4), np.float32)}, spacing=(1, 1, 1))
    raw = bytearray(p.read_bytes())
    raw = raw.replace(b'WholeExtent="0 3 0 3 0 3"',
                      b'WholeExtent="0 3 0 3 0 4"', 1)
    p.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="expected"):
        read_vtk_xml(str(p))


# --------------------------------------------------------------------------
# field_stats: the numbers that decide whether a frame can be seen
# --------------------------------------------------------------------------

def test_constant_field_is_reported_constant():
    st = field_stats(np.full(1000, -12.0))
    assert st["constant"] is True
    assert st["signal_fraction"] == 0.0


def test_signal_fraction_finds_a_needle():
    """The plume case: a handful of live voxels in a sea of floor.

    0.006% is a real measurement from a 64^3 plume frame. Any renderer told
    to map [min, max] over this spends the whole map on vacuum.
    """
    a = np.full(262144, -12.0)
    a[:16] = -7.5
    st = field_stats(a)
    assert st["signal_count"] == 16
    assert st["signal_fraction"] == pytest.approx(16 / 262144)
    # The percentiles describe the SIGNAL, not the floor.
    assert st["p50"] == pytest.approx(-7.5)
    assert st["min"] == -12.0


def test_nan_does_not_poison_the_range():
    a = np.array([1.0, 2.0, np.nan, 3.0])
    st = field_stats(a)
    assert st["min"] == 1.0 and st["max"] == 3.0
    assert st["nonfinite"] == 1


def test_floor_tolerance_is_relative_not_absolute():
    a = np.full(100, 1e6)
    a[:10] = 1e6 * (1 + 10 * FLOOR_TOL)
    assert field_stats(a)["signal_count"] == 10


# --------------------------------------------------------------------------
# pick_field
# --------------------------------------------------------------------------

def test_pick_field_skips_a_constant_preferred_field():
    """THE frame-0 bug, in one test.

    Frame 0 of every scene has a constant log density -- nothing has
    happened yet. Choosing the colour field from that frame silently falls
    through to `material`, which is geometry, and the whole transfer
    function then gets built for the wrong array on the wrong range.
    """
    arrays = {"log10_plasma_density": np.full((4, 4, 4), -12.0),
              "material": np.arange(64, dtype=float).reshape(4, 4, 4)}
    name, st = pick_field(arrays, prefer=("log10_plasma_density", "material"))
    assert name == "material"


def test_pick_field_honours_preference_when_both_vary():
    rng = np.random.default_rng(4)
    arrays = {"log10_plasma_density": rng.random((4, 4, 4)),
              "material": rng.random((4, 4, 4))}
    name, _ = pick_field(arrays, prefer=("log10_plasma_density", "material"))
    assert name == "log10_plasma_density"


def test_pick_field_returns_none_when_everything_is_flat():
    arrays = {"a": np.zeros((3, 3, 3)), "b": np.ones((3, 3, 3))}
    assert pick_field(arrays) == (None, None)


# --------------------------------------------------------------------------
# opacity_plan: the two constants that made the frames black
# --------------------------------------------------------------------------

def _floor_series(tmp_path, spacing, n=6, dims=(16, 16, 16)):
    """A series shaped like the real ones: mostly floor, growing signal."""
    files = []
    nx, ny, nz = dims
    rng = np.random.default_rng(7)
    for i in range(n):
        a = np.full((nx, ny, nz), -12.0, np.float32)
        if i:                                   # frame 0 is constant, as in
            k = 1 + i                           # every real scene
            # Skewed towards the floor, like a real expanding plume: a thin
            # bright core in a large faint halo. A uniform ramp would put
            # the median signal voxel in the middle of the range, which is
            # the one shape that hides the bug being pinned.
            v = -12.0 + 5.6 * rng.beta(1.2, 8.0, k ** 3)
            a[:k, :k, :k] = v.reshape(k, k, k)
            a[0, 0, 0] = -6.4
        fn = f"s_{i:04d}.vti"
        write_vti(str(tmp_path / fn), {"log10_plasma_density": a},
                  spacing=spacing)
        files.append(fn)
    write_pvd(str(tmp_path / "s.pvd"), files,
              [i * 1e-6 for i in range(n)])
    return str(tmp_path / "s.pvd")


def test_map_starts_above_the_floor(tmp_path):
    """The colour and opacity map must not be spent on vacuum.

    With the map anchored at the array minimum, the median signal voxel of
    a real plume frame received opacity 0.0000 from the old ramp. The map
    has to start at the signal, not at the floor.
    """
    plan = opacity_plan(_floor_series(tmp_path, (1e-2, 1e-2, 1e-2)))
    assert plan["floor"] == pytest.approx(-12.0)
    # Strictly above the floor, and anchored on the signal median rather
    # than on whatever the emptiest voxel happens to hold.
    assert plan["lo"] > plan["floor"]
    assert plan["lo"] == pytest.approx(plan["p50"])
    assert plan["points"][0] == pytest.approx(plan["lo"])
    # The map is narrower than the raw range: the excluded part is vacuum.
    assert (plan["hi"] - plan["lo"]) < (plan["hi"] - plan["floor"])


def test_opacity_ramp_gives_the_median_signal_voxel_real_opacity(tmp_path):
    """Regression on the old ramp, stated as the quantity that matters.

    Old: knees at 0.45 and 0.75 of [min, max]; with min at the -12 floor the
    median signal voxel landed below the first knee and got alpha = 0.
    """
    plan = opacity_plan(_floor_series(tmp_path, (1e-2, 1e-2, 1e-2)))
    pts = np.array(plan["points"]).reshape(-1, 4)
    alpha = float(np.interp(plan["p50"], pts[:, 0], pts[:, 1]))

    lo, hi = plan["floor"], plan["hi"]
    frac_old = (plan["p50"] - lo) / (hi - lo)
    assert frac_old < 0.45, "the old ramp's first knee was at 0.45"
    assert alpha >= 0.0
    assert float(np.interp(plan["p90"], pts[:, 0], pts[:, 1])) > 0.05


def test_unit_distance_follows_the_cell_size(tmp_path):
    """`ScalarOpacityUnitDistance` was 1.0e-5 m for every scene.

    VTK accumulates opacity as 1-(1-a)**(sample/unit), so this is not a
    brightness knob. The two real scenes differ by more than three orders of
    magnitude in cell size; one constant cannot serve both.
    """
    fine = opacity_plan(_floor_series(tmp_path / "a", (6.8e-6,) * 3))
    coarse = opacity_plan(_floor_series(tmp_path / "b", (9.7e-3,) * 3))
    assert fine["unit_distance"] == pytest.approx(6.8e-6, rel=1e-6)
    assert coarse["unit_distance"] == pytest.approx(9.7e-3, rel=1e-6)
    assert coarse["unit_distance"] / fine["unit_distance"] > 1000
    # Neither is the old constant.
    assert abs(fine["unit_distance"] - 1e-5) > 1e-6


def test_constant_frames_are_counted_not_hidden(tmp_path):
    """A frame with nothing in it is reported, not quietly averaged away."""
    plan = opacity_plan(_floor_series(tmp_path, (1e-3,) * 3))
    assert plan["constant_frames"] == 1
    assert plan["frames_sampled"] >= 2


def test_plan_is_none_when_nothing_varies(tmp_path):
    """No plan rather than a fabricated one: the caller falls back loudly."""
    files = []
    for i in range(3):
        fn = f"s_{i:04d}.vti"
        write_vti(str(tmp_path / fn),
                  {"log10_plasma_density": np.full((8, 8, 8), -12.0,
                                                   np.float32)},
                  spacing=(1e-3,) * 3)
        files.append(fn)
    write_pvd(str(tmp_path / "s.pvd"), files, [0.0, 1.0, 2.0])
    assert opacity_plan(str(tmp_path / "s.pvd")) is None


def test_plan_does_not_choose_the_field_from_frame_zero(tmp_path):
    """The same frame-0 trap, at the level of the whole series."""
    files = []
    for i in range(5):
        a = np.full((8, 8, 8), -12.0, np.float32)
        if i:
            a[:3, :3, :3] = -7.0
        mat = np.zeros((8, 8, 8), np.float32)
        mat[:4] = 5.0                    # varies in EVERY frame, incl. 0
        fn = f"s_{i:04d}.vti"
        write_vti(str(tmp_path / fn),
                  {"log10_plasma_density": a, "material": mat},
                  spacing=(1e-3,) * 3)
        files.append(fn)
    write_pvd(str(tmp_path / "s.pvd"), files, list(range(5)))
    plan = opacity_plan(str(tmp_path / "s.pvd"),
                        prefer=("log10_plasma_density", "material"))
    assert plan["field"] == "log10_plasma_density"


def test_explicit_field_is_respected(tmp_path):
    pvd = _floor_series(tmp_path, (1e-3,) * 3)
    assert opacity_plan(pvd, field="log10_plasma_density")["field"] == \
        "log10_plasma_density"
