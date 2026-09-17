"""Tests for ejecta identification, fragment clustering and the particle scene.

Most of these build synthetic configurations with a *known* answer -- three
separated cubes are three fragments, however the clustering is implemented --
which is the only way to test a clustering algorithm without begging the
question.

The one that matters most is `test_cutoff_comes_from_the_bulk_not_the_subset`:
it guards a bug that was live in the first version of this module and that a
real MD frame exposed.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest

from hvi_emp.solvers.fragments import (CUTOFF_FACTOR, analyse_fragments,
                                       cluster, cutoff_sensitivity,
                                       identify_ejecta,
                                       nearest_neighbour_distance,
                                       size_distribution)
from hvi_emp.viz.paraview_scene import (PARTICLE_STYLE,
                                        write_particle_scene_script)

M_W = 3.0527e-25          # tungsten atom [kg]
A0 = 3.165e-10            # W bcc lattice constant


def _cube(n: int, spacing: float, origin=(0.0, 0.0, 0.0)) -> np.ndarray:
    """A simple-cubic block of n^3 points."""
    g = np.arange(n) * spacing
    p = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return p + np.asarray(origin)


class TestNearestNeighbour:

    def test_recovers_a_known_lattice_spacing(self):
        p = _cube(8, A0)
        assert nearest_neighbour_distance(p) == pytest.approx(A0, rel=1e-9)

    def test_handles_degenerate_input(self):
        assert nearest_neighbour_distance(np.zeros((0, 3))) == 0.0
        assert nearest_neighbour_distance(np.zeros((1, 3))) == 0.0

    def test_is_deterministic_on_large_input(self):
        """Sampled, but the sample must not change between calls -- a report
        that moves when you re-run it is not a report."""
        rng = np.random.default_rng(1)
        p = rng.normal(size=(60000, 3)) * 1e-9
        assert (nearest_neighbour_distance(p)
                == nearest_neighbour_distance(p))


class TestClustering:

    def test_three_separated_cubes_are_three_fragments(self):
        p = np.vstack([_cube(4, A0),
                       _cube(4, A0, (50 * A0, 0, 0)),
                       _cube(3, A0, (0, 50 * A0, 0))])
        labels, cut = cluster(p, cutoff=1.4 * A0)
        assert len(np.unique(labels)) == 3

    def test_one_block_is_one_fragment(self):
        labels, _ = cluster(_cube(6, A0), cutoff=1.4 * A0)
        assert len(np.unique(labels)) == 1

    def test_a_cutoff_below_the_spacing_shatters_everything(self):
        p = _cube(4, A0)
        labels, _ = cluster(p, cutoff=0.5 * A0)
        assert len(np.unique(labels)) == p.shape[0]

    def test_a_huge_cutoff_merges_everything(self):
        p = np.vstack([_cube(3, A0), _cube(3, A0, (50 * A0, 0, 0))])
        labels, _ = cluster(p, cutoff=100 * A0)
        assert len(np.unique(labels)) == 1

    def test_empty_and_single_inputs(self):
        assert cluster(np.zeros((0, 3)))[0].shape == (0,)
        assert cluster(np.zeros((1, 3)))[0].shape == (1,)

    def test_derived_cutoff_is_the_factor_times_the_spacing(self):
        _, cut = cluster(_cube(6, A0))
        assert cut == pytest.approx(CUTOFF_FACTOR * A0, rel=1e-6)


class TestEjectaIdentification:

    def test_selects_material_above_the_surface_moving_away(self):
        pos = np.array([[0, 0, -1e-9],        # buried
                        [0, 0, +1e-9],        # above, moving up   -> ejecta
                        [0, 0, +1e-9],        # above, moving down
                        [0, 0, -1e-9]])       # buried, moving up
        vel = np.array([[0, 0, 100.0], [0, 0, 100.0],
                        [0, 0, -100.0], [0, 0, 100.0]])
        got = identify_ejecta(pos, vel, surface=0.0)
        assert list(got) == [False, True, False, False]

    def test_margin_suppresses_surface_vibration(self):
        pos = np.array([[0, 0, 1e-11], [0, 0, 5e-9]])
        vel = np.array([[0, 0, 10.0], [0, 0, 10.0]])
        assert identify_ejecta(pos, vel, surface=0.0).sum() == 2
        assert identify_ejecta(pos, vel, surface=0.0, margin=1e-9).sum() == 1

    def test_min_speed_filters_slow_material(self):
        pos = np.array([[0, 0, 1e-9], [0, 0, 1e-9]])
        vel = np.array([[0, 0, 1.0], [0, 0, 5000.0]])
        assert identify_ejecta(pos, vel, surface=0.0,
                               min_speed=100.0).sum() == 1

    def test_axis_is_configurable(self):
        pos = np.array([[1e-9, 0, 0], [-1e-9, 0, 0]])
        vel = np.array([[100.0, 0, 0], [100.0, 0, 0]])
        assert identify_ejecta(pos, vel, surface=0.0, axis=0).sum() == 1

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError):
            identify_ejecta(np.zeros((4, 3)), np.zeros((3, 3)))


class TestAnalyseFragments:

    def test_masses_and_sizes_are_consistent(self):
        p = np.vstack([_cube(4, A0), _cube(3, A0, (60 * A0, 0, 0))])
        v = np.zeros_like(p)
        fs = analyse_fragments(p, v, M_W, cutoff=1.4 * A0)
        assert fs.n_fragments == 2
        assert list(fs.sizes) == [64, 27]          # ordered largest first
        assert np.allclose(fs.masses, fs.sizes * M_W)

    def test_fragment_zero_is_always_the_largest(self):
        p = np.vstack([_cube(2, A0), _cube(5, A0, (60 * A0, 0, 0))])
        fs = analyse_fragments(p, np.zeros_like(p), M_W, cutoff=1.4 * A0)
        assert fs.sizes[0] == 125

    def test_centre_of_mass_velocity_is_the_mean(self):
        p = _cube(3, A0)
        v = np.zeros_like(p)
        v[:, 2] = 1000.0
        fs = analyse_fragments(p, v, M_W, cutoff=1.4 * A0)
        assert fs.velocity[0, 2] == pytest.approx(1000.0)
        assert fs.speed[0] == pytest.approx(1000.0)

    def test_mask_restricts_what_is_clustered(self):
        p = np.vstack([_cube(4, A0), _cube(3, A0, (60 * A0, 0, 0))])
        mask = np.zeros(p.shape[0], dtype=bool)
        mask[64:] = True
        fs = analyse_fragments(p, np.zeros_like(p), M_W, mask=mask,
                               cutoff=1.4 * A0)
        assert fs.n_fragments == 1 and fs.sizes[0] == 27

    def test_empty_selection_returns_an_empty_set(self):
        p = _cube(3, A0)
        fs = analyse_fragments(p, np.zeros_like(p), M_W,
                               mask=np.zeros(p.shape[0], dtype=bool))
        assert fs.n_fragments == 0
        assert "no particles" in " ".join(fs.notes)

    def test_cutoff_comes_from_the_bulk_not_the_subset(self):
        """The bug a real MD frame exposed.

        Deriving the cutoff from the masked subset looks equivalent and is
        not: ejecta are dispersed, so their own nearest-neighbour distance is
        larger than the lattice's and *keeps growing* as they fly apart. A
        cutoff that grows with the cloud progressively merges fragments over
        a time series -- backwards.
        """
        bulk = _cube(10, A0)                         # dense lattice
        far = _cube(2, A0, (200 * A0, 0, 0)) + np.array([0.0, 0.0, 10 * A0])
        spread = far + np.arange(far.shape[0])[:, None] * 3 * A0
        p = np.vstack([bulk, spread])
        v = np.zeros_like(p)
        v[len(bulk):, 2] = 1000.0
        mask = np.zeros(p.shape[0], dtype=bool)
        mask[len(bulk):] = True

        fs = analyse_fragments(p, v, M_W, mask=mask)
        # the whole frame is dominated by the dense lattice
        assert fs.cutoff == pytest.approx(CUTOFF_FACTOR * A0, rel=0.02)
        # and NOT by the dispersed subset, whose spacing is much larger
        subset_nn = nearest_neighbour_distance(p[mask])
        assert fs.cutoff < CUTOFF_FACTOR * subset_nn * 0.9

    def test_the_cutoff_source_is_stated_in_the_notes(self):
        p = _cube(5, A0)
        fs = analyse_fragments(p, np.zeros_like(p), M_W)
        assert any("whole frame" in n for n in fs.notes)

    def test_a_dominant_fragment_is_flagged(self):
        p = _cube(6, A0)
        fs = analyse_fragments(p, np.zeros_like(p), M_W, cutoff=1.4 * A0)
        assert any("holds" in n and "%" in n for n in fs.notes)

    def test_radius_grows_with_mass(self):
        p = np.vstack([_cube(2, A0), _cube(5, A0, (60 * A0, 0, 0))])
        fs = analyse_fragments(p, np.zeros_like(p), M_W, cutoff=1.4 * A0)
        assert fs.radius[0] > fs.radius[1]


class TestSizeDistribution:

    def test_cumulative_mass_reaches_one(self):
        p = np.vstack([_cube(4, A0), _cube(3, A0, (60 * A0, 0, 0)),
                       _cube(2, A0, (0, 60 * A0, 0))])
        fs = analyse_fragments(p, np.zeros_like(p), M_W, cutoff=1.4 * A0)
        sd = size_distribution(fs)
        assert sd["mass_fraction_ge"][-1] == pytest.approx(1.0)
        assert list(sd["n_ge"]) == [1, 2, 3]

    def test_masses_are_descending(self):
        p = np.vstack([_cube(2, A0), _cube(4, A0, (60 * A0, 0, 0))])
        sd = size_distribution(analyse_fragments(p, np.zeros_like(p), M_W,
                                                 cutoff=1.4 * A0))
        assert list(sd["mass"]) == sorted(sd["mass"], reverse=True)

    def test_empty_input(self):
        p = _cube(2, A0)
        fs = analyse_fragments(p, np.zeros_like(p), M_W,
                               mask=np.zeros(p.shape[0], dtype=bool))
        assert size_distribution(fs)["mass"].size == 0


class TestCutoffSensitivity:

    def test_more_fragments_at_smaller_cutoff(self):
        p = _cube(5, A0)
        rows = cutoff_sensitivity(p, np.zeros_like(p), M_W)
        counts = [r["n_fragments"] for r in rows]
        assert counts == sorted(counts, reverse=True)

    def test_reports_the_cutoff_it_used(self):
        p = _cube(4, A0)
        for r in cutoff_sensitivity(p, np.zeros_like(p), M_W):
            assert r["cutoff"] == pytest.approx(r["factor"] * A0, rel=0.02)


class TestParticleScene:

    def test_script_is_valid_python(self, tmp_path):
        t = write_particle_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        ast.parse(t)

    def test_plasma_volume_only_when_a_grid_is_given(self, tmp_path):
        with_grid = write_particle_scene_script(
            "/tmp/a.pvd", str(tmp_path / "a.py"), grid_pvd="/tmp/g.pvd")
        without = write_particle_scene_script(
            "/tmp/a.pvd", str(tmp_path / "b.py"))
        assert "Additive" in with_grid and "GRID = os.path.join" in with_grid
        assert "Additive" not in without and "GRID = None" in without
        ast.parse(with_grid)
        ast.parse(without)

    def test_uses_point_gaussian_not_plain_points(self, tmp_path):
        """Plain square points alias into a moire pattern on a lattice, which
        looks like structure and is not."""
        t = write_particle_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        assert "Point Gaussian" in t

    def test_pins_the_gpu_volume_mapper_for_the_plasma(self, tmp_path):
        t = write_particle_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"),
                                        grid_pvd="/tmp/g.pvd")
        assert "GPU Based" in t

    def test_checks_the_array_exists_before_colouring(self, tmp_path):
        t = write_particle_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        assert "not present; available" in t

    def test_carries_the_inferred_ionisation_caveat(self, tmp_path):
        """The single most important thing on this picture: MD has no
        electrons, so any charge state in it was added afterwards."""
        t = write_particle_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        assert "IONISATION IS INFERRED" in t

    def test_every_styled_array_produces_a_valid_script(self, tmp_path):
        for name in PARTICLE_STYLE:
            t = write_particle_scene_script(
                "/tmp/a.pvd", str(tmp_path / f"{name}.py"), colour_by=name)
            ast.parse(t)
            assert f"COLOUR_BY = {name!r}" in t

    def test_log_scaled_arrays_are_flagged_as_such(self):
        """fragment_atoms spans decades; T_eV does not."""
        assert PARTICLE_STYLE["fragment_atoms"]["log"] is True
        assert PARTICLE_STYLE["T_eV"]["log"] is False

    def test_inferred_quantities_are_labelled_in_the_legend(self):
        for name in ("Zbar", "log10_electron_density"):
            assert "INFERRED" in PARTICLE_STYLE[name]["title"]

    def test_unknown_preset_is_rejected(self, tmp_path):
        with pytest.raises(KeyError):
            write_particle_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"),
                                        preset="cinematic")

    def test_movie_block_is_optional(self, tmp_path):
        plain = write_particle_scene_script("/tmp/a.pvd",
                                            str(tmp_path / "a.py"))
        movie = write_particle_scene_script("/tmp/a.pvd",
                                            str(tmp_path / "b.py"),
                                            movie_path="/tmp/f/x.png")
        assert "SaveAnimation" not in plain
        assert "SaveAnimation" in movie
        ast.parse(movie)
