"""Tests for the impact/plume scenes and the generated ParaView script.

ParaView is not installed here and these tests do not need it. What they can
check is the thing that actually goes wrong: the generated script asks
ParaView to colour by an array, and if that array is not in the `.vti` the
render is silently empty. So the central test parses the script, extracts
every array name it references, and asserts each one exists in the data.

The scene geometry tests guard the bug this work started from: the crater was
0.013 cells across on the default domain, so every render was of the plume
and nothing else.
"""

from __future__ import annotations

import ast
import os
import re

import pytest

from hvi_emp import run_scenario
from hvi_emp.viz.paraview_scene import (FIELD_ALIASES, FIELD_STYLE, PRESETS,
                                        write_scene_for, write_scene_script)
from hvi_emp.viz.vti import (IMPACT_FIELDS, MINIMAL_FIELDS, QUALITY, SCENES,
                             ImpactVolume, estimate_memory_GB, scene_kwargs,
                             write_scene)
from test_viz import read_vti


@pytest.fixture(scope="module")
def scenario():
    return run_scenario("Fe", "Al", mass=1e-12, velocity=50e3, t_end=1e-5)


# ---------------------------------------------------------------------------
# The scale problem this whole feature exists to solve
# ---------------------------------------------------------------------------

class TestSceneGeometry:

    def test_impact_scene_resolves_the_crater(self, scenario):
        """The one number that matters. Below 1 there is nothing to see."""
        vol = ImpactVolume(scenario, shape=(128, 128, 128),
                           **scene_kwargs(scenario, "impact"))
        assert vol.crater_radius / vol.spacing[0] > 5.0

    def test_plume_scene_does_not_and_that_is_expected(self, scenario):
        """Recorded so nobody 'fixes' the plume scene by shrinking it.

        The plume box has to hold a metre of trajectory; the crater is 74 um.
        These are different films.
        """
        vol = ImpactVolume(scenario, shape=(128, 128, 128),
                           **scene_kwargs(scenario, "plume"))
        assert vol.crater_radius / vol.spacing[0] < 0.1

    def test_the_two_scenes_differ_by_orders_of_magnitude(self, scenario):
        a = ImpactVolume(scenario, shape=(64, 64, 64),
                         **scene_kwargs(scenario, "impact"))
        b = ImpactVolume(scenario, shape=(64, 64, 64),
                         **scene_kwargs(scenario, "plume"))
        assert b.spacing[0] / a.spacing[0] > 100.0

    def test_explicit_domain_survives_a_frame_call(self, scenario):
        """The regression that made the impact scene impossible.

        `domain_at` used to ignore an explicitly-supplied domain, so the first
        `frame()` rebuilt the grid at plume scale and silently discarded it.
        """
        vol = ImpactVolume(scenario, shape=(32, 32, 32), domain=2.0e-4,
                           z_below=1.0e-4)
        before = vol.domain
        vol.frame(1e-9)
        assert vol.domain == before
        # linspace over n points spans n-1 intervals
        assert vol.spacing[0] == pytest.approx(2.0 * 2.0e-4 / 31)

    def test_explicit_domain_survives_every_frame_in_a_series(self, scenario):
        vol = ImpactVolume(scenario, shape=(24, 24, 24), domain=2.0e-4,
                           z_below=1.0e-4)
        for t in (-1e-10, 1e-12, 1e-9, 1e-7, 1e-5):
            vol.frame(float(t))
            assert vol.domain == pytest.approx(2.0e-4)

    def test_shock_is_visible_while_the_front_is_inside_the_box(self,
                                                                scenario):
        """Zero at every frame was the symptom of the discarded domain."""
        vol = ImpactVolume(scenario, shape=(64, 64, 64),
                           **scene_kwargs(scenario, "impact"))
        t = vol.shock_transit_time() * 0.5
        assert vol.frame(t)["shock_pressure"].max() > 0.0

    def test_crater_grows_across_the_impact_scene(self, scenario):
        vol = ImpactVolume(scenario, shape=(64, 64, 64),
                           **scene_kwargs(scenario, "impact"))
        cell = vol.spacing[0]
        t_end = 3.0 * max(vol.shock_transit_time(), vol.crater_growth_time())
        times = vol.times(10, mode="log", t_end=t_end)
        r = [vol.crater_radius_at(float(t)) / cell for t in times]
        assert r == sorted(r)
        assert r[-1] > 5.0 and r[-1] > 10.0 * max(r[2], 1e-9)

    def test_impact_scene_stops_before_the_static_tail(self, scenario):
        """Running an impact-scale box to 10 us wastes most of the frames."""
        vol = ImpactVolume(scenario, shape=(32, 32, 32),
                           **scene_kwargs(scenario, "impact"))
        t_end = 3.0 * max(vol.shock_transit_time(), vol.crater_growth_time())
        assert t_end < 1e-6
        assert t_end > vol.crater_growth_time()

    def test_t_end_truncates_and_is_validated(self, scenario):
        vol = ImpactVolume(scenario, shape=(16, 16, 16))
        assert vol.times(8, t_end=1e-8)[-1] <= 1e-8
        with pytest.raises(ValueError):
            vol.times(8, t_end=-1.0)

    def test_unknown_scene_is_rejected(self, scenario):
        with pytest.raises(KeyError):
            scene_kwargs(scenario, "explosion")
        with pytest.raises(KeyError):
            write_scene(scenario, "/tmp/nope", scene="explosion")


class TestWriteScene:

    def test_writes_files_and_reports_the_crater_scale(self, tmp_path,
                                                       scenario):
        res = write_scene(scenario, str(tmp_path / "s"), scene="impact",
                          n_frames=4, shape=(24, 24, 24), verbose=False)
        assert os.path.isfile(res["pvd"])
        assert len(res["files"]) == 4
        assert res["cells_across_crater"] > 1.0
        assert res["scene"] == "impact"

    def test_every_requested_field_reaches_the_file(self, tmp_path, scenario):
        res = write_scene(scenario, str(tmp_path / "s"), scene="impact",
                          n_frames=3, shape=(16, 16, 16), verbose=False)
        got = set(read_vti(res["files"][-1])["fields"])
        assert set(IMPACT_FIELDS) <= got, set(IMPACT_FIELDS) - got

    def test_pvd_header_records_the_crater_scale(self, tmp_path, scenario):
        """So an image can be traced back to whether it could show a crater."""
        res = write_scene(scenario, str(tmp_path / "s"), scene="impact",
                          n_frames=2, shape=(16, 16, 16), verbose=False)
        head = open(res["pvd"]).read()
        assert "cells across" in head
        assert "PROVENANCE" in head

    def test_provenance_only_lists_fields_actually_written(self, tmp_path,
                                                           scenario):
        res = write_scene(scenario, str(tmp_path / "s"), scene="impact",
                          n_frames=2, shape=(16, 16, 16),
                          fields=("log10_plasma_density", "material"),
                          verbose=False)
        head = open(res["pvd"]).read()
        assert "log10_plasma_density" in head
        assert "smoke_density" not in head

    def test_overrides_reach_impactvolume(self, tmp_path, scenario):
        res = write_scene(scenario, str(tmp_path / "s"), scene="impact",
                          n_frames=2, shape=(16, 16, 16), domain=5e-4,
                          verbose=False)
        assert res["volume"].domain == pytest.approx(5e-4)

    @pytest.mark.parametrize("scene", sorted(SCENES))
    def test_all_scenes_run(self, tmp_path, scenario, scene):
        res = write_scene(scenario, str(tmp_path / scene), scene=scene,
                          n_frames=3, shape=(16, 16, 16), verbose=False)
        assert len(res["files"]) == 3


# ---------------------------------------------------------------------------
# The generated ParaView script
# ---------------------------------------------------------------------------

def _arrays_referenced(script_text: str) -> set:
    """Array names the script will ask ParaView for."""
    names = set()
    for var in ("COLOUR_BY", "MATERIAL"):
        m = re.search(rf"^{var} = '([^']+)'", script_text, re.M)
        if m:
            names.add(m.group(1))
    return names


class TestGeneratedScript:

    def test_it_is_valid_python(self, tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        ast.parse(t)

    def test_every_array_it_references_exists_in_the_data(self, tmp_path,
                                                          scenario):
        """The silent failure this suite exists for.

        Colouring by a missing array gives an empty render, not an error, and
        looks exactly like a physics bug.
        """
        res = write_scene(scenario, str(tmp_path / "s"), scene="impact",
                          n_frames=2, shape=(16, 16, 16), verbose=False)
        script = str(tmp_path / "s" / "scene.py")
        text = write_scene_for(res, script)
        present = set(read_vti(res["files"][-1])["fields"])
        missing = _arrays_referenced(text) - present
        assert not missing, missing

    def test_it_colours_by_the_log_field_not_the_raw_density(self, tmp_path,
                                                            scenario):
        """Raw plasma_density on a linear map is the flat-blob bug."""
        res = write_scene(scenario, str(tmp_path / "s"), scene="impact",
                          n_frames=2, shape=(16, 16, 16), verbose=False)
        text = write_scene_for(res, str(tmp_path / "s" / "scene.py"))
        assert "COLOUR_BY = 'log10_plasma_density'" in text

    def test_it_does_not_log_scale_an_already_log_field(self, tmp_path):
        """Log-scaling log10(rho) takes the log of a negative number."""
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"),
                               colour_by="log10_plasma_density")
        assert "lut.UseLogScale = 0" in t
        t2 = write_scene_script("/tmp/a.pvd", str(tmp_path / "s2.py"),
                                colour_by="plasma_density")
        assert "lut.UseLogScale = 1" in t2

    def test_it_rescales_over_time_not_per_frame(self, tmp_path):
        """Per-frame rescaling makes a decaying plume look static."""
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        assert "RescaleTransferFunctionToDataRangeOverTime" in t

    def test_it_writes_a_state_file_rather_than_hand_rolling_one(self,
                                                                 tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        assert "SaveState(STATE)" in t
        assert t.count("<ServerManagerState") == 0     # no hand-written XML

    def test_it_fails_loudly_when_run_by_the_wrong_python(self, tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        assert "from paraview.simple import" in t
        assert "pvpython" in t
        assert "sys.exit" in t

    def test_it_checks_the_array_exists_before_colouring(self, tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        assert "if COLOUR_BY not in available:" in t

    def test_pvd_path_is_relative_so_the_pair_can_be_moved(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        t = write_scene_script(str(d / "impact.pvd"), str(d / "scene.py"))
        assert "'impact.pvd'" in t
        assert str(tmp_path) not in t.split("Colouring by")[1][:400]

    def test_talk_preset_uses_bigger_fonts_than_paper(self, tmp_path):
        assert (PRESETS["talk"]["font"] > PRESETS["paper"]["font"])
        a = write_scene_script("/tmp/a.pvd", str(tmp_path / "a.py"),
                               preset="talk")
        b = write_scene_script("/tmp/a.pvd", str(tmp_path / "b.py"),
                               preset="paper")
        assert f"bar.LabelFontSize = {PRESETS['talk']['font']}" in a
        assert f"bar.LabelFontSize = {PRESETS['paper']['font']}" in b

    def test_target_surface_can_be_turned_off(self, tmp_path):
        on = write_scene_script("/tmp/a.pvd", str(tmp_path / "a.py"),
                                show_target=True)
        off = write_scene_script("/tmp/a.pvd", str(tmp_path / "b.py"),
                                 show_target=False)
        assert "target" in on and "Contour" in on
        assert "Contour" not in off

    def test_comoving_scene_drops_the_target_automatically(self, tmp_path,
                                                           scenario):
        """There is no target in a co-moving box; drawing one puts a slab of
        aluminium through the middle of the plume."""
        res = write_scene(scenario, str(tmp_path / "s"),
                          scene="plume_comoving", n_frames=2,
                          shape=(16, 16, 16), verbose=False)
        text = write_scene_for(res, str(tmp_path / "s" / "scene.py"))
        assert "Contour" not in text

    def test_illustrative_target_is_labelled_as_such(self, tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"),
                               show_target=True)
        assert "ILLUSTRATIVE" in t
        assert "Cour-Palais" in t

    def test_parametric_fields_say_so_in_the_legend(self, tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"),
                               colour_by="shock_pressure")
        assert "PARAMETRIC" in t

    def test_movie_block_is_optional(self, tmp_path):
        plain = write_scene_script("/tmp/a.pvd", str(tmp_path / "a.py"))
        assert "SaveAnimation" not in plain
        withmov = write_scene_script("/tmp/a.pvd", str(tmp_path / "b.py"),
                                     movie_path="/tmp/out.avi")
        assert "SaveAnimation" in withmov
        ast.parse(withmov)

    def test_script_is_executable(self, tmp_path):
        p = tmp_path / "s.py"
        write_scene_script("/tmp/a.pvd", str(p))
        assert os.access(p, os.X_OK)

    @pytest.mark.parametrize("bad,kw", [
        ("preset", {"preset": "cinematic"}),
        ("source", {"source": "isale"}),
    ])
    def test_unknown_preset_or_source_is_rejected(self, tmp_path, bad, kw):
        with pytest.raises(KeyError):
            write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"), **kw)


class TestM2CAliases:
    """The same scene has to work on a solved hydrocode result."""

    def test_aliases_rename_the_arrays(self, tmp_path):
        t = write_scene_script("/tmp/m.pvd", str(tmp_path / "s.py"),
                               source="m2c", colour_by="ionisation")
        assert "COLOUR_BY = 'MeanCharge'" in t
        assert "MATERIAL = 'MaterialID'" in t

    def test_identity_mapping_for_our_own_output(self, tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"),
                               source="hvi_emp", colour_by="ionisation")
        assert "COLOUR_BY = 'ionisation'" in t

    def test_m2c_names_match_what_the_bridge_asks_m2c_to_write(self):
        """Guards the two halves against drifting apart.

        `m2c_stage1` turns on `MeanCharge` and `ElectronDensity` in the
        Output block; the alias table has to use those spellings.
        """
        from hvi_emp.solvers import m2c_stage1
        alias = FIELD_ALIASES["m2c"]
        for name in ("MeanCharge", "ElectronDensity"):
            assert name in alias.values()
            assert name in m2c_stage1.VERIFIED_KEYWORDS

    def test_every_alias_target_is_a_plausible_m2c_output(self):
        from hvi_emp.solvers import m2c_stage1
        known = set(m2c_stage1.VERIFIED_KEYWORDS)
        for ours, theirs in FIELD_ALIASES["m2c"].items():
            assert theirs in known, (ours, theirs)


class TestStyleTable:

    def test_every_impact_field_has_a_style_or_falls_back_safely(self,
                                                                 tmp_path):
        for f in IMPACT_FIELDS:
            t = write_scene_script("/tmp/a.pvd", str(tmp_path / f"{f}.py"),
                                   colour_by=f)
            ast.parse(t)
            assert f"COLOUR_BY = '{f}'" in t

    def test_log_fields_are_not_double_logged(self):
        for name, style in FIELD_STYLE.items():
            if name.startswith("log10_"):
                assert style["log"] is False, name

    def test_scene_colour_by_fields_are_written_by_that_scene(self):
        """A scene must not name a default field its own writer omits."""
        for name, spec in SCENES.items():
            assert spec["colour_by"] in IMPACT_FIELDS, name


# ---------------------------------------------------------------------------
# GPU rendering and the memory guard
# ---------------------------------------------------------------------------

class TestGPURendering:
    """ParaView is absent here, so these check the *instructions*, not a
    render. The property names cannot be verified without it, which is why
    the generated script wraps every one of them."""

    def test_gpu_is_the_default_renderer(self, tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        assert "GPU Based" in t

    def test_smart_is_available_but_warns_about_the_cpu_fallback(self,
                                                                  tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"),
                               renderer="smart")
        assert "GPU Based" not in t
        assert "CPU mapper" in t

    def test_index_keeps_the_gpu_mapper_as_the_fallback(self, tmp_path):
        """IndeX is a plugin and may be absent; losing it must not leave the
        scene on whatever ParaView picked by default."""
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"),
                               renderer="index")
        assert "GPU Based" in t
        assert t.index("GPU Based") < t.index("pvNVIDIAIndeX")
        assert "staying on" in t

    def test_every_gpu_setting_is_wrapped(self, tmp_path):
        """A rendering hint must never take the whole scene down."""
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"),
                               renderer="index")
        body = t.split("renderer setup:")[1].split("# ---")[0]
        for line in body.splitlines():
            if "setattr(disp" in line:
                assert "_try(" in line or "lambda" in line, line

    def test_it_defines_the_try_helper_before_using_it(self, tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        assert t.index("def _try(") < t.index('_try("shading"')

    def test_shading_is_on(self, tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        assert '_try("shading"' in t

    @pytest.mark.parametrize("renderer", ["gpu", "index", "smart"])
    def test_all_renderers_produce_valid_python(self, tmp_path, renderer):
        ast.parse(write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"),
                                     renderer=renderer))

    def test_unknown_renderer_is_rejected(self, tmp_path):
        with pytest.raises(KeyError):
            write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"),
                               renderer="raytrace")

    def test_movie_block_is_headless_and_survives_encoder_failure(self,
                                                                   tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"),
                               movie_path="/tmp/f/impact.png")
        assert "pvbatch" in t
        assert "ffmpeg" in t
        # the .pvsm must already be written by the time the movie is tried
        assert t.index("SaveState(STATE)") < t.index("SaveAnimation")
        assert "animation failed" in t


class TestMemoryEstimate:

    def test_estimate_matches_the_measured_bytes_per_cell(self):
        m = estimate_memory_GB((256, 256, 256))
        assert m["cells"] == 256 ** 3
        assert m["peak_GB"] == pytest.approx(256 ** 3 * 175 / 1e9)

    def test_vram_is_far_below_peak_host_memory(self):
        """The point of the whole table: the GPU is not the constraint."""
        m = estimate_memory_GB((512, 512, 512))
        assert m["peak_GB"] / m["vram_GB"] > 20.0

    def test_quality_table_ram_figures_agree_with_the_estimator(self):
        for name, spec in QUALITY.items():
            m = estimate_memory_GB((spec["n"],) * 3)
            assert m["peak_GB"] == pytest.approx(spec["ram_GB"],
                                                rel=0.02), name

    def test_quality_presets_increase_monotonically(self):
        ns = [QUALITY[k]["n"] for k in
              ("draft", "talk", "high", "workstation", "extreme")]
        assert ns == sorted(ns)

    def test_guard_refuses_a_grid_that_will_not_fit(self, tmp_path, scenario):
        with pytest.raises(MemoryError) as e:
            write_scene(scenario, str(tmp_path / "s"), scene="impact",
                        n_frames=2, shape=(4096, 4096, 4096), verbose=False)
        msg = str(e.value)
        assert "GB peak" in msg
        assert "allow_oversize" in msg      # names the escape hatch
        assert "MINIMAL_FIELDS" in msg      # and a real alternative

    def test_guard_can_be_overridden_deliberately(self, tmp_path, scenario):
        """Refusing is a default, not a policy -- the estimate can be wrong."""
        import inspect
        sig = inspect.signature(write_scene)
        assert sig.parameters["allow_oversize"].default is False

    def test_quality_preset_sets_the_grid(self, tmp_path, scenario):
        res = write_scene(scenario, str(tmp_path / "s"), scene="impact",
                          n_frames=2, quality="draft", verbose=False)
        assert res["volume"].shape == (QUALITY["draft"]["n"],) * 3

    def test_explicit_shape_beats_quality(self, tmp_path, scenario):
        res = write_scene(scenario, str(tmp_path / "s"), scene="impact",
                          n_frames=2, shape=(20, 20, 20), quality="extreme",
                          verbose=False)
        assert res["volume"].shape == (20, 20, 20)

    def test_unknown_quality_is_rejected(self, tmp_path, scenario):
        with pytest.raises(KeyError):
            write_scene(scenario, str(tmp_path / "s"), scene="impact",
                        quality="cinematic", verbose=False)

    def test_minimal_fields_are_a_real_subset_that_still_renders(self,
                                                                 tmp_path,
                                                                 scenario):
        assert set(MINIMAL_FIELDS) < set(IMPACT_FIELDS)
        res = write_scene(scenario, str(tmp_path / "s"), scene="impact",
                          n_frames=2, shape=(16, 16, 16),
                          fields=MINIMAL_FIELDS, verbose=False)
        text = write_scene_for(res, str(tmp_path / "s" / "scene.py"))
        present = set(read_vti(res["files"][-1])["fields"])
        assert not (_arrays_referenced(text) - present)

    def test_memory_is_reported_back_to_the_caller(self, tmp_path, scenario):
        res = write_scene(scenario, str(tmp_path / "s"), scene="impact",
                          n_frames=2, shape=(16, 16, 16), verbose=False)
        assert set(res["memory"]) == {"cells", "peak_GB", "output_GB",
                                      "vram_GB"}


# ---------------------------------------------------------------------------
# What the rendered images exposed
# ---------------------------------------------------------------------------

class TestPlumeDeparture:
    """The plume leaves an impact-scale box far sooner than intuition says.

    It drifts downrange at roughly half the impact speed while expanding at
    only a few km/s -- 25.0 against 3.1 km/s on the reference case. A box
    sized to show a 74 um crater is emptied in nanoseconds, and an animation
    that runs past that point renders most of its frames with nothing in the
    domain, which reads as a stalled simulation.
    """

    def test_departure_is_much_earlier_than_the_shock_transit(self, scenario):
        vol = ImpactVolume(scenario, shape=(32, 32, 32),
                           **scene_kwargs(scenario, "impact"))
        t_shock = 3.0 * max(vol.shock_transit_time(),
                            vol.crater_growth_time())
        assert vol.plume_departure_time() < t_shock

    def test_the_impact_scene_stops_at_departure(self, scenario):
        res = write_scene(scenario, "/tmp/_dep", scene="impact", n_frames=4,
                          shape=(16, 16, 16), verbose=False)
        vol = res["volume"]
        assert res["t_end"] <= vol.plume_departure_time() * 1.001

    def test_the_plume_is_still_in_the_box_at_the_last_frame(self, scenario):
        """The whole point: no frame should be empty."""
        import numpy as np
        res = write_scene(scenario, "/tmp/_dep2", scene="impact", n_frames=6,
                          shape=(20, 20, 20), verbose=False)
        vol = res["volume"]
        last = vol.frame(float(res["t_end"]))
        assert last["plasma_density"].max() > 0.0

    def test_truncation_is_explained_not_silent(self, scenario):
        res = write_scene(scenario, "/tmp/_dep3", scene="impact", n_frames=3,
                          shape=(16, 16, 16), verbose=False)
        notes = " ".join(res["volume"].notes)
        assert "leaves the box" in notes and "scene='plume'" in notes

    def test_departure_scales_with_box_size(self, scenario):
        small = ImpactVolume(scenario, shape=(16,) * 3, domain=1e-4,
                             z_below=5e-5)
        big = ImpactVolume(scenario, shape=(16,) * 3, domain=1e-3,
                           z_below=5e-4)
        assert small.plume_departure_time() < big.plume_departure_time()


class TestProjectileFraction:
    """The model mixes projectile and target into one homogeneous plume.

    That is a real limitation, and a viewer who sees an undifferentiated
    plume has to *infer* it from the picture. Writing the field makes it
    explicit instead.
    """

    def test_the_field_is_written(self, tmp_path, scenario):
        res = write_scene(scenario, str(tmp_path / "s"), scene="impact",
                          n_frames=2, shape=(16, 16, 16), verbose=False)
        assert "projectile_fraction" in read_vti(res["files"][-1])["fields"]

    def test_it_is_uniform_wherever_there_is_plume(self, scenario):
        import numpy as np
        vol = ImpactVolume(scenario, shape=(24,) * 3,
                           **scene_kwargs(scenario, "impact"))
        f = vol.frame(2e-9)
        pf = f["projectile_fraction"]
        live = pf > 0
        assert live.any()
        assert np.allclose(pf[live], pf[live][0])

    def test_it_equals_stage_one_s_own_mass_weight(self, scenario):
        vol = ImpactVolume(scenario, shape=(24,) * 3,
                           **scene_kwargs(scenario, "impact"))
        pf = vol.frame(2e-9)["projectile_fraction"]
        w = scenario.impact.diagnostics["w_projectile"]
        assert pf[pf > 0][0] == pytest.approx(w, rel=1e-5)

    def test_its_provenance_says_uniform_in_capitals(self):
        """So it survives being skim-read."""
        from hvi_emp.viz.vti import FIELD_PROVENANCE
        tag = FIELD_PROVENANCE["projectile_fraction"]
        assert "UNIFORM" in tag and "no spatial mixing" in tag

    def test_the_material_tag_admits_the_shapes_are_primitives(self):
        from hvi_emp.viz.vti import FIELD_PROVENANCE
        tag = FIELD_PROVENANCE["material"]
        assert "SPHERICAL CAP" in tag and "GAUSSIAN ELLIPSOID" in tag


class TestCutaway:

    def test_clip_is_on_by_default(self, tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        assert "cutaway" in t and "ClipType" in t

    def test_clip_can_be_turned_off(self, tmp_path):
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"),
                               clip=False)
        assert "cutaway" not in t
        ast.parse(t)

    def test_clip_comes_after_the_camera_centre_is_known(self, tmp_path):
        """The clip plane is placed at the data centre, so it cannot be
        emitted before cx/cy/cz exist."""
        t = write_scene_script("/tmp/a.pvd", str(tmp_path / "s.py"))
        assert t.index("cx = 0.5") < t.index("_clip.ClipType.Origin")

    def test_both_variants_are_valid_python(self, tmp_path):
        for c in (True, False):
            ast.parse(write_scene_script("/tmp/a.pvd",
                                         str(tmp_path / f"s{c}.py"), clip=c))
