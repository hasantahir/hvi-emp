"""Tests for the M2C Stage-1 bridge.

The unit conversion is the highest-risk part of this bridge by a wide margin:
M2C's input is in mm-g-s-K-A, a wrong factor produces a run that *completes*
and is wrong by a power of a thousand, and nothing in the output announces
it. So the first block of tests pins every conversion against numbers M2C
itself prints in its shipped hypervelocity-impact deck. If one of those
assertions fails, the deck we generate is wrong, not the test.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from hvi_emp.materials import get_material
from hvi_emp.solvers import m2c_stage1 as m2c


def _cfg(**kw):
    base = dict(projectile=get_material("al"), target=get_material("al"),
                diameter=1.0e-3, velocity=1.0e4)
    base.update(kw)
    return m2c.M2CConfig(**base)


# ---------------------------------------------------------------------------
# Units -- pinned against M2C's own shipped deck
# ---------------------------------------------------------------------------

class TestUnits:
    """Every number here is quoted in M2C's Tests/HVImpactShafquatIslam deck.

    Not derived by us, not rounded: these are the literal values that appear
    in a deck M2C runs. Reproducing them is the only evidence available,
    short of running the code, that the conversions are right.
    """

    def test_tantalum_density_matches_the_deck(self):
        # The deck's tantalum: 16650 kg/m^3 -> 16.65e-3 g/mm^3
        assert m2c.si_to_m2c(16650.0, "density") == pytest.approx(16.65e-3,
                                                                  rel=1e-12)

    def test_planck_constant_matches_the_deck(self):
        # deck: PlanckConstant = 6.62607004e-25 (CODATA-2014 value; we carry
        # the exact 2019 definition, so they agree to 8 significant figures)
        assert m2c.M2C_CONSTANTS["PlanckConstant"] == pytest.approx(
            6.62607004e-25, rel=1e-8)

    def test_electron_mass_matches_the_deck(self):
        assert m2c.M2C_CONSTANTS["ElectronMass"] == pytest.approx(
            9.10938356e-28, rel=1e-7)

    def test_boltzmann_constant_matches_the_deck(self):
        assert m2c.M2C_CONSTANTS["BoltzmannConstant"] == pytest.approx(
            1.38064852e-14, rel=1e-7)

    def test_electron_charge_is_unchanged(self):
        # C = A s in both systems, so this one must NOT be scaled. A "tidy-up"
        # that gives every constant a factor is exactly the bug this catches.
        assert m2c.M2C_CONSTANTS["ElectronCharge"] == pytest.approx(
            1.602176634e-19, rel=1e-12)

    def test_specific_heat_matches_the_deck(self):
        # deck: SpecificHeatAtConstantVolume = 139e6 for c_v = 139 J/(kg K)
        assert m2c.si_to_m2c(139.0, "specific_heat") == pytest.approx(139.0e6)

    def test_velocity_matches_the_deck(self):
        # deck: 6 km/s written as 6.0e6 mm/s
        assert m2c.si_to_m2c(6000.0, "velocity") == pytest.approx(6.0e6)

    def test_pressure_is_unchanged(self):
        # kg/(m s^2) and g/(mm s^2) are the same unit; 1 atm stays 1e5.
        assert m2c.si_to_m2c(1.0e5, "pressure") == pytest.approx(1.0e5)

    def test_round_trip_is_exact_for_every_kind(self):
        for kind in m2c.TO_M2C:
            v = 3.7e5
            assert m2c.m2c_to_si(m2c.si_to_m2c(v, kind), kind) == \
                pytest.approx(v, rel=1e-12), kind

    def test_arrays_round_trip_too(self):
        a = np.array([1.0, 2.5, 1e7])
        back = m2c.m2c_to_si(m2c.si_to_m2c(a, "density"), "density")
        assert np.allclose(back, a, rtol=1e-12)

    def test_unknown_unit_kind_is_an_error_not_a_silent_pass_through(self):
        # Silently returning the value unchanged would be the worst possible
        # behaviour here: a wrong deck with no diagnostic.
        with pytest.raises(KeyError):
            m2c.si_to_m2c(1.0, "furlongs")
        with pytest.raises(KeyError):
            m2c.m2c_to_si(1.0, "furlongs")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfig:

    def test_normal_incidence_is_axisymmetric(self):
        assert _cfg().is_3d is False
        assert "2-D" in _cfg().estimate_resources()["dimensionality"]

    def test_oblique_incidence_forces_three_dimensions(self):
        # This is the entire reason the bridge exists: hvi_emp's reduced
        # chain cannot produce an asymmetric plume at any angle.
        c = _cfg(angle_deg=45.0)
        assert c.is_3d is True
        assert "3-D" in c.estimate_resources()["dimensionality"]

    def test_oblique_incidence_is_flagged_in_the_notes(self):
        assert any("3-D Cartesian" in n for n in _cfg(angle_deg=30.0).notes)
        assert _cfg().notes == []

    def test_three_d_gets_its_own_mesh_defaults(self):
        # Cubing the axisymmetric defaults is ~461M cells / 86 GB.
        # Generating that silently would be worse than refusing.
        assert _cfg().cells_per_radius == m2c.DEFAULTS_2D["cells_per_radius"]
        c3 = _cfg(angle_deg=45.0)
        assert c3.cells_per_radius == m2c.DEFAULTS_3D["cells_per_radius"]
        assert c3.estimate_resources()["memory_GB"] < m2c.MEMORY_WARN_GB

    def test_explicit_resolution_is_respected_and_warned_about(self):
        """An explicit 3-D resolution is honoured, and its cost is stated.

        The note used to mention GB because the (uniform-mesh) estimate put
        this at 2.6 TB. With the graded count it is ~86 GB, under the
        single-node guide, so the warning is now about cell count rather
        than memory. Either way it must not be silent.
        """
        c = _cfg(angle_deg=45.0, cells_per_radius=50.0, domain_radii=24.0)
        assert c.cells_per_radius == 50.0        # not overridden
        assert c.notes                           # but not silent either
        assert any("cells" in n for n in c.notes)

    def test_memory_warning_still_fires_when_it_should(self):
        """The GB guard must not have been silently disabled by the fix."""
        c = _cfg(angle_deg=45.0, cells_per_radius=200.0, domain_radii=40.0)
        assert c.estimate_resources()["memory_GB"] > m2c.MEMORY_WARN_GB
        assert any("GB" in n for n in c.notes)

    def test_projectile_mass_is_the_sphere_mass(self):
        c = _cfg(diameter=2.0e-3, projectile=get_material("fe"))
        expected = 7870.0 * np.pi * (2.0e-3) ** 3 / 6.0
        assert c.mass == pytest.approx(expected, rel=1e-12)

    def test_end_time_scales_with_size_and_target_sound_speed(self):
        assert _cfg(diameter=2.0e-3).t_end == pytest.approx(
            2.0 * _cfg(diameter=1.0e-3).t_end)

    def test_explicit_end_time_wins(self):
        assert _cfg(t_end=1.0e-6).t_end == 1.0e-6

    @pytest.mark.parametrize("kw", [
        dict(diameter=-1.0), dict(velocity=0.0), dict(angle_deg=90.0),
        dict(angle_deg=-5.0), dict(ionisation="maybe"),
        dict(depression_model="Handwaving"), dict(projectile_shape="cube"),
        dict(cells_per_radius=2.0), dict(domain_radii=1.0),
    ])
    def test_bad_input_is_rejected(self, kw):
        with pytest.raises(ValueError):
            _cfg(**kw)

    def test_unknown_ambient_gas_is_rejected_at_construction(self):
        # Not at write time, when the user has walked away.
        with pytest.raises(KeyError):
            _cfg(ambient="unobtainium")

    def test_ambient_density_follows_the_ideal_gas_law(self):
        c = _cfg(ambient="ar", ambient_pressure=101325.0)
        # argon at 1 atm, 300 K is about 1.63 kg/m^3
        assert c.ambient_density() == pytest.approx(1.62, rel=0.02)

    def test_lower_pressure_gives_proportionally_lower_density(self):
        hi = _cfg(ambient="ar", ambient_pressure=100.0).ambient_density()
        lo = _cfg(ambient="ar", ambient_pressure=10.0).ambient_density()
        assert hi == pytest.approx(10.0 * lo, rel=1e-12)


class TestConfigFromScenario:

    def test_dict_scenario(self):
        c = m2c.config_from_scenario(
            {"projectile": "fe", "target": "al", "diameter": 1e-3,
             "velocity": 2e4, "angle_deg": 30.0})
        assert c.projectile.name == "Fe" and c.is_3d

    def test_object_scenario(self):
        class S:
            projectile, target = "al", "al"
            diameter, velocity, angle_deg = 5e-4, 1.5e4, 0.0
        c = m2c.config_from_scenario(S())
        assert c.velocity == 1.5e4 and not c.is_3d

    def test_overrides_win(self):
        c = m2c.config_from_scenario(
            {"projectile": "al", "target": "al", "diameter": 1e-3,
             "velocity": 1e4}, ionisation="off", cells_per_radius=8)
        assert c.ionisation == "off" and c.cells_per_radius == 8

    def test_missing_diameter_names_what_it_looked_for(self):
        with pytest.raises(KeyError) as e:
            m2c.config_from_scenario({"projectile": "al", "velocity": 1e4})
        assert "diameter" in str(e.value)


# ---------------------------------------------------------------------------
# The generated deck
# ---------------------------------------------------------------------------

class TestDeck:

    def test_deck_declares_its_unit_system_on_the_first_line(self, tmp_path):
        text = m2c.write_input(_cfg(), str(tmp_path / "input.st"))
        assert text.splitlines()[0].strip() == "//Units: mm, g, s, K, A"

    def test_braces_balance(self, tmp_path):
        for cfg in (_cfg(), _cfg(angle_deg=45.0, ambient="ar"),
                    _cfg(ionisation="off"), _cfg(projectile_shape="rod")):
            text = m2c.write_input(cfg, str(tmp_path / "i.st"))
            code = "\n".join(ln.split("//")[0] for ln in text.splitlines())
            assert code.count("{") == code.count("}"), cfg.projectile_shape

    def test_every_block_the_deck_needs_is_present(self, tmp_path):
        text = m2c.write_input(_cfg(), str(tmp_path / "input.st"))
        for block in ("under Mesh", "under Equations", "under Ionization",
                      "under InitialCondition", "under GeometricEntities",
                      "under BoundaryConditions", "under Space",
                      "under MultiPhase", "under Time", "under Output"):
            assert block in text, block

    def test_material_block_carries_the_converted_eos(self, tmp_path):
        al = get_material("al")
        text = m2c.write_input(_cfg(), str(tmp_path / "input.st"))
        assert f"{m2c.si_to_m2c(al.rho0, 'density'):.6e}" in text
        assert f"{m2c.si_to_m2c(al.c0, 'velocity'):.6e}" in text
        assert f"HugoniotSlope = {al.s:.6f}" in text
        assert f"ReferenceGamma = {al.gamma0:.6f}" in text

    def test_density_upper_limit_stays_below_the_mie_gruneisen_singularity(
            self, tmp_path):
        # Mie-Grueneisen gives c^2 < 0 at rho = rho0 * s/(s-1). A cap above
        # that turns a strong shock into a crash or, worse, nonsense.
        for name in ("al", "fe", "w", "cu"):
            mat = get_material(name)
            text = m2c.write_input(
                _cfg(target=mat, projectile=mat), str(tmp_path / "i.st"))
            cap = float([ln for ln in text.splitlines()
                         if "DensityUpperLimit" in ln][0]
                        .split("=")[1].strip().rstrip(";"))
            singular = m2c.si_to_m2c(mat.rho0, "density") * mat.s / (mat.s - 1)
            assert cap < singular, name

    def test_oblique_incidence_tilts_the_velocity_not_the_target(self,
                                                                tmp_path):
        text = m2c.write_input(_cfg(angle_deg=45.0), str(tmp_path / "i.st"))
        vx = float([ln for ln in text.splitlines()
                    if "VelocityX" in ln and "e+" in ln][0]
                   .split("=")[1].strip().rstrip(";"))
        vy = float([ln for ln in text.splitlines()
                    if "VelocityY" in ln and "e+" in ln][0]
                   .split("=")[1].strip().rstrip(";"))
        assert vx == pytest.approx(vy, rel=1e-6)        # 45 degrees
        assert np.hypot(vx, vy) == pytest.approx(m2c.si_to_m2c(1e4,
                                                               "velocity"))

    def test_normal_incidence_has_no_transverse_velocity(self, tmp_path):
        text = m2c.write_input(_cfg(), str(tmp_path / "i.st"))
        proj = (text.split("under Sphere[0]")[1]
                .split("under CylinderAndCone")[0])
        vy = float(proj.split("VelocityY = ")[1].split(";")[0])
        vx = float(proj.split("VelocityX = ")[1].split(";")[0])
        assert vy == 0.0
        assert vx == pytest.approx(m2c.si_to_m2c(1e4, "velocity"))

    def test_axisymmetric_mesh_starts_at_the_axis(self, tmp_path):
        text = m2c.write_input(_cfg(), str(tmp_path / "i.st"))
        mesh = text.split("under Mesh")[1].split("under Equations")[0]
        assert "Type = Cylindrical;" in mesh
        assert "Y0   = 0.0;" in mesh
        assert "BoundaryConditionY0   = Symmetry;" in mesh
        assert "NumberOfCellsZ = 1;" in mesh

    def test_three_d_mesh_is_cartesian_and_symmetric_about_the_origin(
            self, tmp_path):
        text = m2c.write_input(_cfg(angle_deg=30.0), str(tmp_path / "i.st"))
        mesh = text.split("under Mesh")[1].split("under Equations")[0]
        assert "Type = ThreeDimensional;" in mesh
        assert "NumberOfCellsZ" not in mesh
        assert "ControlPointZ[0]" in mesh

    def test_control_points_are_strictly_increasing(self, tmp_path):
        for cfg in (_cfg(), _cfg(angle_deg=45.0)):
            text = m2c.write_input(cfg, str(tmp_path / "i.st"))
            for axis in ("X", "Y", "Z"):
                coords = [float(ln.split("Coordinate = ")[1].split(";")[0])
                          for ln in text.splitlines()
                          if f"ControlPoint{axis}[" in ln]
                assert coords == sorted(coords), (axis, coords)
                assert len(set(coords)) == len(coords), (axis, coords)

    def test_ionisation_off_removes_the_block_and_the_outputs(self, tmp_path):
        text = m2c.write_input(_cfg(ionisation="off"), str(tmp_path / "i.st"))
        assert "under Ionization" not in text
        assert "MeanCharge = Off;" in text
        assert "ElectronDensity = Off;" in text

    def test_ionisation_on_requests_the_two_fields_stage_three_consumes(
            self, tmp_path):
        text = m2c.write_input(_cfg(), str(tmp_path / "i.st"))
        assert "MeanCharge = On;" in text
        assert "ElectronDensity = On;" in text

    def test_non_ideal_saha_carries_the_continuum_lowering_model(self,
                                                                tmp_path):
        text = m2c.write_input(_cfg(depression_model="Griem"),
                               str(tmp_path / "i.st"))
        assert "Type = NonIdealSahaEquation;" in text
        assert "DepressionModel = Griem;" in text

    def test_ideal_saha_has_no_depression_model(self, tmp_path):
        text = m2c.write_input(_cfg(ionisation="ideal"),
                               str(tmp_path / "i.st"))
        assert "Type = IdealSahaEquation;" in text
        assert "DepressionModel" not in text

    def test_charge_ladder_matches_the_material_yaml(self, tmp_path):
        # M2C must solve the same ionisation ladder hvi_emp does, or the two
        # are not comparable and the bridge is pointless.
        fe = get_material("fe")
        text = m2c.write_input(_cfg(projectile=fe, target=fe),
                               str(tmp_path / "i.st"))
        assert f"MaxChargeNumber = {len(fe.E_ion)};" in text
        assert f"AtomicNumber = {fe.Z};" in text

    def test_ambient_gas_gets_its_own_ionisation_entry(self, tmp_path):
        # Islam et al. (2023): the chamber gas ionises in its own right.
        text = m2c.write_input(_cfg(ambient="ar", ambient_pressure=10.0),
                               str(tmp_path / "i.st"))
        assert "AtomicNumber = 18;" in text
        assert "under Material[0] { // argon" in text

    def test_vacuum_ambient_is_not_given_an_ionisation_entry(self, tmp_path):
        text = m2c.write_input(_cfg(), str(tmp_path / "i.st"))
        ion = (text.split("under Ionization")[1]
               .split("under InitialCondition")[0])
        assert "under Material[0]" not in ion
        assert "under Material[1]" in ion

    def test_ambient_gas_is_flagged_as_a_physics_difference(self):
        notes = " ".join(_cfg(ambient="ar").notes)
        assert "Islam" in notes and "hvi_emp" in notes

    def test_material_ids_follow_the_shipped_deck_convention(self, tmp_path):
        text = m2c.write_input(_cfg(projectile=get_material("fe"),
                                    target=get_material("al")),
                               str(tmp_path / "i.st"))
        eqs = text.split("under Equations")[1].split("under Ionization")[0]
        assert "under Material[1] { // Al" in eqs      # target
        assert "under Material[2] { // Fe" in eqs      # projectile
        assert "MaterialID = 2;" in text               # projectile state
        assert "MaterialID = 1;" in text               # target state

    def test_probes_are_placed_in_projectile_radii(self, tmp_path):
        c = _cfg(diameter=4.0e-3)                      # R = 2 mm
        text = m2c.write_input(c, str(tmp_path / "i.st"))
        assert "under Node[0]" in text
        assert "X = -4.000000" in text                 # -2 R

    def test_rod_and_sphere_produce_different_geometry_blocks(self, tmp_path):
        s = m2c.write_input(_cfg(), str(tmp_path / "a.st"))
        r = m2c.write_input(_cfg(projectile_shape="rod"),
                            str(tmp_path / "b.st"))
        assert "under Sphere[0]" in s and "CylinderWithSphericalCaps" not in s
        assert "under CylinderWithSphericalCaps[0]" in r
        assert "under Sphere[0]" not in r

    def test_projectile_starts_clear_of_the_target_surface(self, tmp_path):
        text = m2c.write_input(_cfg(), str(tmp_path / "i.st"))
        cx = float(text.split("Center_x = ")[1].split(";")[0])
        R = m2c.si_to_m2c(_cfg().radius, "length")
        assert cx + R < 0.0, "projectile overlaps the target at t = 0"

    def test_every_emitted_keyword_is_on_the_verified_list(self, tmp_path):
        """A keyword in the deck but not in VERIFIED_KEYWORDS is one nobody
        has traced to M2C's parser, and one `check_grammar` will not watch.
        Both failures are silent at run time, so they are caught here."""
        import re
        emitted = set()
        for cfg in (_cfg(ambient="ar"),
                    _cfg(angle_deg=30.0, projectile_shape="rod"),
                    _cfg(ionisation="ideal")):
            text = m2c.write_input(cfg, str(tmp_path / "i.st"))
            code = "\n".join(ln.split("//")[0] for ln in text.splitlines())
            # left-hand sides and block names only: values such as
            # `Cylindrical`, `On` and `Ebeling` are not input keywords.
            emitted |= set(re.findall(r"([A-Za-z][A-Za-z0-9_]*)\s*=", code))
            emitted |= set(re.findall(r"under\s+([A-Za-z][A-Za-z0-9_]*)",
                                      code))
        known = set(m2c.VERIFIED_KEYWORDS) | set(m2c.INFERRED_KEYWORDS)
        assert not (emitted - known), sorted(emitted - known)

    def test_the_verified_list_has_no_dead_entries(self, tmp_path):
        """The converse: a keyword nobody emits is one nobody has thought
        about, and it makes `check_grammar` fail on a checkout for no
        reason."""
        import re
        emitted = set()
        for cfg in (_cfg(ambient="ar"),
                    _cfg(angle_deg=30.0, projectile_shape="rod"),
                    _cfg(ionisation="ideal")):
            text = m2c.write_input(cfg, str(tmp_path / "i.st"))
            emitted |= set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", text))
        assert not (set(m2c.VERIFIED_KEYWORDS) - emitted), \
            sorted(set(m2c.VERIFIED_KEYWORDS) - emitted)


# ---------------------------------------------------------------------------
# Problem directory
# ---------------------------------------------------------------------------

class TestProblemDirectory:

    def test_writes_input_and_readme(self, tmp_path):
        res = m2c.write_problem_directory(_cfg(), str(tmp_path / "run"))
        assert os.path.isfile(res["paths"]["input"])
        assert os.path.isfile(res["paths"]["readme"])

    def test_readme_states_the_estimate_is_an_estimate(self, tmp_path):
        res = m2c.write_problem_directory(_cfg(), str(tmp_path / "run"))
        text = open(res["paths"]["readme"]).read()
        assert "ORDER OF MAGNITUDE ONLY" in text
        assert "not measured" in text

    def test_readme_gives_the_gpl_install_route(self, tmp_path):
        res = m2c.write_problem_directory(_cfg(), str(tmp_path / "run"))
        text = open(res["paths"]["readme"]).read()
        assert "git clone https://github.com/kevinwgy/m2c.git" in text
        assert "PETSc" in text          # the non-obvious dependency

    def test_readme_says_m2c_stops_before_the_emp(self, tmp_path):
        # The one thing a reader must not conclude is that M2C replaces
        # Stage 3.
        res = m2c.write_problem_directory(_cfg(), str(tmp_path / "run"))
        text = open(res["paths"]["readme"]).read()
        assert "stops at the plasma state" in text
        assert "emp.py" in text

    def test_notes_reach_the_readme(self, tmp_path):
        res = m2c.write_problem_directory(_cfg(angle_deg=45.0),
                                          str(tmp_path / "run"))
        assert "3-D Cartesian" in open(res["paths"]["readme"]).read()

    def test_command_uses_the_requested_core_count(self):
        assert "-np 8" in m2c.build_command(8)


# ---------------------------------------------------------------------------
# Reading output back
# ---------------------------------------------------------------------------

PROBE_FILE = """\
# time  node0  node1  node2
0.0        1.0e10   2.0e10   0.0
1.0e-7     5.0e12   9.0e12   1.0e11
2.0e-7     3.0e12   4.0e12   2.0e11
"""


class TestReadBack:

    def test_probe_header_is_parsed_not_assumed(self, tmp_path):
        p = tmp_path / "ionization_probes.txt"
        p.write_text(PROBE_FILE)
        pr = m2c.read_probes(str(p))
        assert pr["columns"][0] == "time"
        assert pr["t"].shape == (3,)
        assert pr["values"].shape == (3, 3)

    def test_electron_density_is_converted_to_si(self, tmp_path):
        p = tmp_path / "ionization_probes.txt"
        p.write_text(PROBE_FILE)
        pr = m2c.read_ionization_probes(str(p))
        # 9.0e12 per mm^3 is 9.0e21 per m^3
        assert pr["n_e"].max() == pytest.approx(9.0e21)

    def test_plume_state_finds_the_peak_and_its_time(self, tmp_path):
        p = tmp_path / "ionization_probes.txt"
        p.write_text(PROBE_FILE)
        st = m2c.to_plume_state(m2c.read_ionization_probes(str(p)),
                                get_material("al"))
        assert st["n_e_peak_m3"] == pytest.approx(9.0e21)
        assert st["t_peak_s"] == pytest.approx(1.0e-7)
        assert st["source"] == "M2C"

    def test_ragged_rows_do_not_crash_the_reader(self, tmp_path):
        p = tmp_path / "probes.txt"
        p.write_text("# t a b\n0.0 1.0 2.0\n1.0 3.0\n")
        pr = m2c.read_probes(str(p))
        # truncated to the common width
        assert pr["values"].shape[1] == 1

    def test_empty_file_returns_empty_arrays_rather_than_raising(self,
                                                                 tmp_path):
        p = tmp_path / "probes.txt"
        p.write_text("# t a b\n")
        assert m2c.read_probes(str(p))["t"].size == 0

    def test_empty_probes_give_an_actionable_error(self, tmp_path):
        p = tmp_path / "probes.txt"
        p.write_text("# t a\n")
        with pytest.raises(ValueError) as e:
            m2c.to_plume_state(m2c.read_ionization_probes(str(p)),
                               get_material("al"))
        assert "ElectronDensity = On" in str(e.value)

    def test_missing_file_is_a_file_error(self):
        with pytest.raises(FileNotFoundError):
            m2c.read_probes("/nonexistent/probes.txt")


# ---------------------------------------------------------------------------
# Grammar checking and discovery
# ---------------------------------------------------------------------------

class TestGrammarCheck:

    def test_a_tree_containing_every_keyword_passes(self, tmp_path):
        (tmp_path / "IoData.h").write_text(
            "\n".join(m2c.VERIFIED_KEYWORDS + m2c.INFERRED_KEYWORDS))
        res = m2c.check_grammar(str(tmp_path))
        assert res["missing"] == []
        assert res["inferred_missing"] == []

    def test_a_missing_keyword_is_reported_not_swallowed(self, tmp_path):
        keep = [k for k in m2c.VERIFIED_KEYWORDS if k != "HugoniotSlope"]
        (tmp_path / "IoData.h").write_text("\n".join(keep))
        res = m2c.check_grammar(str(tmp_path))
        assert "HugoniotSlope" in res["missing"]

    def test_a_tree_that_is_not_m2c_says_so(self, tmp_path):
        (tmp_path / "readme.md").write_text("not m2c")
        with pytest.raises(FileNotFoundError) as e:
            m2c.check_grammar(str(tmp_path))
        assert "IoData.h" in str(e.value)

    def test_build_and_atomic_data_directories_are_skipped(self, tmp_path):
        # A stale build/ copy must not be what validates the grammar.
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "IoData.h").write_text("Sphere Radius")
        with pytest.raises(FileNotFoundError):
            m2c.check_grammar(str(tmp_path))

    def test_missing_directory_is_an_error(self):
        with pytest.raises(FileNotFoundError):
            m2c.check_grammar("/nonexistent/m2c")


class TestFindM2C:

    def test_env_var_is_authoritative_even_when_wrong(self, tmp_path,
                                                      monkeypatch):
        # Silently finding a different copy when M2C_HOME is wrong would
        # produce a run against a build the user did not choose.
        monkeypatch.setenv("M2C_HOME", str(tmp_path / "nope"))
        assert m2c.find_m2c() is None

    def test_source_only_checkout_is_reported_as_such(self, tmp_path,
                                                      monkeypatch):
        (tmp_path / "IoData.h").write_text("")
        monkeypatch.setenv("M2C_HOME", str(tmp_path))
        got = m2c.find_m2c()
        assert got is not None and "not compiled" in got

    def test_built_tree_is_reported_as_built(self, tmp_path, monkeypatch):
        exe = tmp_path / "m2c"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        monkeypatch.setenv("M2C_HOME", str(tmp_path))
        assert "built" in m2c.find_m2c()


class TestDirectExecutionGuard:
    """Running the file as a script must not silently half-work."""

    def test_running_the_file_directly_explains_itself(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "hvi_emp", "solvers", "m2c_stage1.py")
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, cwd=here)
        assert r.returncode != 0
        assert "python -m hvi_emp.solvers.m2c_stage1" in r.stderr

    def test_module_cli_reports_a_missing_checkout(self, tmp_path):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, M2C_HOME="")
        r = subprocess.run(
            [sys.executable, "-m", "hvi_emp.solvers.m2c_stage1", "--check",
             str(tmp_path)],
            capture_output=True, text=True, cwd=here, env=env)
        assert r.returncode == 2
        assert "M2C checkout" in r.stdout

    def test_module_cli_passes_on_a_tree_with_the_full_vocabulary(
            self, tmp_path):
        (tmp_path / "IoData.h").write_text("\n".join(m2c.VERIFIED_KEYWORDS))
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run(
            [sys.executable, "-m", "hvi_emp.solvers.m2c_stage1", "--check",
             str(tmp_path)],
            capture_output=True, text=True, cwd=here)
        assert r.returncode == 0, r.stdout
        assert "all present" in r.stdout


# ===========================================================================
# Locating a built M2C (reported: compiled tree reported as "not compiled")
# ===========================================================================

@pytest.fixture
def m2c_tree(tmp_path):
    """The normal layout: source root with a build/ subdirectory."""
    root = tmp_path / "src" / "m2c"
    (root / "build").mkdir(parents=True)
    (root / "Main.cpp").write_text("int main(){}\n")
    (root / "IoData.h").write_text("\n")
    exe = root / "build" / "m2c"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    return root


def test_built_tree_wins_over_source_root(m2c_tree, monkeypatch):
    """A compiled M2C must not be reported as "not compiled".

    The search list has the source root before its own build/ directory,
    and returning the first hit meant the *normal* layout always reported
    source-only. That is the common case, not an edge case.
    """
    from hvi_emp.solvers.m2c_stage1 import find_m2c

    monkeypatch.delenv("M2C_HOME", raising=False)
    got = find_m2c(roots=[str(m2c_tree), str(m2c_tree / "build")])
    assert "built" in got
    assert "not compiled" not in got
    assert str(m2c_tree / "build") in got


def test_source_only_still_reported_when_nothing_is_built(tmp_path,
                                                          monkeypatch):
    from hvi_emp.solvers.m2c_stage1 import find_m2c

    root = tmp_path / "m2c"
    (root / "build").mkdir(parents=True)
    (root / "Main.cpp").write_text("int main(){}\n")
    monkeypatch.delenv("M2C_HOME", raising=False)
    got = find_m2c(roots=[str(root), str(root / "build")])
    assert "source only, not compiled" in got


def test_m2c_home_pointing_at_the_executable_is_explained(m2c_tree,
                                                          monkeypatch):
    """Pointing M2C_HOME at the binary is a common slip.

    Saying "not found" sends the user hunting for a build problem they do
    not have; name the mistake and the right value instead.
    """
    from hvi_emp.solvers.m2c_stage1 import find_m2c

    exe = m2c_tree / "build" / "m2c"
    monkeypatch.setenv("M2C_HOME", str(exe))
    got = find_m2c()
    assert got is not None
    assert "is the executable" in got
    assert str(m2c_tree / "build") in got


def test_m2c_home_directory_is_authoritative(m2c_tree, monkeypatch):
    from hvi_emp.solvers.m2c_stage1 import find_m2c

    monkeypatch.setenv("M2C_HOME", str(m2c_tree / "build"))
    assert "built" in find_m2c()
    monkeypatch.setenv("M2C_HOME", str(m2c_tree))
    assert "source only" in find_m2c()


def test_nonexistent_m2c_home_is_not_found(tmp_path, monkeypatch):
    from hvi_emp.solvers.m2c_stage1 import find_m2c

    monkeypatch.setenv("M2C_HOME", str(tmp_path / "nope"))
    assert find_m2c() is None


# ===========================================================================
# AtomicData generation (M2C ships none -- reported as a startup abort)
# ===========================================================================

def test_write_problem_directory_emits_atomic_data(tmp_path):
    """The deck references AtomicData/; it must exist or the run aborts.

    Reported: `Cannot open ionization energy file AtomicData/I_13.txt`.
    The docs said to symlink M2C's shipped directory -- the repository has
    no such directory, so that advice could not be followed.
    """
    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import M2CConfig, write_problem_directory

    cfg = M2CConfig(projectile=get_material("Fe"), target=get_material("Al"),
                    diameter=6.24e-6, velocity=5e4)
    write_problem_directory(cfg, str(tmp_path), cores=8)

    ad = tmp_path / "AtomicData"
    assert ad.is_dir()
    # Al is Z=13 -- the exact file that was missing.
    assert (ad / "I_13.txt").is_file()
    assert (ad / "E_13_0.txt").is_file()
    assert (ad / "g_13_0.txt").is_file()
    # and the projectile's element too
    assert (ad / "I_26.txt").is_file()


def test_atomic_data_files_contain_no_comments(tmp_path):
    """M2C's reader turns a comment into 1000 zeros, silently.

    GetDataInFile does `file >> double` and breaks only on eof. A '#' sets
    failbit, not eofbit, so the loop never terminates early and appends the
    failed value (0 since C++11) up to MaxCount. A commented file therefore
    loads as zero ionisation energies and the run *completes*, with
    nonsense -- far worse than an error. Verified against the real reader.
    """
    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import write_atomic_data

    write_atomic_data(str(tmp_path), [get_material("Al")])
    for p in tmp_path.glob("[IEg]_*.txt"):
        text = p.read_text()
        assert "#" not in text, f"{p.name} has a comment M2C cannot skip"
        # every token must parse as a float
        for tok in text.split():
            float(tok)


def test_atomic_data_energies_are_in_m2c_units(tmp_path):
    """1 J = 1e9 in mm-g-s-K-A, consistent with the emitted k_B."""
    from hvi_emp import get_material
    from hvi_emp.constants import EV
    from hvi_emp.solvers.m2c_stage1 import (M2C_CONSTANTS, si_to_m2c,
                                            write_atomic_data)

    al = get_material("Al")
    write_atomic_data(str(tmp_path), [al])
    vals = [float(x) for x in
            (tmp_path / "I_13.txt").read_text().split()]

    assert len(vals) == len(al.E_ion)
    # First ionisation of Al is 5.9858 eV.
    assert vals[0] == pytest.approx(5.9858 * EV * 1e9, rel=1e-4)
    assert vals[0] == pytest.approx(si_to_m2c(al.E_ion[0], "energy"))
    # The same 1e9 that scales k_B must scale these.
    assert (M2C_CONSTANTS["BoltzmannConstant"] / 1.380649e-23
            == pytest.approx(1e9))
    # Monotonic: each stage costs more than the last.
    assert vals == sorted(vals)


def test_ground_state_partition_function_is_the_degeneracy(tmp_path):
    """E=0 for one level makes U_r = g_r exactly, matching hvi_emp."""
    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import write_atomic_data

    al = get_material("Al")
    write_atomic_data(str(tmp_path), [al])
    for r in range(len(al.E_ion)):
        e = [float(x) for x in (tmp_path / f"E_13_{r}.txt").read_text().split()]
        g = [float(x) for x in (tmp_path / f"g_13_{r}.txt").read_text().split()]
        assert e == [0.0]
        assert g == [pytest.approx(al.g_ion[r])]


def test_rmax_cannot_collapse_to_zero(tmp_path):
    """All three file sets must be present for every charge state.

    M2C sets rmax = min(len(I), len(E), len(g)). Supplying I alone gives
    rmax = 0, which switches ionisation off with only a warning -- the
    quiet failure this generator exists to prevent.
    """
    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import write_atomic_data

    al = get_material("Al")
    write_atomic_data(str(tmp_path), [al])
    n_I = len((tmp_path / "I_13.txt").read_text().split())
    n_E = len(list(tmp_path.glob("E_13_*.txt")))
    n_g = len(list(tmp_path.glob("g_13_*.txt")))
    assert n_I == n_E == n_g >= 1


def test_duplicate_elements_written_once(tmp_path):
    """Al on Al must not write I_13.txt twice."""
    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import write_atomic_data

    al = get_material("Al")
    out = write_atomic_data(str(tmp_path), [al, al])
    assert len([p for p in out if p.endswith("I_13.txt")]) == 1


def test_atomic_data_skipped_when_ionisation_is_off(tmp_path):
    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import M2CConfig, write_problem_directory

    cfg = M2CConfig(projectile=get_material("Fe"), target=get_material("Al"),
                    diameter=6.24e-6, velocity=5e4, ionisation="off")
    write_problem_directory(cfg, str(tmp_path), cores=8)
    assert not (tmp_path / "AtomicData").exists()


def test_run_command_guards_against_unset_m2c_home(tmp_path):
    """`$M2C_HOME/m2c` with M2C_HOME unset becomes the literal "/m2c".

    Reported: mpirun then says "unable to launch ... Executable: /m2c
    ... lacked permissions", which reads as a permissions problem on a
    binary that was never there. ${VAR:?msg} makes the shell stop and name
    the unset variable instead.
    """
    import subprocess

    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import M2CConfig, write_problem_directory

    cfg = M2CConfig(projectile=get_material("Fe"), target=get_material("Al"),
                    diameter=6.24e-6, velocity=5e4)
    write_problem_directory(cfg, str(tmp_path), cores=8)
    readme = (tmp_path / "README.txt").read_text()

    line = next(ln.strip() for ln in readme.splitlines() if "mpirun" in ln)
    assert "${M2C_HOME:?" in line, line
    assert "$M2C_HOME/m2c" not in line       # the unguarded form is gone

    env = {k: v for k, v in os.environ.items() if k != "M2C_HOME"}
    r = subprocess.run(["bash", "-c", f"echo {line}"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert "/m2c" not in r.stdout            # never expands to the bare path
    assert "M2C_HOME" in r.stderr            # and says which variable

    env["M2C_HOME"] = "/opt/m2c/build"
    r = subprocess.run(["bash", "-c", f"echo {line}"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert "/opt/m2c/build/m2c input.st" in r.stdout


def test_partition_function_is_on_the_fly_for_ground_state_data():
    """CubicSplineInterpolation aborts on ground-state-only atomic data.

    Reported: AtomicIonizationData.cpp:218 `assert(expmin<expmax)` failed
    on every rank. InitializeInterpolationForCharge takes the first
    NON-ZERO excitation energy as factor = -E[r][i]/kb. write_atomic_data()
    emits a single level at E = 0, so factor stays 0 and
    expmin = expmax = exp(0) = 1, making the assertion 1 < 1.

    On-the-fly is also the correct choice on physics grounds here: with one
    level, U_r = g_r is constant in temperature, so there is nothing for a
    spline to interpolate.
    """
    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import M2CConfig

    cfg = M2CConfig(projectile=get_material("Fe"), target=get_material("Al"),
                    diameter=6.24e-6, velocity=5e4)
    assert cfg.partition_function == "OnTheFly"


def test_emitted_deck_uses_on_the_fly(tmp_path):
    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import M2CConfig, write_problem_directory

    cfg = M2CConfig(projectile=get_material("Fe"), target=get_material("Al"),
                    diameter=6.24e-6, velocity=5e4)
    write_problem_directory(cfg, str(tmp_path), cores=8)
    deck = (tmp_path / "input.st").read_text()
    assert "PartitionFunctionEvaluation = OnTheFly;" in deck
    assert "CubicSplineInterpolation" not in deck


def test_zero_excitation_energy_would_break_the_spline():
    """Reproduce M2C's own arithmetic, so the reason is pinned, not recalled."""
    import math

    E_levels = [0.0]                      # what write_atomic_data emits
    kb, Tmin, Tmax = 1.380649e-14, 100.0, 100000.0

    factor = 0.0
    for e in E_levels:
        factor = -e / kb
        if factor != 0:
            break
    expmin = math.exp(factor / Tmin)
    expmax = math.exp(factor / Tmax)
    assert expmin == expmax == 1.0        # the assert(expmin<expmax) failure

    # A real excitation energy would make it well-posed again.
    factor = -(3.14 * 1.602e-19 * 1e9) / kb
    assert math.exp(factor / Tmin) < math.exp(factor / Tmax)


def test_check_grammar_examines_enum_values(tmp_path):
    """Keys alone gave false confidence: 133/133 found, deck still aborted."""
    from hvi_emp.solvers.m2c_stage1 import (VERIFIED_VALUES, check_grammar)

    # A fake checkout that registers the keys but only the WRONG value.
    (tmp_path / "IoData.h").write_text("\n".join(VERIFIED_KEYWORDS_SAMPLE))
    (tmp_path / "Other.cpp").write_text('ClassToken(ca, "X", 1, '
                                        '"CubicSplineInterpolation", 1);')
    res = check_grammar(str(tmp_path))
    assert "values_missing" in res
    assert "OnTheFly" in res["values_missing"]
    assert "CubicSplineInterpolation" in res["values_found"]
    assert set(VERIFIED_VALUES) == set(res["values_found"]) | set(
        res["values_missing"])


def test_check_grammar_scans_beyond_iodata(tmp_path):
    """The ionisation assigner is not in IoData.cpp, so a narrow scan misses it."""
    from hvi_emp.solvers.m2c_stage1 import check_grammar

    (tmp_path / "IoData.h").write_text("// nothing useful here\n")
    (tmp_path / "SomeOther.cpp").write_text('"OnTheFly"\n')
    res = check_grammar(str(tmp_path))
    assert any(p.endswith("SomeOther.cpp") for p in res["sources"])
    assert "OnTheFly" in res["values_found"]


VERIFIED_KEYWORDS_SAMPLE = ["PartitionFunctionEvaluation", "MaxIts"]


# ===========================================================================
# Cost model, calibrated against a real run (reported from reznor)
# ===========================================================================

#: What M2C printed for the default Fe->Al deck, and what it then cost.
_OBSERVED_CELLS = 994 * 356          # "Total number of nodes/cells: 353864"
_OBSERVED_S_PER_STEP = 13.12         # steady state, steps 2-5, 32 ranks
_OBSERVED_RANKS = 32


def _default_cfg():
    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import M2CConfig
    return M2CConfig(projectile=get_material("Fe"),
                     target=get_material("Al"),
                     diameter=6.237e-6, velocity=5e4)


def test_cell_count_matches_the_graded_mesh_m2c_builds():
    """The estimate assumed a uniform fine mesh and was 16x high.

    M2C reported 994 x 356 = 353,864 cells for the deck that the old
    cell_count() put at 5.76e6.
    """
    cfg = _default_cfg()
    assert cfg.cell_count() == pytest.approx(_OBSERVED_CELLS, rel=0.10)


def test_wall_time_matches_the_measured_run():
    """End-to-end: cells x steps x rate must reproduce 9.4 days.

    Two compensating errors used to hide each other here -- a 590x
    optimistic rate against a 16x pessimistic cell count. Pinning the
    product stops that recurring.
    """
    cfg = _default_cfg()
    est = cfg.estimate_resources(cores=_OBSERVED_RANKS)
    measured_h = _OBSERVED_S_PER_STEP * est["n_steps"] / 3600.0
    assert est["wall_hours_estimate"] == pytest.approx(measured_h, rel=0.15)
    assert est["wall_hours_estimate"] > 24.0      # days, not hours


def test_rate_is_flagged_measured_only_for_the_nonideal_path():
    """Do not let an assumed number masquerade as a measured one."""
    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import M2CConfig

    on = _default_cfg().estimate_resources(cores=32)
    assert on["rate_is_measured"] is True
    assert "Saha" in on["cost_dominated_by"]

    off = M2CConfig(projectile=get_material("Fe"), target=get_material("Al"),
                    diameter=6.237e-6, velocity=5e4, ionisation="off")
    assert off.estimate_resources(cores=32)["rate_is_measured"] is False


def test_ionisation_dominates_the_cost():
    """The Saha solver, not the hydro, is what makes this expensive."""
    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import M2CConfig

    base = dict(projectile=get_material("Fe"), target=get_material("Al"),
                diameter=6.237e-6, velocity=5e4)
    on = M2CConfig(**base).estimate_resources(cores=32)
    off = M2CConfig(**base, ionisation="off").estimate_resources(cores=32)
    assert on["wall_hours_estimate"] > 100 * off["wall_hours_estimate"]


def test_results_directory_is_created(tmp_path):
    """M2C does not mkdir its own output directory.

    Reported: the run initialised everything -- mesh, level sets, both Saha
    solvers, 24 probes -- then died on
    "Cannot open file 'results/density_probes.txt'".
    """
    from hvi_emp.solvers.m2c_stage1 import write_problem_directory

    write_problem_directory(_default_cfg(), str(tmp_path), cores=8)
    assert (tmp_path / "results").is_dir()
    # and the deck must actually be pointing there
    assert 'Prefix = "results/"' in (tmp_path / "input.st").read_text()
