"""Tests for the jetting model.

The reference data is Jean & Rollins, *AIAA J.* **8**, 1742 (1970) -- three
tables from a 1970 ballistic range. Two of the three predictions have no free
parameters at all, so these are real tests rather than curve-fits checking
themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from hvi_emp.jetting import (FAST_JET_FACTOR, V_MIN_SPHERE, critical_angle,
                             downrange_threshold, jet_velocity,
                             jet_velocity_at_angle)
from hvi_emp.materials import get_material

AL = get_material("al")
CU = get_material("cu")
FE = get_material("fe")
W = get_material("w")

# Jean & Rollins Table 1: Cu -> Al, (v km/s, measured alpha_c deg, error deg)
TABLE_1 = [(1.0, 12.5, 0.7), (3.0, 17.0, 2.0), (3.5, 22.0, 3.0),
           (5.5, 28.0, 1.0), (6.3, 32.0, 3.0)]

# Table 2: Cu-Al cones, (collision angle deg, measured Vj/V1, their calc)
TABLE_2 = [(10, 13.0, 11.55), (22, 5.5, 5.91), (25, 4.9, 4.96),
           (27, 4.7, 4.38), (30, 3.2, 3.79), (35, 3.3, 3.17), (40, 3.5, 2.86)]

# Table 3: Al sphere on Al, (impact km/s, luminous ring km/s, fast jet km/s)
TABLE_3 = [(5.30, 15.6, 33.0), (4.94, 17.0, 37.6), (3.89, 14.6, 32.0),
           (6.05, 18.6, 35.0), (6.04, 16.8, 32.4), (5.88, 16.0, 30.6)]


class TestCriticalAngle:
    """tan(alpha_c) = v / U_s, with nothing fitted."""

    def test_mean_error_against_table_1_is_under_two_degrees(self):
        errs = [abs(np.degrees(critical_angle(CU, AL, v * 1e3)) - meas)
                for v, meas, _ in TABLE_1]
        assert np.mean(errs) < 2.0, dict(zip([t[0] for t in TABLE_1], errs))

    def test_most_points_land_inside_the_quoted_experimental_error(self):
        inside = sum(
            abs(np.degrees(critical_angle(CU, AL, v * 1e3)) - meas) <= err
            for v, meas, err in TABLE_1)
        # The 1 km/s point is the known miss: lowest pressure, where the
        # linear Us-Up fit is weakest and material strength is not negligible.
        assert inside >= 3

    def test_the_known_miss_is_the_lowest_velocity_point(self):
        errs = [abs(np.degrees(critical_angle(CU, AL, v * 1e3)) - meas)
                for v, meas, _ in TABLE_1]
        assert int(np.argmax(errs)) == 0

    def test_angle_increases_with_impact_speed(self):
        """U_s = c0 + s*u_p grows more slowly than v, so alpha_c opens up.

        This is the non-obvious direction: faster impacts jet more readily.
        """
        angles = [critical_angle(CU, AL, v) for v in
                  (1e3, 3e3, 6e3, 12e3, 25e3)]
        assert angles == sorted(angles)

    def test_angle_stays_in_the_first_quadrant(self):
        for v in (1e3, 1e4, 1e5):
            assert 0.0 < critical_angle(FE, AL, v) < np.pi / 2

    def test_non_positive_velocity_is_rejected(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError):
                critical_angle(FE, AL, bad)


class TestConeJetVelocity:
    """v_jet = v * cot(alpha/2), the closed form behind Table 2."""

    def test_matches_jean_and_rollins_own_calculation(self):
        for ang, _meas, their_calc in TABLE_2:
            ours = jet_velocity_at_angle(1.0, np.radians(ang))
            assert ours == pytest.approx(their_calc, rel=0.15), ang

    def test_mean_ratio_to_measurement_is_near_unity(self):
        ratios = [jet_velocity_at_angle(1.0, np.radians(a)) / m
                  for a, m, _ in TABLE_2]
        assert 0.85 < np.mean(ratios) < 1.15

    def test_identity_cot_half_angle(self):
        """cot(a) + csc(a) == cot(a/2) -- the simplification in the paper."""
        for ang in np.radians([10.0, 25.0, 40.0, 75.0]):
            longhand = 1.0 / np.tan(ang) + 1.0 / np.sin(ang)
            assert jet_velocity_at_angle(1.0, ang) == pytest.approx(longhand)

    def test_shallower_angles_jet_faster(self):
        speeds = [jet_velocity_at_angle(1.0, np.radians(a))
                  for a in (10, 20, 30, 45, 60)]
        assert speeds == sorted(speeds, reverse=True)

    def test_scales_linearly_with_impact_speed(self):
        a = np.radians(30.0)
        assert (jet_velocity_at_angle(2.0, a)
                == pytest.approx(2.0 * jet_velocity_at_angle(1.0, a)))

    @pytest.mark.parametrize("bad", [0.0, -0.1, np.pi, 4.0])
    def test_angle_outside_the_open_interval_is_rejected(self, bad):
        with pytest.raises(ValueError):
            jet_velocity_at_angle(1.0, bad)


class TestSphereJet:

    def test_steady_jet_overpredicts_the_luminous_ring_by_about_18_percent(
            self):
        """It SHOULD overpredict.

        Jean & Rollins explain that the fastest jet material is too sparse to
        photograph, so their measured ring is a lower bound on the theoretical
        maximum. A model that matched exactly would be suspicious.
        """
        ratios = [
            jet_velocity(AL, AL, v * 1e3, warn=False).v_steady / 1e3 / ring
            for v, ring, _ in TABLE_3]
        assert np.mean(ratios) == pytest.approx(1.18, abs=0.05)
        assert np.std(ratios) < 0.12
        assert all(r > 1.0 for r in ratios)

    def test_fast_jet_factor_is_consistent_across_the_six_shots(self):
        ratios = [
            fast / (jet_velocity(AL, AL, v * 1e3, warn=False).v_steady / 1e3)
            for v, _, fast in TABLE_3]
        assert np.mean(ratios) == pytest.approx(FAST_JET_FACTOR, rel=0.02)
        assert np.std(ratios) < 0.25

    def test_fast_jet_reaches_the_velocities_they_measured(self):
        """30-38 km/s Al jets from 3.9-6.1 km/s impacts."""
        for v, _ring, fast in TABLE_3:
            js = jet_velocity(AL, AL, v * 1e3, warn=False)
            assert js.v_fast / 1e3 == pytest.approx(fast, rel=0.45)

    def test_copper_can_reach_the_fifty_km_per_second_they_report(self):
        """'fast jet velocities as high as 50 km/sec have been measured'.

        Jean & Rollins fired copper into cadmium -- soft and dense -- at up to
        8 km/s. Cadmium is not in the material library, so aluminium stands in
        as the soft target; the point is that the model reaches their band at
        the top of their velocity range, not that it hits 50.0 exactly.
        """
        js = jet_velocity(CU, AL, 8.0e3, warn=False)
        assert 40e3 < js.v_fast < 60e3

    def test_ratio_falls_as_impact_speed_rises(self):
        """The amplification is largest for slow impacts, and bounded above."""
        ratios = [jet_velocity(AL, AL, v, warn=False).fast_ratio
                  for v in (4e3, 8e3, 20e3, 50e3)]
        assert ratios == sorted(ratios, reverse=True)
        assert ratios[-1] > 2.0          # never collapses to unity

    def test_absolute_jet_speed_still_rises_with_impact_speed(self):
        speeds = [jet_velocity(AL, AL, v, warn=False).v_fast
                  for v in (4e3, 8e3, 20e3, 50e3)]
        assert speeds == sorted(speeds)

    def test_jet_always_outruns_the_projectile(self):
        for mat in (AL, CU, FE, W):
            for v in (4e3, 10e3, 30e3):
                js = jet_velocity(mat, AL, v, warn=False)
                assert js.v_steady > v
                assert js.v_fast > js.v_steady

    def test_summary_says_mass_is_not_predicted(self):
        """The single most important caveat must survive refactoring."""
        s = jet_velocity(AL, AL, 6e3, warn=False).summary()
        assert "mass fraction not predicted" in s

    def test_below_the_validated_floor_it_warns(self):
        with pytest.warns(RuntimeWarning, match="vacuous upper bound"):
            jet_velocity(AL, AL, 1.0e3)

    def test_inside_the_validated_range_it_does_not_warn(self):
        import warnings as w
        with w.catch_warnings():
            w.simplefilter("error")
            jet_velocity(AL, AL, 5.0e3)

    def test_the_divergence_the_warning_is_about_is_real(self):
        """As v -> 0 the prediction tends to 2*U_s, not to 0.

        Documented rather than silently clipped, because clipping would hide
        that the formula stops meaning anything down there.
        """
        js = jet_velocity(AL, AL, 200.0, warn=False)
        assert js.v_steady == pytest.approx(2.0 * js.U_s, rel=0.02)
        assert js.v_steady > 10e3          # absurd for a 0.2 km/s impact


class TestDownrangeThreshold:

    def test_jet_vaporises_a_rear_wall_far_below_the_bulk_threshold(self):
        r = downrange_threshold(FE, AL, AL, which="P_vap")
        assert not r["saturated"]
        assert r["v_downrange"] < r["v_bulk"]
        assert r["gain"] > 3.0

    def test_gain_is_similar_across_projectile_materials(self):
        """It is set by jet kinematics, not by what you throw."""
        gains = [downrange_threshold(p, AL, AL, which="P_vap")["gain"]
                 for p in (AL, CU, FE, W)]
        assert max(gains) / min(gains) < 1.3

    def test_saturation_is_reported_not_papered_over(self):
        """Where the jet clears the threshold at every validated speed, the
        function must say so rather than return the search floor."""
        r = downrange_threshold(FE, AL, AL, which="P_boil")
        assert r["saturated"] is True
        assert np.isnan(r["v_downrange"])
        assert np.isnan(r["gain"])
        assert "validity floor" in r["note"]

    def test_search_never_starts_below_the_sphere_validity_floor(self):
        for which in ("P_melt", "P_boil", "P_vap"):
            r = downrange_threshold(FE, AL, AL, which=which)
            if not r["saturated"]:
                assert r["v_downrange"] > V_MIN_SPHERE

    def test_wall_defaults_to_the_target(self):
        a = downrange_threshold(FE, AL, which="P_vap")
        b = downrange_threshold(FE, AL, AL, which="P_vap")
        assert a["v_downrange"] == b["v_downrange"]

    def test_a_denser_wall_is_vaporised_by_a_slower_impact_not_a_faster_one(
            self):
        """The counter-intuitive one, and it is the model being right.

        Tungsten is far harder to vaporise than aluminium in energy terms, so
        the naive expectation is that it needs a faster impact. It does not.
        Shock pressure goes as rho0 * U_s * u_p, and tungsten's density is 7x
        aluminium's, so tungsten reaches its (higher) complete-vaporisation
        pressure of ~2840 GPa at a particle velocity of only 9.4 km/s, where
        aluminium needs 18.5 km/s to reach 1500 GPa. High impedance converts
        jet velocity into pressure efficiently enough to more than pay for the
        higher threshold.

        Recorded as a test because it is exactly the kind of result someone
        will later "fix".
        """
        soft = downrange_threshold(FE, AL, AL, which="P_vap")["v_downrange"]
        hard = downrange_threshold(FE, AL, W, which="P_vap")["v_downrange"]
        assert hard < soft

    def test_the_impedance_reason_for_that_holds_in_the_eos_layer(self):
        """Guards the explanation above, not just the outcome."""
        from hvi_emp.eos import (particle_velocity_from_pressure,
                                 phase_thresholds)
        up_al = particle_velocity_from_pressure(
            AL, phase_thresholds(AL)["P_vap"])
        up_w = particle_velocity_from_pressure(
            W, phase_thresholds(W)["P_vap"])
        assert up_w < up_al
        assert phase_thresholds(W)["P_vap"] > phase_thresholds(AL)["P_vap"]

    def test_unknown_threshold_name_is_rejected(self):
        with pytest.raises(KeyError):
            downrange_threshold(FE, AL, AL, which="P_sublime")


class TestValidationWiring:

    def test_jetting_checks_are_in_the_validation_suite(self):
        from hvi_emp.validation import check_jetting
        checks = check_jetting()
        assert len(checks) >= 5
        assert all("Jean & Rollins" in c.source or "Jean" in c.source
                   for c in checks)

    def test_they_all_pass(self):
        from hvi_emp.validation import check_jetting
        failed = [c.name for c in check_jetting() if not c.passed]
        assert not failed, failed
