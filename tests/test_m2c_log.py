"""Reading an M2C log, and the run that cost 12.4 hours without saying so.

Every number here is from a real failure: a deck that reached
t = 8.414028e-10 s after 129521 steps with dt = 1.885042e-22, then ran on
until a level-set check aborted it.
"""

import math

import pytest

from hvi_emp.solvers.m2c_log import (C_LIGHT_M_S, dt_verdict, eta,
                                     human_time, implied_wave_speed, onset,
                                     parse_log, wave_speed_ratio)

REAL_STEP = ("Step 129521: t = 8.414028e-10, dt = 1.885042e-22, "
             "cfl = 1.0000e-01. Computation time: 4.4740e+04 s.")
REAL_FATAL = ("*** Error: Node (519,18,0) belongs to two material "
              "subdomains. phi[0(matid:1)] = -5.600041e-03, "
              "phi[1(matid:2)] = -2.026143e-02.")


def _log(n=400, collapse_from=None, dt0=6.9e-15, extra=()):
    lines, t, wall = [], 0.0, 0.0
    for i in range(1, n + 1):
        dt = dt0
        if collapse_from and i >= collapse_from:
            dt = dt0 * math.exp(-40.0 * (i - collapse_from) / (n - collapse_from + 1))
        t += dt
        wall += 0.345
        lines.append(f"Step {i}: t = {t:.6e}, dt = {dt:.6e}, "
                     f"cfl = 1.0000e-01. Computation time: {wall:.4e} s.")
    return list(lines) + list(extra)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def test_parses_the_real_step_line():
    """The trailing period after the cfl value used to break float()."""
    log = parse_log([REAL_STEP])
    assert log["steps"] == [129521]
    assert log["t"][0] == pytest.approx(8.414028e-10)
    assert log["dt"][0] == pytest.approx(1.885042e-22)
    assert log["cfl"][0] == pytest.approx(0.1)
    assert log["wall"][0] == pytest.approx(4.4740e4)


def test_counts_riemann_edges_not_just_lines():
    lines = ["Warning: Riemann solver failed to find a bracketing interval "
             "or to converge on 487 edge(s).",
             "Warning: Riemann solver failed to find a bracketing interval "
             "or to converge on 485 edge(s)."]
    log = parse_log(lines)
    assert log["counts"]["riemann_edges"] == 2
    assert log["totals"]["riemann_edges"] == 972


def test_recognises_the_secant_division_by_zero():
    log = parse_log(["Warning: Division-by-zero while using the secant "
                     "method to solve the Riemann problem."])
    assert log["counts"]["riemann_divzero"] == 1


def test_recognises_the_two_subdomain_abort():
    log = parse_log([REAL_FATAL])
    assert log["counts"]["two_subdomains"] == 1
    assert "two material subdomains" in log["fatal"]


def test_recognises_levelset_reinit_failure():
    log = parse_log(["Warning: L-S Reinitialization failed to converge "
                     "after 100 iterations."])
    assert log["counts"]["levelset_reinit"] == 1


def test_empty_log_is_not_a_crash():
    log = parse_log([])
    assert log["n_steps"] == 0
    assert dt_verdict(log)["state"] == "unknown"


# --------------------------------------------------------------------------
# the physics check
# --------------------------------------------------------------------------

def test_the_real_dt_implies_a_superluminal_wave_speed():
    """The claim the whole diagnosis rests on, with the real numbers.

    dt is not a free parameter -- the solver set it from the CFL
    condition -- so inverting it gives the wave speed the EOS must have
    returned. dx here is the deck's cell size, ~0.097 um.
    """
    v = implied_wave_speed(1.885042e-22, 0.1, 9.746e-8)
    assert v == pytest.approx(5.17e13, rel=0.02)
    assert v / C_LIGHT_M_S > 1e5


def test_absolute_speed_needs_the_finest_cell_not_the_nominal_one():
    """Why the absolute form is the weaker argument.

    The healthy dt with the NOMINAL spacing implies 1.4e6 m/s -- already
    100x any material sound speed, which cannot be right for a run that
    was behaving. The CFL condition uses the finest cell of a graded mesh,
    and at ~2 nm the same dt gives a sane ~30 km/s. So the absolute number
    is only as good as the dx handed to it.
    """
    nominal = implied_wave_speed(6.9e-15, 0.1, 9.746e-8)
    finest = implied_wave_speed(6.9e-15, 0.1, 2.07e-9)
    assert nominal > 1e6                    # implausible: wrong dx
    assert 1e4 < finest < 1e5               # plausible: shocked aluminium
    assert nominal / finest == pytest.approx(9.746e-8 / 2.07e-9, rel=1e-6)


def test_wave_speed_guards_against_nonsense_input():
    assert implied_wave_speed(0.0, 0.1, 1e-6) is None
    assert implied_wave_speed(1e-15, 0.1, 0.0) is None


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------

def test_healthy_run_is_not_cried_wolf_over():
    """A watchdog that always fires is a watchdog nobody reads."""
    v = dt_verdict(parse_log(_log(n=400)))
    assert v["state"] == "healthy"


def test_collapse_is_caught():
    v = dt_verdict(parse_log(_log(n=400, collapse_from=250)))
    assert v["state"] == "collapsed"
    assert v["ratio"] > 1e3


def test_a_long_trailing_median_must_not_hide_the_collapse():
    """THE bug in the first version of this watchdog.

    With a 200-entry trailing median over a sparsely sampled log, the
    verdict was 'dt is within 1.04x of the start' on a run whose final dt
    was 2.2e-31 s. A false healthy is worse than no watchdog at all.
    """
    # Healthy for most of the log, collapsing only at the very end -- the
    # shape a long median is built to ignore.
    lines = _log(n=300) + _log(n=20, collapse_from=1, dt0=6.9e-15)[-6:]
    v = dt_verdict(parse_log(lines))
    assert v["state"] == "collapsed", v


def test_a_single_small_step_is_not_a_collapse():
    """One spike is noise; a collapse is sustained."""
    lines = _log(n=300)
    lines.insert(150, "Step 150: t = 1.0e-12, dt = 1.0e-30, "
                      "cfl = 1.0000e-01. Computation time: 5.0e+01 s.")
    assert dt_verdict(parse_log(lines))["state"] == "healthy"


# --------------------------------------------------------------------------
# onset and eta
# --------------------------------------------------------------------------

def test_onset_finds_where_it_went_wrong():
    """Which hours were wasted, and which output is still usable."""
    log = parse_log(_log(n=400, collapse_from=250))
    o = onset(log)
    assert o is not None
    assert 245 <= o["step"] <= 275
    assert o["wall"] < log["wall"][-1]


def test_onset_is_none_for_a_healthy_run():
    assert onset(parse_log(_log(n=400))) is None


def test_eta_extrapolates_the_current_rate_however_absurd():
    """An absurd answer IS the answer: it says stop waiting."""
    log = parse_log([REAL_STEP])
    log["steps"].append(129522)
    log["t"].append(8.414028e-10)
    log["dt"].append(1.885042e-22)
    log["cfl"].append(0.1)
    log["wall"].append(4.4741e4)
    e = eta(log, t_end=9.22e-9)
    assert e["ok"] and not e["done"]
    # (9.22e-9 - 8.414e-10) / 1.885e-22
    assert e["steps_needed"] == pytest.approx(4.44e13, rel=0.02)


def test_eta_knows_when_it_is_already_done():
    log = parse_log(_log(n=50))
    assert eta(log, t_end=1e-30)["done"] is True


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def test_human_time_switches_at_the_natural_boundary():
    """44740 s is 12.4 h. Reporting '746 min' makes a lost day read small."""
    assert human_time(4.4740e4) == "12.4 h"
    assert human_time(30) == "30 s"
    assert human_time(600) == "10 min"
    assert human_time(3 * 86400) == "3 days"
    assert "years" in human_time(1e14)


def test_human_time_handles_none():
    assert human_time(None) == "unknown"


# --------------------------------------------------------------------------
# the dx-free argument
# --------------------------------------------------------------------------

def test_wave_speed_ratio_needs_no_cell_size():
    """dx cancels, so the conclusion cannot be wrong about the mesh."""
    log = parse_log(_log(n=400, collapse_from=250))
    r = wave_speed_ratio(log)
    assert r["ratio"] == pytest.approx(r["dt_early"] / r["dt_now"])
    # Superluminal for every plausible starting speed, not just the
    # convenient one.
    assert min(r["implied"].values()) > C_LIGHT_M_S


def test_wave_speed_ratio_is_quiet_on_a_healthy_run():
    r = wave_speed_ratio(parse_log(_log(n=400)))
    assert r["ratio"] < 10
    assert max(r["implied"].values()) < C_LIGHT_M_S


def test_the_real_dt_ratio():
    """3.66e7, from the two numbers in Hasan's log and nothing else."""
    assert 6.9e-15 / 1.885042e-22 == pytest.approx(3.66e7, rel=0.02)
