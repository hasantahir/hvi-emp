"""Read an M2C log and decide whether the run is still worth waiting for.

An M2C run does not usually fail by stopping. It fails by taking smaller
and smaller steps until the time step is physically meaningless, and then
running on for hours in that state before something downstream finally
aborts. The log keeps printing `Step N: ...` throughout, so from the outside
it looks like progress.

The number that gives it away is `dt`. A run that reached t = 8.4e-10 s in
129521 steps has an average step of 6.5e-15 s; when the log says
`dt = 1.885042e-22`, the step has fallen **seven orders of magnitude** below
its own average. Nothing recovers from that.

What makes it unambiguous rather than merely suspicious is that dt is not a
free parameter. The solver sets it from the CFL condition,

    dt = CFL * dx / (|u| + c)

so a dt implies a wave speed. The useful form of that is a RATIO, because
with the same mesh and the same CFL the cell size cancels:

    (|u| + c)_now / (|u| + c)_early = dt_early / dt_now

For that run the ratio is 3.7e7. Even granting the early state a generous
100 km/s, the final one is over 1000 times the speed of light. No assumption
about dx is needed, which matters: these decks use a graded mesh, the CFL
condition sees the finest cell, and quoting an absolute speed from the
nominal spacing over-states it by the grading factor -- roughly 50x here,
and in the alarming direction.

It is not a small time step. It is a broken thermodynamic state: the
equation of state has returned a sound speed it cannot have, almost always
because it is being evaluated outside the density range it was fitted for.

So this module turns a log into:

  * `parse_log`        -- the step history and the warnings, as data
  * `dt_verdict`       -- healthy / stalling / collapsed, with the numbers
  * `wave_speed_ratio` -- the dx-free version of the argument above
  * `onset`            -- which step it went wrong at, so the wasted hours
                          and the still-usable output can be told apart
  * `eta`              -- how long the run needs at its CURRENT rate, which
                          is what "is it worth waiting for" reduces to

It reads a finished log for a post-mortem, or a growing one to decide
whether to kill a job that is quietly wasting a queue slot.
"""

from __future__ import annotations

import re

__all__ = ["parse_log", "dt_verdict", "eta", "onset", "human_time",
           "implied_wave_speed", "wave_speed_ratio",
           "WARNING_PATTERNS", "C_LIGHT_M_S"]

C_LIGHT_M_S = 2.99792458e8

#: A float, not "any run of digits and symbols". A character class like
#: `[-\d.eE+]+` is greedy enough to swallow the sentence-ending period in
#: `cfl = 1.0000e-01.`, and then float() raises on the whole line.
_F = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"

#: `Step 129521: t = 8.414028e-10, dt = 1.885042e-22, cfl = 1.0000e-01.`
_STEP = re.compile(
    rf"Step\s+(\d+)\s*:\s*t\s*=\s*{_F}\s*,\s*"
    rf"dt\s*=\s*{_F}\s*,\s*cfl\s*=\s*{_F}")
_WALL = re.compile(rf"Computation time\s*:\s*{_F}\s*s")

#: Each is (key, regex, what it means). Order is the order they are reported.
#: These are M2C's own strings -- if a future version rewords them the
#: watchdog goes quiet rather than wrong, which is why dt is the primary
#: signal and these are corroboration.
WARNING_PATTERNS = (
    ("riemann_edges",
     re.compile(r"Riemann solver failed .*? on (\d+) edge\(s\)"),
     "Riemann solver could not bracket or converge"),
    ("riemann_divzero",
     re.compile(r"Division-by-zero while using the secant method"),
     "secant method had f0 == f1: the pressure function is flat, so the "
     "states on either side are not physically distinguishable"),
    ("levelset_reinit",
     re.compile(r"(?:L-S|Level ?[Ss]et).{0,40}[Rr]einitializ\w*"
                r".{0,40}(?:fail|not converge|unsuccessful)"),
     "level-set reinitialisation did not converge: the interface is no "
     "longer a signed distance function and the fronts can drift"),
    ("two_subdomains",
     re.compile(r"belongs to two material subdomains"),
     "a node is inside two materials at once: the level sets have "
     "overlapped, which is fatal and is usually a consequence, not a cause"),
    ("negative_density",
     re.compile(r"[Nn]egative (?:density|pressure|internal energy)"),
     "a state variable went negative: the EOS is outside its valid range"),
)


def parse_log(lines) -> dict:
    """Step history and warning counts from an M2C log.

    `lines` may be any iterable of strings, so this works on a file, on a
    growing tail, or on a list in a test.
    """
    steps, times, dts, cfls, walls = [], [], [], [], []
    counts = {k: 0 for k, _r, _w in WARNING_PATTERNS}
    totals = {k: 0 for k, _r, _w in WARNING_PATTERNS}
    first_seen = {}
    fatal = None

    for line in lines:
        m = _STEP.search(line)
        if m:
            steps.append(int(m.group(1)))
            times.append(float(m.group(2)))
            dts.append(float(m.group(3)))
            cfls.append(float(m.group(4)))
            w = _WALL.search(line)
            walls.append(float(w.group(1)) if w else None)
            continue
        for key, rx, _why in WARNING_PATTERNS:
            m = rx.search(line)
            if not m:
                continue
            counts[key] += 1
            if m.groups():
                try:
                    totals[key] += int(m.group(1))
                except (ValueError, IndexError):
                    pass
            first_seen.setdefault(key, (steps[-1] if steps else None,
                                        line.strip()[:160]))
        if "*** Error" in line or "Error:" in line:
            fatal = fatal or line.strip()[:200]

    return {"steps": steps, "t": times, "dt": dts, "cfl": cfls,
            "wall": walls, "counts": counts, "totals": totals,
            "first_seen": first_seen, "fatal": fatal,
            "n_steps": len(steps)}


def implied_wave_speed(dt: float, cfl: float, dx: float):
    """|u| + c implied by the solver's own CFL choice, in dx's units per s.

    dt is not free: the solver picked it. Inverting the CFL condition turns
    an opaque "the time step got small" into a speed.

    Be careful with `dx` on a graded mesh. The CFL condition uses the
    SMALLEST cell, which in these decks is in the refined zone around the
    impact point and can be a factor of 50 below the nominal cell size.
    Pass the nominal spacing and this over-estimates the speed by exactly
    that factor. `wave_speed_ratio` below needs no dx at all and is the
    number to argue from.
    """
    if dt <= 0 or dx <= 0 or cfl <= 0:
        return None
    return cfl * dx / dt


def wave_speed_ratio(log: dict, window: int = 200) -> dict | None:
    """How much faster the solver thinks waves are now than when it was well.

    This is the honest form of the superluminal argument, because **dx
    cancels**. From dt = CFL*dx/(|u|+c) with the same mesh and the same CFL,

        (|u|+c)_now / (|u|+c)_early = dt_early / dt_now

    so no assumption about cell size is needed -- which matters, because
    the decks use a graded mesh and the nominal spacing is not the spacing
    the CFL condition sees. On the 12.4-hour run this ratio is 3.7e7: even
    if the early state was a generous 100 km/s, the final one is over 1000
    times the speed of light. An equation of state cannot return that, so
    the state it is being handed is not physical.
    """
    dts = [d for d in log["dt"] if d > 0]
    if len(dts) < 10:
        return None
    early = sorted(dts[:min(window, max(len(dts) // 4, 10))])
    ref = early[len(early) // 2]
    now = dts[-1]
    if now <= 0:
        return None
    ratio = ref / now
    return {
        "ratio": ratio, "dt_early": ref, "dt_now": now,
        # What the final speed would be for a range of plausible early
        # speeds. The conclusion has to survive the whole range to be worth
        # stating, and here it does.
        "implied": {v: v * ratio for v in (1.0e4, 3.0e4, 1.0e5)},
    }


def dt_verdict(log: dict, window: int = 200, collapse: float = 1e3,
               stall: float = 30.0) -> dict:
    """healthy / stalling / collapsed, with the ratio that decided it.

    The reference is the run's OWN early behaviour, not an absolute
    threshold: dt varies by orders of magnitude between a 5 nm MD box and a
    400 mm plume, so a fixed "dt is too small" number would be meaningless
    across scales. What is meaningful is dt falling far below what this run
    was managing when it was well.
    """
    dts = [d for d in log["dt"] if d > 0]
    if len(dts) < 10:
        return {"state": "unknown", "reason": "too few steps to judge",
                "ratio": None}

    early = sorted(dts[:min(window, max(len(dts) // 4, 10))])
    ref = early[len(early) // 2]                 # median, not mean

    # The recent window must be SHORT.
    #
    # A median over a long trailing window is exactly the wrong statistic
    # for a terminal decline: it is designed to ignore the tail, and the
    # tail is the whole signal. The first version of this used the last 200
    # entries and pronounced "dt is within 1.04x of the start" on a log
    # whose final dt was 2.2e-31 s -- a healthy verdict on a dead run,
    # which is worse than no watchdog.
    #
    # A handful of entries is enough to reject a single spike while still
    # following the collapse down, and dt collapse does not recover, so
    # there is no case for smoothing it away.
    tail = dts[-min(max(len(dts) // 20, 5), len(dts)):]
    now = sorted(tail)[len(tail) // 2]
    ratio = ref / now if now > 0 else float("inf")

    # Cross-check against the single last value, so a long sparse log (one
    # line per 1000 steps) cannot hide a collapse inside the window either.
    last = dts[-1]
    ratio_last = ref / last if last > 0 else float("inf")
    ratio = max(ratio, ratio_last) if ratio_last >= collapse else ratio

    if ratio >= collapse:
        state, reason = "collapsed", (
            f"dt has fallen {ratio:.3g}x below this run's own early median "
            f"-- the solver is no longer advancing")
    elif ratio >= stall:
        state, reason = "stalling", (
            f"dt is {ratio:.3g}x below this run's early median and heading "
            f"the wrong way")
    else:
        state, reason = "healthy", f"dt is within {ratio:.3g}x of the start"
    return {"state": state, "reason": reason, "ratio": ratio,
            "dt_early": ref, "dt_now": now, "dt_last": last,
            "ratio_last": ratio_last}


def onset(log: dict, factor: float = 10.0, window: int = 200) -> dict | None:
    """The first step where dt had fallen `factor` below the early median.

    This is the question worth answering after a 12-hour run dies: not
    "did it break" but "when", because everything after that point was
    wasted and everything before it is still usable output. It also
    separates cause from consequence -- a level-set overlap reported at the
    last step means much less if dt had already collapsed 30000 steps
    earlier.
    """
    dts = [d for d in log["dt"] if d > 0]
    if len(dts) < 10:
        return None
    early = sorted(dts[:min(window, max(len(dts) // 4, 10))])
    ref = early[len(early) // 2]
    for i, d in enumerate(log["dt"]):
        if d > 0 and ref / d >= factor:
            return {"index": i, "step": log["steps"][i], "t": log["t"][i],
                    "dt": d, "ratio": ref / d,
                    "wall": log["wall"][i] if i < len(log["wall"]) else None}
    return None


def eta(log: dict, t_end: float) -> dict:
    """How long the run needs at its CURRENT step size and step cost.

    Deliberately not an estimate of how long it *should* take. The question
    a stalled run poses is whether to keep waiting, and that is answered by
    extrapolating what it is doing now, however absurd the answer looks.
    An absurd answer IS the answer.
    """
    if not log["dt"] or not log["t"]:
        return {"ok": False, "reason": "no steps parsed"}
    t_now, dt_now = log["t"][-1], log["dt"][-1]
    remaining = t_end - t_now
    if remaining <= 0:
        return {"ok": True, "done": True, "steps_needed": 0, "seconds": 0.0}
    if dt_now <= 0:
        return {"ok": False, "reason": "dt is zero or negative"}

    per_step = None
    walls = [w for w in log["wall"] if w is not None]
    if len(walls) >= 2 and log["steps"][-1] > log["steps"][0]:
        per_step = ((walls[-1] - walls[0])
                    / max(log["steps"][-1] - log["steps"][0], 1))
    steps_needed = remaining / dt_now
    return {
        "ok": True, "done": False,
        "t_now": t_now, "dt_now": dt_now, "remaining": remaining,
        "steps_needed": steps_needed,
        "seconds_per_step": per_step,
        "seconds": (steps_needed * per_step) if per_step else None,
    }


def human_time(seconds) -> str:
    """A duration a person can react to. Years, when it is years."""
    if seconds is None:
        return "unknown"
    # Switch at the natural boundary, not at 1000x it: 44740 s is 12.4 h,
    # and reporting it as "746 min" makes a 12-hour waste look like a blip.
    for limit, scale, unit in ((60.0, 1.0, "s"),
                               (3600.0, 60.0, "min"),
                               (86400.0, 3600.0, "h"),
                               (86400.0 * 365.25, 86400.0, "days")):
        if seconds < limit:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds / (86400.0 * 365.25):.3g} years"
