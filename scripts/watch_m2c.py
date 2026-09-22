#!/usr/bin/env python3
"""Is this M2C run still worth waiting for?

    python scripts/watch_m2c.py run.log                  # post-mortem
    python scripts/watch_m2c.py run.log --dx 9.7e-8      # + wave-speed check
    python scripts/watch_m2c.py run.log --follow --abort # kill it if dead

Why
---
M2C rarely fails by stopping. It fails by taking smaller and smaller steps
while still printing `Step N: ...`, so from outside it looks like progress.
One run reached t = 8.4e-10 s after 129521 steps and 12.4 hours with
`dt = 1.885042e-22` -- seven orders of magnitude below its own average step.
It then ran on in that state until an unrelated check finally aborted it.
Every hour after the collapse was wasted, and nothing in the log said so.

dt is the signal because the solver is not free to choose it:

    dt = CFL * dx / (|u| + c)

so dt implies a wave speed. Give `--dx` and this inverts it. For that run
the implied |u| + c was ~5e13 m/s, about 10^5 times the speed of light --
which settles the question. It is not a small time step, it is an equation
of state returning a sound speed it cannot have, because it is being
evaluated outside the density range it was fitted for.

`--follow --abort` is for job scripts: watch the log, and when dt has
collapsed, stop rather than hold the allocation until walltime.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp.solvers.m2c_log import (C_LIGHT_M_S, WARNING_PATTERNS,
                                     dt_verdict, eta, human_time,
                                     implied_wave_speed, onset, parse_log,
                                     wave_speed_ratio)

GRN, RED, YLW, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def ok(m):
    print(f"  {GRN}[ ok ]{RST} {m}")


def warn(m):
    print(f"  {YLW}[warn]{RST} {m}")


def bad(m):
    print(f"  {RED}[FAIL]{RST} {m}")


def head(m):
    print(f"\n{m}\n" + "=" * 70)


def report(log, args) -> int:
    """0 healthy, 1 stalling, 2 collapsed or aborted."""
    head("Progress")
    if not log["n_steps"]:
        bad("no `Step N: t = ..., dt = ...` lines found")
        print("      Either the run has not started, or this is not an M2C "
              "log.")
        return 2

    s, t, d = log["steps"][-1], log["t"][-1], log["dt"][-1]
    walls = [w for w in log["wall"] if w is not None]
    print(f"  last step   {s}")
    print(f"  t           {t:.6e} s")
    print(f"  dt          {d:.6e} s")
    if walls:
        print(f"  wall        {human_time(walls[-1])} "
              f"({walls[-1] / max(s, 1):.3g} s/step)")
    print(f"  mean dt     {t / max(s, 1):.6e} s  (t / steps so far)")

    v = dt_verdict(log)
    head("Time step")
    print(f"  early median dt   {v.get('dt_early', float('nan')):.4e} s")
    print(f"  recent median dt  {v.get('dt_now', float('nan')):.4e} s")
    if v["state"] == "collapsed":
        bad(v["reason"])
    elif v["state"] == "stalling":
        warn(v["reason"])
    elif v["state"] == "healthy":
        ok(v["reason"])
    else:
        warn(v["reason"])

    o = onset(log)
    if o and v["state"] in ("collapsed", "stalling"):
        head("When it went wrong")
        print(f"  dt first fell 10x below the early median at step "
              f"{o['step']}, t = {o['t']:.4e} s")
        if o["wall"] is not None and log["wall"][-1] is not None:
            wasted = log["wall"][-1] - o["wall"]
            print(f"  everything after that was wasted: "
                  f"{human_time(wasted)} of the "
                  f"{human_time(log['wall'][-1])} run")
        print("  Output written BEFORE that step is still usable; output "
              "after it is not.")

    # ------------------------------------------------------------------
    # The wave-speed argument, in the form that needs no cell size.
    #
    # dt = CFL*dx/(|u|+c) with the same mesh and CFL, so the RATIO of
    # implied speeds is just the inverse ratio of dt -- dx cancels. That
    # matters here: the decks use a graded mesh, the CFL condition sees the
    # smallest cell, and the nominal spacing over-states the speed by the
    # grading factor. An earlier version of this quoted an absolute speed
    # from the nominal dx and was wrong by ~50x -- in the alarming
    # direction, which is the worst way to be wrong.
    r = wave_speed_ratio(log)
    if r and r["ratio"] > 10:
        head("Implied wave speed")
        print(f"  dt = CFL * dx / (|u| + c). Same mesh, same CFL, so dx")
        print(f"  cancels and the speed ratio is just the dt ratio:")
        print(f"      (|u|+c) now / (|u|+c) early = {r['ratio']:.3e}")
        print()
        print(f"  Whatever the early state really was, the final one is:")
        for v0, v1 in sorted(r["implied"].items()):
            mark = "  <-- superluminal" if v1 > C_LIGHT_M_S else ""
            print(f"      early {v0/1e3:6.0f} km/s  ->  final {v1:.3e} m/s "
                  f"= {v1/C_LIGHT_M_S:8.3g} c{mark}")
        if min(r["implied"].values()) > C_LIGHT_M_S:
            bad("superluminal across every plausible starting value")
            print("      A time step cannot be argued with: the EOS returned "
                  "a sound")
            print("      speed it cannot have. This is a thermodynamic-state "
                  "failure,")
            print("      not a resolution or a tuning problem, and no "
                  "walltime fixes it.")
        else:
            warn("the wave speed has grown far beyond anything physical")

    if args.dx:
        head("Absolute wave speed (needs the SMALLEST cell, not the nominal)")
        sp = implied_wave_speed(d, log["cfl"][-1], args.dx)
        print(f"  with dx = {args.dx:.3e} m:  |u| + c = {sp:.4e} m/s "
              f"({sp / C_LIGHT_M_S:.3g} c)")
        print(f"  On a graded mesh the CFL condition uses the finest cell.")
        print(f"  If you passed the nominal spacing, this is too high by the")
        print(f"  grading factor -- prefer the ratio above, which needs no dx.")

    # ------------------------------------------------------------------
    head("Warnings")
    any_warn = False
    for key, _rx, why in WARNING_PATTERNS:
        n = log["counts"][key]
        if not n:
            continue
        any_warn = True
        tot = log["totals"].get(key, 0)
        extra = f", {tot} edge(s)/occurrence(s) total" if tot else ""
        step_at, sample = log["first_seen"].get(key, (None, ""))
        at = f" from step {step_at}" if step_at is not None else ""
        warn(f"{key}: {n} line(s){extra}{at}")
        print(f"         {why}")
    if not any_warn:
        ok("none of the known failure signatures appear")

    # ------------------------------------------------------------------
    if args.t_end:
        head("Worth waiting for?")
        e = eta(log, args.t_end)
        if not e.get("ok"):
            warn(e.get("reason", "cannot estimate"))
        elif e.get("done"):
            ok(f"already past t_end = {args.t_end:.3e} s")
        else:
            print(f"  remaining      {e['remaining']:.4e} s of simulated time")
            print(f"  at current dt  {e['steps_needed']:.4e} more steps")
            if e["seconds"] is not None:
                txt = human_time(e["seconds"])
                print(f"  wall needed    {txt}")
                if e["seconds"] > 30 * 86400:
                    bad(f"{txt} at the current rate -- this run will not "
                        f"finish")
                elif e["seconds"] > 2 * 86400:
                    warn(f"{txt} at the current rate")
                else:
                    ok(f"{txt} at the current rate")

    # ------------------------------------------------------------------
    head("Verdict")
    if log["fatal"]:
        bad(f"the run aborted: {log['fatal']}")
    if v["state"] == "collapsed" or log["fatal"]:
        if log["counts"].get("two_subdomains"):
            print("  A node in two material subdomains means the level sets")
            print("  overlapped. That is fatal, but it is usually the LAST")
            print("  thing to break rather than the first: check whether dt")
            print("  and the Riemann warnings started earlier (above).")
        print("\n  Do not re-run this deck unchanged. Cheapest experiments,")
        print("  in order:")
        print("    1. Same deck at a velocity inside the EOS's calibrated")
        print("       range (~20 km/s). If that survives, the EOS out at")
        print("       50 km/s is the cause and the rest is downstream.")
        print("    2. Same velocity, coarser mesh, short t_end -- does the")
        print("       collapse arrive at the same simulated TIME or the same")
        print("       step COUNT? Time means physics; count means numerics.")
        print("    3. Only then tune the level-set reinitialisation.")
        print("\n  See docs/M2C_FAILURE_MODES.md")
        return 2
    if v["state"] == "stalling":
        return 1
    return 0


def follow(path: Path, args) -> int:
    """Watch a growing log and stop when it is clearly dead."""
    head(f"Following {path}")
    print(f"  checking every {args.interval:.0f}s; "
          f"{'will abort the run' if args.abort else 'reporting only'}")
    seen = 0
    while True:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as exc:
            warn(f"cannot read log: {exc}")
            time.sleep(args.interval)
            continue
        log = parse_log(lines)
        v = dt_verdict(log)
        if log["n_steps"] != seen:
            seen = log["n_steps"]
            print(f"  step {log['steps'][-1] if log['steps'] else '?'}  "
                  f"dt {log['dt'][-1]:.3e}  [{v['state']}]")
        if v["state"] == "collapsed" or log["fatal"]:
            print()
            rc = report(log, args)
            if args.abort and args.pid:
                bad(f"aborting pid {args.pid}")
                try:
                    os.kill(args.pid, signal.SIGTERM)
                except OSError as exc:
                    warn(f"could not signal {args.pid}: {exc}")
            elif args.abort:
                warn("--abort given without --pid: nothing to signal")
                print("      Pass the launcher's pid, e.g. in a job script:")
                print("        mpirun ... &")
                print("        python scripts/watch_m2c.py run.log --follow "
                      "--abort --pid $!")
            return rc
        time.sleep(args.interval)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="M2C stdout/stderr log")
    ap.add_argument("--dx", type=float,
                    help="cell size in metres; turns dt into a wave speed")
    ap.add_argument("--t-end", type=float,
                    help="target simulated time, for the ETA")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--abort", action="store_true",
                    help="with --follow and --pid, SIGTERM a dead run")
    ap.add_argument("--pid", type=int)
    args = ap.parse_args(argv)

    path = Path(args.log)
    if not path.is_file():
        bad(f"no such log: {path}")
        return 2
    if args.follow:
        return follow(path, args)
    return report(parse_log(path.read_text(errors="replace").splitlines()),
                  args)


if __name__ == "__main__":
    raise SystemExit(main())
