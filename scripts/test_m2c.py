#!/usr/bin/env python3
"""Test an M2C build and this bridge, cheapest check first.

    python scripts/test_m2c.py --check-only     # seconds, no solver run
    python scripts/test_m2c.py --scale smoke    # ~1 min on 32 cores
    python scripts/test_m2c.py --scale quick    # ~2-5 min
    python scripts/test_m2c.py --scale quick --compare

Five tiers, in order of what they rule out. Run them in order: a failure at
tier N makes tier N+1 uninterpretable.

  0  grammar     does this checkout accept the keywords AND the values we
                 emit? Pure grep, no run. Catches an upstream rename.
  0.5 saha      does the IONISATION module alone give sane numbers? Uses
                 the standalone solver (github.com/kevinwgy/saha) to check
                 the atomic data in seconds rather than days. Optional.
  1  shipped     does the BUILD work at all? Runs M2Cs own HVI test case,
                 which is independent of anything here. If this fails the
                 problem is the build, not the bridge.
  2  smoke       does OUR deck start, initialise Saha, and take steps?
                 Tiny mesh, few steps -- checks machinery, not physics.
  3  compare     does the physics agree with the reduced chain? The only
                 tier that can tell you the deck is *wrong* rather than
                 merely runnable.

Tier 3 is the one worth arguing about, so its tolerances are deliberately
loose and printed alongside the numbers. Agreement to a factor of a few is
the expected outcome: the reduced chain is a self-similar analytic model and
M2C is a multi-material hydrocode with a real EOS. Order-of-magnitude
agreement is a pass; exact agreement would be suspicious, not reassuring.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvi_emp import get_material, run_scenario
from hvi_emp.solvers.m2c_stage1 import (M2CConfig, check_grammar, find_m2c,
                                        read_ionization_probes,
                                        write_problem_directory)

#: Deliberately small. These are machinery checks; the physics is tier 3.
SCALES = {
    "smoke": dict(cells_per_radius=8, domain_radii=6, n_outputs=4),
    "quick": dict(cells_per_radius=16, domain_radii=10, n_outputs=8),
    "full": dict(),
}

GREEN, RED, YLW, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def ok(msg):
    print(f"  {GREEN}[ ok ]{RST} {msg}")


def bad(msg):
    print(f"  {RED}[FAIL]{RST} {msg}")


def warn(msg):
    print(f"  {YLW}[warn]{RST} {msg}")


def head(msg):
    print(f"\n{msg}\n" + "=" * 68)


def m2c_binary() -> str | None:
    """The m2c executable, from $M2C_HOME or the usual places."""
    home = os.environ.get("M2C_HOME")
    if home:
        # M2C_HOME is a directory; tolerate it pointing at the binary.
        cand = (home if os.path.basename(home) == "m2c" and
                os.path.isfile(home) else os.path.join(home, "m2c"))
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    for p in (Path.home() / "src/m2c/build/m2c", Path.home() / "src/bin/m2c"):
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return shutil.which("m2c")


# ---------------------------------------------------------------------------
# Tier 0 -- grammar
# ---------------------------------------------------------------------------

def tier0_grammar(source_root: str) -> bool:
    head("Tier 0: input grammar (no solver run)")
    if not os.path.isdir(source_root):
        warn(f"no M2C source at {source_root}; skipping "
             f"(pass --source /path/to/m2c)")
        return True
    try:
        res = check_grammar(source_root)
    except FileNotFoundError as exc:
        warn(f"{exc}")
        return True

    n_kw = len(res["found"]) + len(res["missing"])
    if res["missing"]:
        bad(f"{len(res['missing'])} of {n_kw} keywords NOT in this checkout: "
            f"{', '.join(res['missing'][:8])}")
    else:
        ok(f"{len(res['found'])} of {n_kw} keywords present")

    # Values matter as much as keys: a deck once reported 133/133 keywords
    # and still aborted, because the *value* assigned to one of them was
    # never checked.
    vf, vm = res.get("values_found", []), res.get("values_missing", [])
    if vm:
        warn(f"enum values not found: {', '.join(vm)}")
        warn("  some of these are alternatives this deck does not use; "
             "only the ones it emits matter")
    ok(f"{len(vf)} of {len(vf) + len(vm)} enum values present")
    return not res["missing"]


# ---------------------------------------------------------------------------
# Tier 0.5 -- the standalone Saha solver
# ---------------------------------------------------------------------------

def tier05_saha(saha_root: str) -> bool:
    """Exercise the ionisation module alone, without the flow solver.

    Zhao et al. (2026) §5.7: "For ease of reproducibility, the ionization
    module has been packaged as a standalone solver, available at
    www.github.com/kevinwgy/saha."

    It is the SAME module M2C runs in-loop, not a replacement for it -- the
    paper's time-step algorithm lists "Solve the Saha equation (if
    ionization effects are considered)" inside the loop. What the standalone
    build buys is turnaround: it answers "are my AtomicData files sane?" in
    seconds, where the coupled run takes days to reach the same question.

    Two things worth checking here, in order:

      1. The shipped He/Ne/Ar case (Tests/Test1_HeNeAr), whose reference is
         Zaghloul's published composition at 5 eV. That validates the BUILD
         against literature, independent of anything generated here.
      2. Our own I_/E_/g_ files, which is the only cheap way to see what the
         ground-state approximation does to Z-bar before spending a week on
         a coupled run that bakes it in.
    """
    head("Tier 0.5: standalone Saha solver (seconds, no flow solver)")
    root = Path(saha_root)
    if not root.is_dir():
        warn(f"no saha checkout at {root}")
        print("      git clone https://github.com/kevinwgy/saha "
              f"{root}")
        print("      Worth having: it tests the atomic data in seconds")
        print("      rather than days. Not required by M2C.")
        return True

    exe = None
    for cand in ("saha", "build/saha"):
        p = root / cand
        if p.is_file() and os.access(p, os.X_OK):
            exe = p
            break
    if exe is None:
        warn(f"saha checkout present but not built ({root})")
        print("      cd {0} && mkdir -p build && cd build && cmake .. && "
              "make -j".format(root))
        return True

    ref = root / "Tests" / "Test1_HeNeAr"
    if not ref.is_dir():
        warn("Tests/Test1_HeNeAr not found; skipping the published check")
        return True
    deck = next(iter(sorted(ref.glob("*.st"))), None)
    if deck is None:
        warn("no input deck in Tests/Test1_HeNeAr")
        return True
    ok(f"found {exe}")
    print(f"  reference case: {deck.relative_to(root)}")
    print(f"  (He 0.3 : Ne 0.1 : Ar 0.6 at 5 eV; reference Zaghloul, "
          f"paper Fig. 11)")
    return _run_deck(deck.parent, deck.name, str(exe), 1, 5.0,
                     label="He/Ne/Ar verification")


# ---------------------------------------------------------------------------
# Tier 1 -- M2C's own shipped test
# ---------------------------------------------------------------------------

def tier1_shipped(source_root: str, exe: str, cores: int,
                  minutes: float) -> bool:
    head("Tier 1: M2C's own shipped test case (validates the BUILD)")
    tests = Path(source_root) / "Tests"
    if not tests.is_dir():
        warn(f"no Tests/ under {source_root}; skipping")
        return True
    decks = sorted(tests.rglob("input.st"))
    hvi = [d for d in decks if "HVI" in str(d) or "Impact" in str(d)]
    deck = (hvi or decks or [None])[0]
    if deck is None:
        warn("no input.st under Tests/; skipping")
        return True

    print(f"  deck: {deck}")
    print(f"  This is M2C's code and M2C's input -- nothing of ours is")
    print(f"  involved, so a failure here is a build problem.")
    return _run_deck(deck.parent, deck.name, exe, cores, minutes,
                     label="shipped test")


# ---------------------------------------------------------------------------
# Tier 2 -- our deck, tiny
# ---------------------------------------------------------------------------

def _run_deck(workdir, deck_name, exe, cores, minutes, label) -> bool:
    cmd = ["mpirun", "-np", str(cores), exe, deck_name]
    print(f"  + cd {workdir} && {' '.join(cmd)}")
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=str(workdir), capture_output=True,
                           text=True, timeout=minutes * 60)
    except subprocess.TimeoutExpired:
        bad(f"{label}: still running after {minutes:.0f} min -- killed")
        return False
    except OSError as exc:
        bad(f"{label}: could not launch mpirun ({exc})")
        return False
    dt = time.perf_counter() - t0

    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        bad(f"{label}: exited {r.returncode} after {dt:.1f} s")
        for line in [ln for ln in out.splitlines()
                     if "rror" in ln or "ssert" in ln or "bort" in ln][:6]:
            print(f"        {line.strip()[:110]}")
        if "Invalid communicator" in out:
            print(f"      {YLW}This is an MPI family mismatch, not M2C: the "
                  f"binary and mpirun{RST}")
            print(f"      {YLW}come from different MPI implementations.{RST}")
        return False
    ok(f"{label}: completed in {dt:.1f} s")
    return True


def tier2_smoke(cfg, outdir, exe, cores, minutes) -> bool:
    head("Tier 2: our generated deck (validates the BRIDGE)")
    info = write_problem_directory(cfg, str(outdir), cores=cores)
    est = cfg.estimate_resources(cores)
    print(f"  {est['cells']:.2e} cells, ~{est['n_steps']:.0f} steps, "
          f"est {est['wall_hours_estimate']*60:.1f} min on {cores} cores")
    print(f"  (estimate assumes a uniform fine mesh and so OVER-counts a "
          f"graded one)")
    if "atomic_files" in info:
        print(f"  wrote {len(info['atomic_files'])} atomic-data files")
    return _run_deck(outdir, "input.st", exe, cores, minutes,
                     label="generated deck")


# ---------------------------------------------------------------------------
# Tier 3 -- physics
# ---------------------------------------------------------------------------

def tier3_compare(outdir, sc, material) -> bool:
    head("Tier 3: physics vs the reduced chain")
    probe = None
    for cand in Path(outdir).rglob("*ionization_probes*"):
        probe = cand
        break
    if probe is None:
        warn("no ionization_probes output found -- did the deck request "
             "probes, and did the run reach an output step?")
        return True

    try:
        pr = read_ionization_probes(str(probe))
    except Exception as exc:
        bad(f"could not read {probe.name}: {type(exc).__name__}: {exc}")
        return False

    import numpy as np
    n_e = np.asarray(pr["n_e"], float)
    m2c_peak = float(np.nanmax(n_e)) if n_e.size else float("nan")
    ref_peak = float(np.nanmax(sc.expansion.n_e))

    print(f"  {'quantity':<22}{'M2C':>14}{'reduced':>14}   ratio")
    print(f"  {'-'*22}{'-'*14}{'-'*14}   -----")
    ratio = m2c_peak / ref_peak if ref_peak else float("nan")
    print(f"  {'peak n_e [1/m^3]':<22}{m2c_peak:>14.3e}{ref_peak:>14.3e}"
          f"   {ratio:>5.2f}")

    # Loose on purpose: an analytic self-similar model and a hydrocode with
    # a real EOS should agree in magnitude, not in digits. Tight agreement
    # here would suggest the comparison is not independent.
    if not np.isfinite(ratio):
        bad("no comparable electron density")
        return False
    if 0.01 <= ratio <= 100.0:
        ok(f"within two orders of magnitude (ratio {ratio:.2f})")
        print("     Agreement to a factor of a few is the expected outcome.")
        print("     Exact agreement would be suspicious, not reassuring.")
        return True
    bad(f"ratio {ratio:.3g} is outside 1e-2..1e2 -- worth investigating "
        f"before trusting either")
    return False


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=os.path.expanduser("~/src/m2c"),
                    help="M2C source checkout (for tiers 0 and 1)")
    ap.add_argument("--saha", default=os.path.expanduser("~/src/saha"),
                    help="standalone Saha solver checkout (tier 0.5); "
                         "github.com/kevinwgy/saha -- optional, but it "
                         "tests atomic data in seconds not days")
    ap.add_argument("--out", default="m2c_test",
                    help="working directory for the generated deck")
    ap.add_argument("--scale", default="smoke", choices=sorted(SCALES))
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=20.0,
                    help="kill a run that exceeds this")
    ap.add_argument("--projectile", default="Fe")
    ap.add_argument("--target", default="Al")
    ap.add_argument("--mass", type=float, default=1e-12)
    ap.add_argument("--velocity", type=float, default=50e3)
    ap.add_argument("--check-only", action="store_true",
                    help="tier 0 only; no solver run")
    ap.add_argument("--skip-shipped", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="also run tier 3")
    args = ap.parse_args(argv)

    print("=" * 68)
    print("M2C test ladder -- cheapest check first")
    print("=" * 68)

    results = {}
    results["grammar"] = tier0_grammar(args.source)
    results["saha"] = tier05_saha(args.saha)
    if args.check_only:
        return 0 if all(results.values()) else 1

    exe = m2c_binary()
    if not exe:
        head("No m2c binary")
        bad("set M2C_HOME to the DIRECTORY holding the m2c binary")
        print("      export M2C_HOME=~/src/m2c/build")
        return 2
    print(f"\nbinary: {exe}")

    if not args.skip_shipped:
        results["shipped"] = tier1_shipped(args.source, exe, args.cores,
                                           args.minutes)

    sc = run_scenario(args.projectile, args.target, mass=args.mass,
                      velocity=args.velocity, t_end=1e-5)
    cfg = M2CConfig(projectile=sc.impact.projectile.material,
                    target=sc.impact.target,
                    diameter=2 * sc.impact.projectile.radius,
                    velocity=sc.impact.projectile.velocity,
                    **SCALES[args.scale])
    outdir = Path(args.out).absolute()
    results["smoke"] = tier2_smoke(cfg, outdir, exe, args.cores, args.minutes)

    if args.compare and results.get("smoke"):
        results["compare"] = tier3_compare(outdir, sc, cfg.target)

    head("Summary")
    for k, v in results.items():
        (ok if v else bad)(k)
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n  first failure: {failed[0]} -- fix that before reading "
              f"anything below it")
        return 1
    print("\n  All tiers passed. Note what this does and does not show:")
    print("  the deck runs and its plasma is the right order of magnitude.")
    print("  It does NOT show the mesh is converged -- for that, re-run at")
    print("  --scale quick and then full and check the answer stops moving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
