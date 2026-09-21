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
import re
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
        return None
    try:
        res = check_grammar(source_root)
    except FileNotFoundError as exc:
        warn(f"{exc}")
        return None

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

def tier05_saha(saha_root: str):
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
        return None      # skipped, not passed

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
        return None

    ref = root / "Tests" / "Test1_HeNeAr"
    if not ref.is_dir():
        warn("Tests/Test1_HeNeAr not found; skipping the published check")
        return None
    deck = next(iter(sorted(ref.glob("*.st"))), None)
    if deck is None:
        warn("no input deck in Tests/Test1_HeNeAr")
        return None
    ok(f"found {exe}")
    print(f"  reference case: {deck.relative_to(root)}")
    print(f"  (He 0.3 : Ne 0.1 : Ar 0.6 at 5 eV; reference Zaghloul, "
          f"paper Fig. 11)")
    return _run_deck(deck.parent, deck.name, str(exe), 1, 5.0,
                     label="He/Ne/Ar verification", launcher="mpirun")


# ---------------------------------------------------------------------------
# Tier 1 -- M2C's own shipped test
# ---------------------------------------------------------------------------

def tier1_shipped(source_root: str, exe: str, cores: int,
                  minutes: float, launcher: str = "mpirun"):
    head("Tier 1: M2C's own shipped test case (validates the BUILD)")
    tests = Path(source_root) / "Tests"
    if not tests.is_dir():
        warn(f"no Tests/ under {source_root}; skipping")
        return None
    decks = sorted(tests.rglob("input.st"))
    hvi = [d for d in decks if "HVI" in str(d) or "Impact" in str(d)]
    deck = (hvi or decks or [None])[0]
    if deck is None:
        warn("no input.st under Tests/; skipping")
        return None

    print(f"  deck: {deck}")
    print(f"  This is M2C's code and M2C's input -- nothing of ours is")
    print(f"  involved, so a failure here is a build problem.")
    return _run_deck(deck.parent, deck.name, exe, cores, minutes,
                     label="shipped test", launcher=launcher)


# ---------------------------------------------------------------------------
# Tier 2 -- our deck, tiny
# ---------------------------------------------------------------------------

#: Launchers worth trying, most specific first. `srun` last: on a cluster it
#: works only inside an allocation, so finding it on PATH does not mean a
#: bare `srun` will run here.
_LAUNCHERS = ("mpirun", "mpiexec", "orterun", "srun")

#: Sonames, so the launcher can be matched to the family the binary needs.
#: `libmpi.so.12` is MPICH, `libmpi.so.40` is Open MPI. Launching one with
#: the other's mpirun gives "Invalid communicator" -- already hit once in
#: this project, and it is not obvious from the error that MPI is at fault.
_MPI_SONAME = re.compile(r"libmpi(?:_[a-z0-9]+)?\.so\.\d+")


def linked_mpi(exe: str) -> tuple:
    """(soname, resolved path) for the MPI the binary is actually linked to.

    The binary is the authority here. M2C was built against one MPI, and
    that MPI's launcher is the only one that can start it -- which is worth
    finding out from the ELF rather than from whatever happens to be on
    PATH today.
    """
    try:
        r = subprocess.run(["ldd", exe], capture_output=True, text=True,
                           timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None, None
    for line in (r.stdout or "").splitlines():
        m = _MPI_SONAME.search(line)
        if not m:
            continue
        parts = line.split("=>")
        path = parts[1].strip().split(" (")[0] if len(parts) > 1 else ""
        return m.group(0), (path or None)
    return None, None


def find_launcher(exe: str) -> tuple:
    """(launcher path or None, list of advice lines).

    Looks on PATH first, then beside the MPI the binary is linked against.
    An MPI installed under a prefix keeps `mpirun` in `<prefix>/bin` and
    `libmpi.so` in `<prefix>/lib`, so the library path names the launcher
    even when nothing is on PATH -- which is the usual state of a login
    shell that has not activated the environment M2C was built in.
    """
    soname, libpath = linked_mpi(exe)
    advice = []
    if soname:
        family = ("Open MPI" if soname.endswith((".40", ".20", ".12"))
                  and "libmpi.so.40" in soname else
                  "MPICH" if soname == "libmpi.so.12" else "unknown family")
        advice.append(f"the binary needs {soname} ({family})")

    for name in _LAUNCHERS:
        p = shutil.which(name)
        if p:
            return p, advice + [f"found {name} on PATH: {p}"]

    # Not on PATH. Derive it from the library the binary resolved to.
    if libpath:
        prefix = Path(libpath).resolve().parent
        for up in (prefix.parent, prefix.parent.parent):
            for name in _LAUNCHERS[:3]:
                cand = up / "bin" / name
                if cand.is_file() and os.access(cand, os.X_OK):
                    advice.append(
                        f"not on PATH, but the MPI this binary is linked "
                        f"against has one:")
                    advice.append(f"    {cand}")
                    advice.append(f"  add it for this shell with:")
                    advice.append(f"    export PATH={up / 'bin'}:$PATH")
                    return str(cand), advice

    advice.append("no MPI launcher found on PATH or beside the linked MPI")
    if libpath:
        advice.append(f"  the binary resolves {soname} to {libpath},")
        advice.append(f"  so an MPI IS installed -- its bin/ is just not on "
                      f"PATH")
    advice.append("  If M2C was built inside a conda/mamba environment, "
                  "activate it:")
    advice.append("    micromamba activate <env>   # or: conda activate <env>")
    advice.append("  If it was built against a cluster module, load it:")
    advice.append("    module avail mpi   &&   module load <the one used>")
    advice.append("  Match the family above: launching a binary with another "
                  "MPI's")
    advice.append("  mpirun gives 'Invalid communicator', not a clear error.")
    return None, advice


def _run_deck(workdir, deck_name, exe, cores, minutes, label,
              launcher="mpirun") -> bool:
    cmd = [launcher, "-np", str(cores), exe, deck_name]
    print(f"  + cd {workdir} && {' '.join(cmd)}")
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=str(workdir), capture_output=True,
                           text=True, timeout=minutes * 60)
    except subprocess.TimeoutExpired:
        bad(f"{label}: still running after {minutes:.0f} min -- killed")
        return False
    except OSError as exc:
        bad(f"{label}: could not launch {launcher} ({exc})")
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


def tier2_smoke(cfg, outdir, exe, cores, minutes, launcher="mpirun") -> bool:
    head("Tier 2: our generated deck (validates the BRIDGE)")
    info = write_problem_directory(cfg, str(outdir), cores=cores)
    est = cfg.estimate_resources(cores)
    print(f"  {est['cells']:.2e} cells, ~{est['n_steps']:.0f} steps, "
          f"est {est['wall_hours_estimate']*60:.1f} min on {cores} cores")
    print(f"  (estimate assumes a uniform fine mesh and so OVER-counts a "
          f"graded one)")
    if "atomic_files" in info:
        print(f"  wrote {len(info['atomic_files'])} atomic-data files")

    # Do not start a run the timeout will certainly kill.
    #
    # `--scale full` on 8 cores estimates ~38 DAYS. Launching it under a
    # 20-minute kill produces a [FAIL] that says nothing about the bridge:
    # the deck would have been fine, it just never got near finishing. A
    # tier that cannot answer its question should decline to run rather
    # than answer it wrongly.
    est_min = est["wall_hours_estimate"] * 60.0
    if est_min > 4.0 * minutes:
        warn(f"declining to launch: estimated {est_min / 60:.1f} h "
             f"({est_min / 1440:.1f} days) against a {minutes:.0f} min kill")
        print("      This tier checks that the deck STARTS and takes steps.")
        print("      At this scale it cannot finish, so a timeout here would")
        print("      say nothing about the bridge. Either:")
        print(f"        python scripts/test_m2c.py --scale smoke")
        print(f"        python scripts/test_m2c.py --scale full "
              f"--minutes {est_min * 1.2:.0f}   # and mean it")
        print("      The cost model itself has a known pending correction "
              "(issue #34/#35);")
        print("      treat this number as a guard rail, not a schedule.")
        return None
    return _run_deck(outdir, "input.st", exe, cores, minutes,
                     label="generated deck", launcher=launcher)


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
        return _summary(results)

    exe = m2c_binary()
    if not exe:
        head("No m2c binary")
        bad("set M2C_HOME to the DIRECTORY holding the m2c binary")
        print("      export M2C_HOME=~/src/m2c/build")
        return 2
    print(f"\nbinary: {exe}")

    # ---------------------------------------------------------------------
    # One pre-flight for the launcher, before any tier tries to use it.
    #
    # Without this, a missing mpirun is reported once per tier as though
    # each were a separate failure, and the summary blames the tiers. There
    # is only one problem and it is not M2C's.
    # ---------------------------------------------------------------------
    head("MPI launcher")
    launcher, advice = find_launcher(exe)
    for line in advice:
        print(f"  {line}")
    if launcher is None:
        bad("cannot run any deck without an MPI launcher")
        print("\n  Nothing below this can run, so nothing below it is being")
        print("  attempted. The grammar check above needs no MPI and passed;")
        print("  that part of the bridge is fine.")
        results["launcher"] = False
        _summary(results)
        return 2
    ok(f"using {launcher}")

    if not args.skip_shipped:
        results["shipped"] = tier1_shipped(args.source, exe, args.cores,
                                           args.minutes, launcher=launcher)

    sc = run_scenario(args.projectile, args.target, mass=args.mass,
                      velocity=args.velocity, t_end=1e-5)
    cfg = M2CConfig(projectile=sc.impact.projectile.material,
                    target=sc.impact.target,
                    diameter=2 * sc.impact.projectile.radius,
                    velocity=sc.impact.projectile.velocity,
                    **SCALES[args.scale])
    outdir = Path(args.out).absolute()
    results["smoke"] = tier2_smoke(cfg, outdir, exe, args.cores, args.minutes,
                                   launcher=launcher)

    if args.compare and results.get("smoke"):
        results["compare"] = tier3_compare(outdir, sc, cfg.target)

    return _summary(results)


def _summary(results) -> int:
    """Three states, not two.

    A tier that could not run is not a tier that passed. Printing `[ ok ]`
    for a skipped Saha check -- which is what happened when every early
    return said `True` -- claims the atomic data was verified when the
    solver for it was never even present.
    """
    head("Summary")
    for k, v in results.items():
        if v is None:
            warn(f"{k}  (skipped -- not run, so not evidence of anything)")
        else:
            (ok if v else bad)(k)
    failed = [k for k, v in results.items() if v is False]
    skipped = [k for k, v in results.items() if v is None]
    if failed:
        print(f"\n  first failure: {failed[0]} -- fix that before reading "
              f"anything below it")
        return 1
    if skipped:
        print(f"\n  Nothing failed, but {len(skipped)} tier(s) were skipped: "
              f"{', '.join(skipped)}.")
        print("  The ladder is only as strong as the rungs that ran.")
    print("\n  What a full pass does and does not show: the deck runs and its")
    print("  plasma is the right order of magnitude. It does NOT show the")
    print("  mesh is converged -- for that, re-run at --scale quick and then")
    print("  full and check the answer stops moving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
