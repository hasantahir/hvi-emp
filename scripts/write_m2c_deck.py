#!/usr/bin/env python3
"""Write an M2C run directory (input.st + AtomicData + results/) for the waterfall.

    python scripts/write_m2c_deck.py                       # Al -> Al, 32 km/s
    python scripts/write_m2c_deck.py --velocity 52e3
    python scripts/write_m2c_deck.py --series              # all six of Fletcher's speeds
    python scripts/write_m2c_deck.py --projectile W        # W -> Al (W not yet on Tillotson)

Defaults follow Fletcher (2021) Fig. 4: a 1 mm projectile (the report does
not state the size), normal incidence, run to 12 us. The projectile defaults
to Al rather than his W because W has no verified Tillotson constants yet,
and a W projectile vaporises in exactly the regime where Mie-Grueneisen has
no physics. See docs/WATERFALL.md.

Each directory gets:

    input.st     the deck (Tillotson where verified, all 14 keys written)
    AtomicData/  ionisation data the deck references
    results/     created empty; M2C will not create it itself
    README.txt   run command, cost estimate, and every note about the deck

Then:  cd <dir> && mpirun -np 64 $M2C_HOME/m2c input.st
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

GRN, YLW, RED, RST = "\033[32m", "\033[33m", "\033[31m", "\033[0m"
FLETCHER_VELOCITIES = (12e3, 22e3, 32e3, 42e3, 52e3, 62e3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projectile", default="Al")
    ap.add_argument("--target", default="Al")
    ap.add_argument("--diameter", type=float, default=1.0e-3, help="[m]")
    ap.add_argument("--velocity", type=float, default=32e3, help="[m/s]")
    ap.add_argument("--series", action="store_true",
                    help="write all six of Fletcher's speeds, 12-62 km/s")
    ap.add_argument("--t-end", type=float, default=12e-6,
                    help="[s]; Fletcher's Fig. 4 is at 12 us")
    ap.add_argument("--n-outputs", type=int, default=120,
                    help="snapshots; the handoff picks among these")
    ap.add_argument("--eos", choices=("auto", "tillotson", "mie-gruneisen"),
                    default="auto")
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--cells-per-radius", type=float,
                    help="resolution; cost goes as its cube in 2-D "
                         "(cells^2 x steps). Try 20 for a first, cheap run")
    ap.add_argument("--out", default=str(_REPO / "runs" / "m2c"),
                    help="parent directory for the run directories")
    args = ap.parse_args(argv)

    from hvi_emp import get_material
    from hvi_emp.solvers.m2c_stage1 import M2CConfig, write_problem_directory

    proj, targ = get_material(args.projectile), get_material(args.target)
    speeds = FLETCHER_VELOCITIES if args.series else (args.velocity,)

    for v in speeds:
        cfg = M2CConfig(projectile=proj, target=targ, diameter=args.diameter,
                        velocity=v, t_end=args.t_end,
                        n_outputs=args.n_outputs, eos=args.eos,
                        cells_per_radius=args.cells_per_radius)
        d = os.path.join(args.out, f"{proj.name.lower()}_{targ.name.lower()}"
                                   f"_v{v / 1e3:.0f}kms")
        res = write_problem_directory(cfg, d, cores=args.cores)
        deck = res["paths"]["input"] if "paths" in res else \
            os.path.join(d, "input.st")
        text = open(deck).read()
        eos = re.findall(r"EquationOfState = (\w+);\s*//?.*?\n?", text)
        eos = re.findall(r"under Material\[(\d)\] \{ // ([^\n]+)\n"
                         r"(?:.*\n)*?\s*EquationOfState = (\w+);", text)
        est = res.get("estimate", {})

        print(f"\n{GRN}{os.path.abspath(deck)}{RST}")
        for mid, name, kind in eos:
            role = {"0": "ambient", "1": "target", "2": "projectile"}[mid]
            print(f"  material {mid} ({role:10s}) {name.split()[0]:4s} -> {kind}")
        if est:
            lo = est.get("wall_hours_estimate", float("nan"))
            hi = est.get("wall_hours_if_dt_as_last_run", float("nan"))
            def h(x):
                return f"{x:.1f} h" if x < 10 else f"{x:.0f} h"
            days = (f" ({lo / 24:.1f}-{hi / 24:.0f} days)"
                    if hi >= 48 else "")
            print(f"  {est.get('cells', 0) / 1e6:.2f} M cells; "
                  f"{h(lo)} on {args.cores} cores at the CFL step, "
                  f"{h(hi)} if the step shrinks as it did last run{days}")
        for n in cfg.notes:
            print(f"  {YLW}! {n}{RST}")

    m2c = os.environ.get("M2C_HOME")
    if m2c:
        sys.path.insert(0, str(_REPO / "scripts"))
        from patch_m2c_tillotson import find_header, is_patched
        try:
            hdr = find_header(Path(m2c).expanduser().resolve())
        except FileNotFoundError:
            hdr = None
        exe = Path(m2c).expanduser() / "m2c"
        if hdr is not None and is_patched(hdr.read_text()):
            if exe.is_file() and exe.stat().st_mtime < hdr.stat().st_mtime:
                print(f"\n{RED}M2C source is patched but {exe} is older than "
                      f"the patch: rebuild it first.{RST}")
                print(f"  make -C {exe.parent} -j 16")
            else:
                print(f"\n{GRN}M2C source is patched for Tillotson "
                      f"({hdr}).{RST}")
        elif hdr is not None:
            print(f"\n{RED}M2C at {hdr.parent} is NOT patched; it will abort "
                  f"(signal 6) on this deck. First:{RST}")
            print(f"  python {_REPO / 'scripts' / 'patch_m2c_tillotson.py'} "
                  f"{hdr.parent}")
            print(f"  make -C {hdr.parent / 'build'} -j 16")

    print("\nRun, with the dt-collapse watchdog alongside:")
    print("  cd <dir>")
    print(f"  mpirun -np {args.cores} $M2C_HOME/m2c input.st > m2c.log 2>&1 &")
    print(f"  python {_REPO / 'scripts' / 'watch_m2c.py'} m2c.log "
          f"--follow --abort --pid $!")
    print("Then hand over:")
    print(f"  python {_REPO / 'scripts' / 'm2c_handoff.py'} <dir>/results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
