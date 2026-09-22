#!/usr/bin/env python3
"""Re-emit the PIConGPU .param files against the PIConGPU you actually have.

    python scripts/fix_picongpu_params.py full_chain/pic
    python scripts/fix_picongpu_params.py full_chain/pic --picongpu ~/src/picongpu

The symptom this exists for
---------------------------
    error: type "picongpu::UsedParticleShape" has already been defined
           (previous definition at line 48 of .../picongpu/param/species.param)
    error: invalid redeclaration of type name "picongpu::UsedField2Particle"
    error: invalid redeclaration of type name "picongpu::UsedParticleCurrentSolver"

PIConGPU moved those aliases out of `speciesDefinition.param` and into
`species.param`. `species.param` is included first, so a generated
`speciesDefinition.param` that declares them again is not merely redundant --
nvcc stops.

This does not patch the broken file line by line. It regenerates it from the
same configuration, with the generator inspecting your checkout to decide
which aliases it may declare. The old file is kept alongside with a
`.superseded` suffix so the diff is there to read.

Everything else in the directory is left alone: only the `.param` files this
project generates are rewritten.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRN, RED, YLW, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def ok(m):
    print(f"  {GRN}[ ok ]{RST} {m}")


def warn(m):
    print(f"  {YLW}[warn]{RST} {m}")


def bad(m):
    print(f"  {RED}[FAIL]{RST} {m}")


def head(m):
    print(f"\n{m}\n" + "=" * 70)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pic_dir", nargs="?", default="full_chain/pic",
                    help="the generated PIConGPU input directory")
    ap.add_argument("--picongpu", help="PIConGPU source checkout "
                                       "(default: $PICSRC, then ~/src/picongpu)")
    ap.add_argument("--projectile", default="Fe")
    ap.add_argument("--target", default="Al")
    ap.add_argument("--mass", type=float, default=1e-12)
    ap.add_argument("--velocity", type=float, default=50e3)
    ap.add_argument("--frequency", type=float, default=916e6,
                    help="band the plume epoch is chosen for [Hz]")
    ap.add_argument("--dry-run", "-n", action="store_true")
    args = ap.parse_args(argv)

    from hvi_emp.solvers.picongpu_stage3 import (SHARED_ALIASES,
                                                 find_picongpu,
                                                 upstream_aliases)

    root = Path(args.pic_dir)
    param_dir = root / "include" / "picongpu" / "param"
    if not param_dir.is_dir():
        bad(f"no {param_dir}")
        print("  Point this at the directory the chain wrote, e.g.")
        print("    python scripts/fix_picongpu_params.py full_chain/pic")
        return 2

    # ---------------------------------------------------------------- check
    head("Your PIConGPU")
    pic = find_picongpu(args.picongpu)
    if pic is None:
        bad("cannot find a PIConGPU checkout")
        print("  Pass --picongpu /path/to/picongpu, or export PICSRC.")
        print("  Without it there is no way to tell which layout to emit for.")
        return 2
    ok(f"{pic}")
    found = upstream_aliases(pic)
    if found:
        for name, where in sorted(found.items()):
            print(f"    declares {name:<26} in {where}")
        print("  -> the generated file must NOT declare these.")
    else:
        warn("this checkout declares none of the shared aliases")
        print("  -> the generated file must declare them itself "
              "(the older layout).")

    # --------------------------------------------------------------- damage
    head("The file on disk")
    target = param_dir / "speciesDefinition.param"
    if not target.is_file():
        bad(f"no {target.name} to fix")
        return 2
    current = target.read_text(errors="replace")
    clashing = sorted(a for a in SHARED_ALIASES
                      if f"using {a} =" in current and a in found)
    if clashing:
        bad(f"declares {len(clashing)} alias(es) your PIConGPU already has: "
            f"{', '.join(clashing)}")
        print("  This is exactly the compile error you are seeing.")
    elif found and not clashing:
        ok("no clashing declarations -- this file is already correct")
        print("  If the build still fails, the error is elsewhere; send the "
              "first\n  error rather than the last.")
        return 0
    else:
        ok("nothing clashes with this checkout")

    if args.dry_run:
        print(f"\n  dry run: would regenerate {target}")
        return 0

    # ------------------------------------------------------------ regenerate
    head("Regenerating")
    from hvi_emp import run_scenario
    from hvi_emp.solvers.picongpu_stage3 import (PIConGPUConfig,
                                                 write_species_definition_param)

    sc = run_scenario(args.projectile, args.target, mass=args.mass,
                      velocity=args.velocity, t_end=1e-5)
    # Same constructor the chain uses, so the regenerated file carries the
    # same plasma state and not a default one.
    try:
        cfg = PIConGPUConfig.from_expansion(sc.expansion, at_frequency=args.frequency)
        ok(f"config from the expansion stage: n_e0 = {cfg.n_e0:.3e} m^-3, "
           f"T = {cfg.T_eV:.3g} eV")
    except Exception as exc:                                # noqa: BLE001
        warn(f"could not rebuild the config ({type(exc).__name__}: {exc})")
        print("      Falling back to defaults. The shared-alias block is what")
        print("      this fix changes, but check the densities in the file")
        print("      before running anything expensive.")
        cfg = PIConGPUConfig(n_e0=1.0e24, T_eV=2.0)

    backup = target.with_suffix(".param.superseded")
    shutil.copy2(target, backup)
    write_species_definition_param(cfg, str(target), picongpu_root=pic)
    ok(f"rewrote {target}")
    print(f"  previous version kept at {backup.name}")
    print(f"    diff {backup} {target}")

    new = target.read_text()
    still = sorted(a for a in SHARED_ALIASES
                   if f"using {a} =" in new and a in found)
    if still:
        bad(f"still declares {', '.join(still)} -- this did not work")
        return 1
    ok("no alias in this file collides with your PIConGPU")

    head("Next")
    print("  Rebuild. pic-build compiles the .param files into the binary, so")
    print("  nothing changes until you do:")
    print(f"    cd {root}")
    print("    pic-build -b cuda")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
