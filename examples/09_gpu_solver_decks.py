#!/usr/bin/env python3
"""Generate GPU-solver inputs: Idefix (Stage 2 MHD) and PIConGPU (Stage 3 PIC).

Runs the reduced chain, then writes complete, compilable input sets for both
GPU-capable solvers from the computed plume state.  Neither solver needs to be
installed for this to work -- that is the point of the bridges.

    python examples/09_gpu_solver_decks.py
    python examples/09_gpu_solver_decks.py --gpu A100 --velocity 30e3
    python examples/09_gpu_solver_decks.py --outdir my_decks --dim 2
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvi_emp import run_scenario
from hvi_emp.solvers.idefix_stage2 import (IdefixConfig, cmake_command,
                                           write_problem_directory)
from hvi_emp.solvers.picongpu_stage3 import (PIConGPUConfig, build_command,
                                             write_input_set)

#: Kokkos architecture macro per card, for the Idefix cmake line.
KOKKOS_ARCH = {"V100": "VOLTA70", "A100": "AMPERE80", "A30": "AMPERE80",
               "A40": "AMPERE86", "RTX3090": "AMPERE86",
               "L40": "ADA89", "L40S": "ADA89", "RTX4090": "ADA89",
               "RTX6000Ada": "ADA89",
               "H100": "HOPPER90", "H200": "HOPPER90",
               "B200": "BLACKWELL100", "GB200": "BLACKWELL100",
               "RTX5070Ti": "BLACKWELL120", "RTX5080": "BLACKWELL120",
               "RTX5090": "BLACKWELL120", "RTXPRO6000": "BLACKWELL120"}


def banner(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projectile", default="Fe")
    ap.add_argument("--target", default="Al")
    ap.add_argument("--mass", type=float, default=1e-12, help="kg")
    ap.add_argument("--velocity", type=float, default=50e3, help="m/s")
    ap.add_argument("--gpu", default=None,
                    help=f"card name, one of {sorted(KOKKOS_ARCH)}")
    ap.add_argument("--dim", type=int, default=3, choices=(2, 3),
                    help="PIConGPU dimensionality")
    ap.add_argument("--frequency", type=float, default=916e6,
                    help="band to initialise the PIC run for [Hz]")
    ap.add_argument("--outdir", default="gpu_decks")
    ap.add_argument("--mpi", action="store_true",
                    help="add MPI to the Idefix build")
    args = ap.parse_args()

    banner("Reduced chain")
    sc = run_scenario(args.projectile, args.target, mass=args.mass,
                      velocity=args.velocity)
    print(sc.summary())

    if sc.expansion is None:
        print("\nStage 1 produced no plasma at this velocity -- nothing to "
              "hand to a solver. Try a higher --velocity.")
        return 1

    os.makedirs(args.outdir, exist_ok=True)

    # -- Stage 2: Idefix ---------------------------------------------------
    banner("Stage 2 -> Idefix (MHD, spherical, GPU via Kokkos)")
    icfg = IdefixConfig.from_expansion(sc.expansion)
    arch = KOKKOS_ARCH.get(args.gpu or "")
    idir = os.path.join(args.outdir, "idefix_cavity")
    out = write_problem_directory(icfg, idir, gpu_arch=arch, mpi=args.mpi)

    n = icfg.normalised()
    est = icfg.cavity()
    cost = icfg.cost_estimate()
    print(f"  geometry           {icfg.geometry}, "
          f"{icfg.n_r} x {icfg.n_theta}, r in "
          f"[{icfg.r_min_frac:.3g}, {icfg.domain_R:.4g}] R0 (log)")
    print(f"  R0                 {icfg.R0:.4e} m")
    print(f"  ambient beta       {n['beta_ambient']:.3e}")
    print(f"  v_exp / v_A        {n['v_expansion_hat']:.3e}")
    print(f"  magnetic Reynolds  {n['Rm']:.3e}")
    print(f"  expected cavity    {est['R_cavity']:.3e} m "
          f"({est['R_cavity'] / icfg.R0:.1f} R0) at "
          f"t = {est['t_formation']:.3e} s")
    print(f"  estimated cost     {cost['hours']:.2f} GPU-hours, "
          f"{cost['output_GB']:.2f} GB of VTK")
    print(f"  written to         {idir}/")
    for f in sorted(os.listdir(idir)):
        print(f"                       {f}")
    print("\n  build with:")
    for line in out["cmake"].splitlines():
        print(f"    {line}")

    if icfg.notes:
        print("\n  configuration audit:")
        for w in icfg.notes:
            print(f"    ! {w}")

    # -- Stage 3: PIConGPU -------------------------------------------------
    banner("Stage 3 -> PIConGPU (EM PIC, GPU-native)")
    from hvi_emp.solvers.picongpu_stage3 import GPU_MEMORY_GB
    pcfg = PIConGPUConfig.from_expansion(
        sc.expansion, at_frequency=args.frequency, dim=args.dim,
        gpu_memory_GB=GPU_MEMORY_GB.get(args.gpu or "", 24.0))
    pdir = os.path.join(args.outdir, "picongpu_emp")
    written = write_input_set(pcfg, pdir, gpu=args.gpu)

    g = pcfg.grid()
    pcost = pcfg.cost_estimate()
    print(f"  initialised at     {args.frequency / 1e6:.0f} MHz resonance")
    print(f"  peak n_e           {pcfg.n_e0:.4e} m^-3   T_e {pcfg.T_eV:.4g} eV")
    print(f"  Debye length       {g['lambda_debye']:.4e} m")
    print(f"  cell size          {g['dx']:.4e} m "
          f"({g['cells_per_debye_actual']:.3g} per Debye, "
          f"{'resolved' if g['debye_resolved'] else 'UNDER-RESOLVED'})")
    print(f"  timestep           {g['dt']:.4e} s "
          f"(Courant {g['dt_courant']:.3e}, plasma {g['dt_plasma']:.3e})")
    print(f"  grid               {g['n_cells']}^{pcfg.dim}, "
          f"{g['steps']} steps, {g['t_end']:.3e} s")
    print(f"  ion drift          {pcfg.v_drift:.4e} m/s "
          f"(gamma - 1 = {pcfg.gamma_ion - 1:.3e})")
    print(f"  electron drift     {pcfg.v_electron_drift:.4e} m/s "
          f"({pcfg.electron_drift_ratio:.4g} x ion)")
    print(f"  estimated cost     {pcost['hours']:.2f} GPU-hours, "
          f"{pcost['gpu_memory_GB']:.1f} GB of particles")
    print(f"  written to         {pdir}/")
    for name in sorted(written):
        print(f"                       {os.path.relpath(written[name], pdir)}")
    print(f"\n  build with:\n    {build_command(pcfg, gpu=args.gpu)}")

    if pcfg.notes:
        print("\n  configuration audit:")
        for w in pcfg.notes:
            print(f"    ! {w}")

    banner("Next")
    print(f"""Both directories are complete and need no further editing.

  Idefix:    cd {idir} && <the cmake line above> && make -j 8 && ./idefix
  PIConGPU:  pic-create $PIC_EXAMPLES/KelvinHelmholtz ~/picInputs/hviEMP
             cp -r {pdir}/include {pdir}/etc ~/picInputs/hviEMP/
             cd ~/picInputs/hviEMP && <the pic-build line above>

Each directory has a README.txt with the full recipe, the code-unit
conversions, and what to look for in the output.

Installation for both: docs/INSTALLING_SOLVERS.md sections 3 and 4.
Read the configuration audit above before trusting any number that comes out.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
