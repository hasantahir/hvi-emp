"""Example 15 -- ejecta particles, fragments and plasma, from real MD.

The look these two reference videos have in common is that they are made of
**particles**, not fields: an atomistic dust cloud in one, a cloud of discrete
fragments in the other. This example produces both from one LAMMPS run.

    python examples/15_md_ejecta_fragments.py --run          # run MD first
    python examples/15_md_ejecta_fragments.py --dump impact.dump

    pvpython md_scenes/scene_thermal.py      # the dust cloud + plasma
    pvpython md_scenes/scene_fragments.py    # the debris cloud

What is simulated and what is not
---------------------------------
* **Positions, velocities and temperature are simulated.** They come out of
  the MD integration.
* **Ionisation is inferred.** Classical MD has no electrons; an EAM potential
  cannot ionise anything. Saha is applied afterwards to the MD density and
  temperature to ask what charge state *would* obtain in LTE. Every output
  file carries that caveat.
* **Fragments are a clustering choice.** "Connected" means "within a cutoff",
  and the cutoff changes the answer -- `--sensitivity` shows by how much.

The scale, stated plainly
-------------------------
MD reaches nm-scale projectiles over ~100 ps. A 1 pg micrometeoroid is ~10^13
atoms and its plume evolves over microseconds, so this is six orders of
magnitude below the case the rest of the framework models. It is a real
picture of a small impact, not a small picture of a real one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from hvi_emp.solvers.fragments import cutoff_sensitivity, size_distribution
from hvi_emp.solvers.lammps_postprocess import (attach_fragments,
                                                analyse_frame, convert_dump,
                                                read_lammps_dump)
from hvi_emp.viz.paraview_scene import write_particle_scene_script


def run_md(path: str, velocity: float, radius: float, cells: int,
           picoseconds: float) -> str:
    from hvi_emp.solvers.lammps_stage1 import LammpsImpactConfig, run_impact

    cfg = LammpsImpactConfig(
        material="W", velocity=velocity, projectile_radius=radius,
        target_cells=(cells, cells, int(cells * 0.7)),
        t_equilibrate=2e-13, t_production=picoseconds * 1e-12,
        timestep=2e-16, dump_every=1500, dump_path=path)
    print(f"running MD: W sphere r={radius*1e9:.1f} nm at "
          f"{velocity/1e3:.1f} km/s, {picoseconds:.1f} ps ...")
    run_impact(cfg, screen=False)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default="impact.dump")
    ap.add_argument("--out", default="md_scenes")
    ap.add_argument("--run", action="store_true",
                    help="run LAMMPS first (needs the lammps module)")
    ap.add_argument("--velocity", type=float, default=9e3)
    ap.add_argument("--radius", type=float, default=0.8e-9)
    ap.add_argument("--cells", type=int, default=26)
    ap.add_argument("--ps", type=float, default=6.0)
    ap.add_argument("--surface", type=float, default=None,
                    help="free-surface height [m]; estimated if omitted")
    ap.add_argument("--cutoff", type=float, default=None,
                    help="fragment cutoff [m]; from the lattice if omitted")
    ap.add_argument("--sensitivity", action="store_true",
                    help="re-cluster at several cutoffs and report the spread")
    ap.add_argument("--preset", default="talk",
                    choices=("talk", "paper", "diagnostic"))
    ap.add_argument("--movie", action="store_true")
    args = ap.parse_args(argv)

    if args.run:
        run_md(args.dump, args.velocity, args.radius, args.cells, args.ps)
    if not os.path.isfile(args.dump):
        print(f"no dump at {args.dump}. Run with --run, or point --dump at "
              f"a LAMMPS trajectory.")
        return 2

    # ---------------------------------------------------------------- data
    print(f"\nconverting {args.dump} ...")
    res = convert_dump(args.dump, args.out, material="W", fragments=True,
                       surface=args.surface, cutoff=args.cutoff,
                       name="md", verbose=True)

    print("\nper frame:")
    print(f"  {'t [ps]':>9}{'T_max [eV]':>12}{'Zbar_max':>10}"
          f"{'ionised %':>11}{'ejecta':>8}{'fragments':>11}")
    for pk in res["peaks"]:
        print(f"  {pk['t']*1e12:>9.3f}{pk['T_eV_max']:>12.3f}"
              f"{pk['Zbar_max']:>10.3f}"
              f"{100*pk.get('ionised_fraction', 0.0):>11.2f}"
              f"{pk.get('n_ejecta', 0):>8}"
              f"{pk.get('n_fragments', 0):>11}")

    # ------------------------------------------------------------ analysis
    last = list(read_lammps_dump(args.dump))[-1]
    st = analyse_frame(last, material="W")
    attach_fragments(st, material="W", surface=args.surface,
                     cutoff=args.cutoff)
    print()
    for n in st.notes:
        print(f"  - {n}")

    frags = st.fragments
    if frags is not None and frags.n_fragments:
        sd = size_distribution(frags)
        print("\nfragment size distribution (largest first):")
        print(f"  {'atoms':>8}{'mass [kg]':>13}{'N(>=m)':>9}"
              f"{'cum. mass %':>13}")
        for i in list(range(min(5, frags.n_fragments))):
            print(f"  {int(frags.sizes[i]):>8}{frags.masses[i]:>13.3e}"
                  f"{int(sd['n_ge'][i]):>9}"
                  f"{100*sd['mass_fraction_ge'][i]:>13.1f}")
        if frags.n_fragments > 5:
            print(f"  ... and {frags.n_fragments - 5} smaller")

    if args.sensitivity:
        pos = last.positions() * 1e-10
        vel = last.velocities() * 1e2
        from hvi_emp.materials import get_material
        print("\ncutoff sensitivity -- the one free parameter:")
        print(f"  {'factor':>8}{'cutoff [A]':>13}{'fragments':>11}"
              f"{'largest':>9}{'singletons':>12}")
        for r in cutoff_sensitivity(pos, vel, get_material("W").m_atom,
                                    mask=st.atom_is_ejecta):
            print(f"  {r['factor']:>8.1f}{r['cutoff']*1e10:>13.2f}"
                  f"{r['n_fragments']:>11}{r['largest_atoms']:>9}"
                  f"{r['singletons']:>12}")
        print("  If your conclusion moves across this table, say so.")

    # -------------------------------------------------------------- scenes
    print()
    scenes = [
        ("scene_thermal.py", "T_eV", res.get("grid_pvd"),
         "atomistic dust cloud, plasma glowing behind it"),
        ("scene_fragments.py", "fragment_id", None,
         "discrete debris, one colour per fragment"),
        ("scene_plasma.py", "log10_electron_density", res.get("grid_pvd"),
         "which of the debris is actually ionised"),
    ]
    for name, colour, grid, what in scenes:
        path = os.path.join(args.out, name)
        write_particle_scene_script(
            res["atoms_pvd"], path, grid_pvd=grid, colour_by=colour,
            preset=args.preset,
            movie_path=(os.path.join(args.out, "frames",
                                     name.replace(".py", ".png"))
                        if args.movie else None))
        print(f"  {path:<34} {what}")

    print("\n  pvpython " + os.path.join(args.out, "scene_thermal.py"))
    print("\nPositions and temperature are simulated. IONISATION IS "
          "INFERRED -- classical")
    print("MD has no electrons. Fragments are a clustering choice; "
          "--sensitivity shows")
    print("how much the cutoff matters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
