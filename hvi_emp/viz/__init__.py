"""Volumetric visualisation output (VTK ImageData / `.vti`).

Writes the impact, plume and condensate ("smoke") as a time series of `.vti`
files plus a `.pvd` collection, openable directly in ParaView or VisIt.

    from hvi_emp import run_scenario
    from hvi_emp.viz import write_impact_series

    sc = run_scenario("Fe", "Al", mass=1e-9, velocity=30e3)
    write_impact_series(sc, "vti_out", n_frames=80)
    # then: paraview vti_out/impact.pvd

No dependencies beyond NumPy: the VTK XML format is written directly, with
raw appended binary payloads (so the files are compact and load fast).

**What is solved and what is drawn.**  Read `docs/VISUALISATION.md` before
presenting these images.  The plume fields (density, temperature, ionisation,
velocity) come from the Stage-2 ODE and are as good as the model.  The shock
front in the target, the crater and the condensate cloud are *rendered from
the model's own scaling laws*, not from a solved continuum field -- the
reduced chain does not resolve the solid.  Each field is tagged in
`FIELD_PROVENANCE` so the distinction survives into the output.

**Pick a scene.** One domain cannot show both the crater and the plume: on a
plume-sized box the crater is 0.013 cells across.  `write_scene(sc, dir,
scene="impact")` sizes everything for the impact site; ``scene="plume"`` is
the old behaviour.  `SCENES` has the arithmetic.  `write_scene_for` then
writes a ParaView script that builds the view and saves a `.pvsm`.
"""

from .paraview_scene import write_scene_for, write_scene_script
from .vti import (FIELD_PROVENANCE, IMPACT_FIELDS, SCENES, ImpactVolume,
                  scene_kwargs, write_impact_series, write_pvd, write_scene,
                  write_vti)

__all__ = ["write_vti", "write_pvd", "write_impact_series", "ImpactVolume",
           "FIELD_PROVENANCE", "write_scene", "scene_kwargs", "SCENES",
           "IMPACT_FIELDS", "write_scene_script", "write_scene_for"]
