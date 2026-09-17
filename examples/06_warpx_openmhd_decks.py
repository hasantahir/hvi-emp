"""Example 6 -- generate WarpX and OpenMHD decks from a framework scenario.

Runs the reduced chain for the Close et al. (2013) experimental configuration,
then emits:

  solver_decks/warpx_emp.py    PICMI script  (python warpx_emp.py, needs pywarpx)
  solver_decks/warpx_inputs    native input  (warpx.2d warpx_inputs)
  solver_decks/model.f90       OpenMHD diamagnetic-cavity initial condition
  solver_decks/OPENMHD_RUN.md  build/run/SI-conversion instructions

Every deck is initialised from the *computed* Stage-2 plume state, so the
three levels of description (reduced closure, PIC, MHD) are simulating the
same physical impact.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import numpy as np

from hvi_emp import run_scenario
from hvi_emp.solvers import availability
from hvi_emp.solvers.openmhd_stage2 import (OpenMHDConfig, cavity_estimates,
                                            write_model_f90, write_run_notes)
from hvi_emp.solvers.warpx_stage3 import (WarpXEMPConfig, write_picmi_script,
                                          write_warpx_inputs)

OUT = Path(__file__).resolve().parents[1] / "solver_decks"
OUT.mkdir(exist_ok=True)

print("solver availability:", availability(), "\n")

# The Close et al. (2013) configuration: Fe microparticle onto biased W.
sc = run_scenario("Fe", "W", mass=1e-16, velocity=50e3, r_sensor=0.30)
print(sc.impact.summary(), "\n")

# --- WarpX: the affordable epoch (peak f_pe at the antenna band) -----------
cfg_late = WarpXEMPConfig.from_expansion(sc.expansion, at_frequency=916e6,
                                         case="cold")
g = cfg_late.grid()
print(f"WarpX (916 MHz epoch): n_e0 = {cfg_late.n_e0:.2e} m^-3, "
      f"R = {cfg_late.R_plume*1e3:.2f} mm")
print(f"  grid {g['n_cells']}^2, dt = {g['dt']:.2e} s, "
      f"{g['n_steps']:,} steps, Debye resolved: {g['debye_resolved']}")
write_picmi_script(cfg_late, str(OUT / "warpx_emp.py"))
write_warpx_inputs(cfg_late, str(OUT / "warpx_inputs"))

# The transition epoch, for the honest cost statement:
cfg_tr = WarpXEMPConfig.from_expansion(sc.expansion)
gt = cfg_tr.grid()
print(f"  (transition epoch would need dt = {gt['dt']:.1e} s to resolve "
      f"f_pe = {cfg_tr.omega_pe/2/np.pi:.1e} Hz -- the scale-separation "
      "wall Fletcher & Close also hit)\n")

# --- OpenMHD: diamagnetic cavity for an orbital impact ---------------------
sc2 = run_scenario("Olivine", "Al", mass=1e-9, velocity=59e3, t_end=2e-4)
cfg_mhd = OpenMHDConfig.from_expansion(sc2.expansion, B0=3e-5)
write_model_f90(cfg_mhd, str(OUT / "model.f90"))
write_run_notes(cfg_mhd, str(OUT / "OPENMHD_RUN.md"))
E_kin = 0.5 * sc2.impact.m_vapour * sc2.expansion.v_z[-1] ** 2
est = cavity_estimates(E_kin, 3e-5, float(sc2.expansion.v_z[-1]))
print("OpenMHD (1 ug Perseid, LEO field):")
print(f"  predicted cavity radius {est['R_cavity']:.2f} m, "
      f"formation {est['t_formation']:.2e} s, "
      f"characteristic f {est['f_characteristic']:.0f} Hz")

print(f"\ndecks written to {OUT}/")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:<18} {f.stat().st_size:>7,} bytes")
