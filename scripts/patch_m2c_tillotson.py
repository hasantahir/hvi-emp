#!/usr/bin/env python3
"""Patch the aborts out of M2C's Tillotson cold curve, then rebuild M2C.

    python scripts/patch_m2c_tillotson.py ~/src/m2c          # patch
    python scripts/patch_m2c_tillotson.py ~/src/m2c --check  # report only
    python scripts/patch_m2c_tillotson.py ~/src/m2c --selftest
    cd ~/src/m2c/build && make -j 16

--selftest compiles M2C's own VarFcnTillot.h on its own (g++, ~10 s) and
drives it the way the impact run does. Unpatched, it aborts exactly as M2C
did; patched, it prints PASS.

With `TemperatureDependsOnDensity = Yes` (the hvi_emp deck uses it, because
`No` puts material at rest near -5300 K), M2C integrates the cold curve
e_cold(rho) for T = T0 + (e - e_cold(rho))/cv, and aborts with signal 6 in
three ways. All three are in VarFcnTillot.h, reproduced against M2C's own
header (tests/test_m2c_patch.py compiles it when g++ is available).

1. At start-up, on every rank (main -> constructor -> assert).
   The constructor integrates the cold curve down to rho0/2 *before* it sets
   `elat = eCV - eIV`. For Al the curve passes e_IV (3 MJ/kg) above rho0/2,
   into the partial-vaporisation blend, which divides by `elat` -- still
   uninitialised. The step goes non-finite; `assert(!err)` aborts.
   Fix: set `elat` first.

2. Mid-run, once the shock compresses a cell past 2 rho0 (Al at 32 km/s
   reaches ~2.5 rho0). The curve is extended on demand, starting with the
   previous step size; asked for a density less than one step past its end,
   runge_kutta_45 refuses (N <= 0) and the assert fires.
   Fix: cap the first step at the requested extension.

3. Mid-run, once any cell expands well below rho0/2 -- i.e. as soon as there
   is a plume. runge_kutta_45's "do not step past tf" clamp only works
   marching forward, so going down its first step is the whole interval
   (to the density floor) and goes non-finite. Had it not, it would stall
   anyway: below rho_IV e_cold rises to e_CV, where the EOS alternates
   between Case 3 (p < 0) and Case 2 (p > 0) and the ODE has no solution
   off the line e = e_CV.
   Fix: integrate downwards with classical RK4 in 0.2 % density steps, stop
   where e_cold reaches e_CV, and hold it there below that density -- the
   cohesive plateau (e_CV = 13.9 MJ/kg for Al; measured cohesive energy
   12.1 MJ/kg). Hot vapour then gets T = T0 + (e - e_CV)/cv.

Idempotent; keeps VarFcnTillot.h.orig; refuses if the source does not match.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

MARK = "hvi_emp patch"

# --- 1. elat before the cold curve -----------------------------------------
OLD_1 = """  if(temperature_depends_on_density) {
    use_cp = false; //we use e-T relation to determine the thermal vibration energy
    SetupColdEnergyTrajectory(); //!< fill ecold_plus, rho_plus, ecold_minus, rho_minus"""

NEW_1 = """  elat = eCV-eIV; // hvi_emp patch: the cold-curve integration below uses elat
                  // (Case 1|2 blend), so it must be set before it runs

  if(temperature_depends_on_density) {
    use_cp = false; //we use e-T relation to determine the thermal vibration energy
    SetupColdEnergyTrajectory(); //!< fill ecold_plus, rho_plus, ecold_minus, rho_minus"""

# --- 2. upwards: first step no longer than the extension -------------------
OLD_2 = """  } else
    drho0 = 1e-3*rho0;
"""
NEW_2 = """  } else
    drho0 = 1e-3*rho0;
  drho0 = std::min(drho0, rhomax - rho_plus[mysize-1]); // hvi_emp patch: else RK45 returns -1
"""

# --- 3. downwards: RK4 in relative steps, plateau at eCV --------------------
OLD_3 = """VarFcnTillot::ExtendColdEnergyTrajectoryDownwards(double rhomin)
{
  if(rhomin>=rho_minus.back())
    return;

  assert(rhomin>0.0);

  // sets up the ODE
  auto fun = [&](double *ec, double rho, double *dec_drho) {
    *dec_drho = GetPressure(rho, *ec)/(rho*rho); return;};

  // determine the initial step size (<0 in this case)
  double drho0;
  int mysize = rho_minus.size();
  if(mysize>2) {
    drho0 = rho_minus[mysize-2] - rho_minus[mysize-3]; //rho_minus[mysize-1]-rho_minus[mysize-2] may not
                                                       //be the desired step size, as the last element
                                                       //of rho_minus is pre-specified/fixed.
  } else
    drho0 = -1e-3*rho0;

  // integration
  double ecold_min;
  std::vector<double> rho_new, ecold_new;
  int err = MathTools::runge_kutta_45(fun, 1, rho_minus[mysize-1], &ecold_minus[mysize-1], drho0, rhomin,
                                      &ecold_min, [](double*){return false;},
                                      tol/100.0, Nmax-mysize, &rho_new, &ecold_new); //smaller tol, because we will interpolate
  assert(!err);

  int new_size = rho_new.size();
  assert(new_size == (int)ecold_new.size());

  rho_minus.reserve(mysize + new_size - 1);
  ecold_minus.reserve(mysize + new_size - 1);
  for(int i=1; i<new_size; i++) {
    rho_minus.push_back(rho_new[i]);
    ecold_minus.push_back(ecold_new[i]);
  }
}"""

NEW_3 = """VarFcnTillot::ExtendColdEnergyTrajectoryDownwards(double rhomin)
{
  // hvi_emp patch. Replaces a runge_kutta_45 call whose first step, marching
  // backwards, is the whole interval (its "t+dt>tf" clamp is forward-only),
  // and which stalls where e_cold reaches eCV: there the EOS alternates
  // between Case 3 (p<0) and Case 2 (p>0), and the ODE has no solution off
  // e = eCV. Classical RK4 in 0.2% density steps; below the eCV crossing
  // e_cold is held at eCV (cohesive plateau).
  if(rhomin>=rho_minus.back())
    return;

  assert(rhomin>0.0);

  double r  = rho_minus.back();
  double ec = ecold_minus.back();

  if(ec>=eCV) { // already on the plateau
    rho_minus.push_back(rhomin);
    ecold_minus.push_back(ec);
    return;
  }

  auto f = [&](double rr, double ee) {return GetPressure(rr, ee)/(rr*rr);};
  const double frac = 2.0e-3;

  while(r>rhomin) {
    double h  = -std::min(frac*r, r-rhomin);
    double k1 = f(r, ec);
    double k2 = f(r+0.5*h, ec+0.5*h*k1);
    double k3 = f(r+0.5*h, ec+0.5*h*k2);
    double k4 = f(r+h, ec+h*k3);
    double en = ec + h*(k1+2.0*k2+2.0*k3+k4)/6.0;
    if(!std::isfinite(en)) {
      fprintf(stdout,"\\033[0;31m*** Error: Tillotson cold curve non-finite at rho = %e "
                     "(e_cold = %e).\\033[0m\\n", r+h, ec);
      exit(-1);
    }
    if(en>=eCV) { // crossing: interpolate it, then hold
      double rx = r + h*(eCV-ec)/(en-ec);
      if(rx<r) {
        rho_minus.push_back(rx);
        ecold_minus.push_back(eCV);
      } else
        ecold_minus.back() = eCV;
      if(rhomin<rho_minus.back()) {
        rho_minus.push_back(rhomin);
        ecold_minus.push_back(eCV);
      }
      if(verbose>=1)
        fprintf(stdout,"Note: Tillotson cold curve reaches eCV (%e) at rho = %e; held constant below.\\n",
                eCV, rx);
      return;
    }
    r += h;
    ec = en;
    rho_minus.push_back(r);
    ecold_minus.push_back(ec);
  }
}"""

EDITS = [("elat before the cold curve", OLD_1, NEW_1),
         ("first step upwards", OLD_2, NEW_2),
         ("downward cold curve", OLD_3, NEW_3)]


def find_header(root: Path) -> Path:
    for cand in (root, root.parent, root / "src"):
        p = cand / "VarFcnTillot.h"
        if p.is_file():
            return p
    raise FileNotFoundError(f"VarFcnTillot.h not under {root} (give the M2C "
                            "source directory, or its build/ directory)")


def is_patched(text: str) -> bool:
    return MARK in text


def _block(old: str) -> "re.Pattern[str]":
    """Match `old` line by line, ignoring trailing whitespace (M2C has some)."""
    lines = [re.escape(ln.rstrip()) + r"[ \t]*" for ln in old.split("\n")]
    return re.compile("\n".join(lines))


def patch_text(text: str) -> str:
    if is_patched(text):
        return text
    for name, old, new in EDITS:
        pat = _block(old)
        n = len(pat.findall(text))
        if n != 1:
            raise ValueError(f"'{name}': expected the M2C source to contain "
                             f"this block once, found {n}. Your M2C differs "
                             "from the version this patch was written for "
                             "(kevinwgy/m2c, Sep 2026).")
        text = pat.sub(lambda _m: new, text, count=1)
    return text


HARNESS = Path(__file__).resolve().parent / "m2c_harness"


def selftest(hdr: Path, cxx: str = "g++") -> int:
    """Compile and run the harness against the header in `hdr`'s tree."""
    import subprocess
    import tempfile
    src = hdr.parent
    exe = Path(tempfile.mkdtemp()) / "tillotson_selftest"
    cmd = [cxx, "-O2", "-std=c++17", "-w",
           f"-I{HARNESS / 'stub'}", f"-I{src}", f"-I{src / 'MathTools'}",
           str(HARNESS / "tillotson_selftest.cpp"),
           str(src / "MathTools" / "polynomial_equations.cpp"), "-o", str(exe)]
    print("building:", " ".join(cmd))
    b = subprocess.run(cmd, capture_output=True, text=True)
    if b.returncode:
        print(b.stderr[-3000:])
        print("selftest could not be built (see above)")
        return 2
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    print(r.stdout, end="")
    if r.returncode == 0:
        return 0
    print(r.stderr[-2000:], end="")
    if r.returncode in (-6, 134):
        print("FAIL: aborted (signal 6) -- the crash M2C showed. "
              + ("Patch, then rebuild M2C." if not is_patched(hdr.read_text())
                 else "The patch did not cover this; please report it."))
    else:
        print(f"FAIL: exit code {r.returncode}")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("m2c", nargs="?", help="M2C source (or build) directory; "
                    "default: $M2C_HOME")
    ap.add_argument("--check", action="store_true",
                    help="say whether the source is patched; change nothing")
    ap.add_argument("--selftest", action="store_true",
                    help="compile M2C's VarFcnTillot.h alone and exercise it; "
                         "changes nothing")
    ap.add_argument("--cxx", default="g++")
    args = ap.parse_args(argv)

    root = args.m2c or os.environ.get("M2C_HOME")
    if not root:
        print("give the M2C source directory, or set M2C_HOME")
        return 2
    try:
        hdr = find_header(Path(root).expanduser().resolve())
    except FileNotFoundError as e:
        print(e)
        return 2

    text = hdr.read_text()
    if args.selftest:
        return selftest(hdr, args.cxx)
    if args.check:
        print(f"{hdr}: {'patched' if is_patched(text) else 'NOT patched'}")
        return 0 if is_patched(text) else 1
    if is_patched(text):
        print(f"{hdr}: already patched; nothing to do")
        return 0
    try:
        new = patch_text(text)
    except ValueError as e:
        print(f"refusing to patch {hdr}:\n  {e}")
        return 1
    shutil.copy2(hdr, hdr.with_suffix(".h.orig"))
    hdr.write_text(new)
    build = hdr.parent / "build"
    print(f"patched {hdr}  (original kept as {hdr.name}.orig)")
    print("Rebuild M2C -- the header is compiled into Main.cpp:")
    print(f"  cd {build if build.is_dir() else '<your M2C build dir>'} "
          f"&& make -j 16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
