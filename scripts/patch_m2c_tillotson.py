#!/usr/bin/env python3
"""Patch the aborts out of M2C's Tillotson path, then rebuild M2C.

    python scripts/patch_m2c_tillotson.py ~/src/m2c          # patch
    python scripts/patch_m2c_tillotson.py ~/src/m2c --check  # report only
    python scripts/patch_m2c_tillotson.py ~/src/m2c --selftest
    cd ~/src/m2c/build && make -j 16

Every fix below was found by running M2C itself (built from kevinwgy/m2c,
Sep 2026) on the hvi_emp Al->Al 32 km/s deck, each one the next thing to
stop it. Edits VarFcnTillot.h and ExactRiemannSolverBase.cpp.

Signal 6 (assert):
 1. start-up: the constructor builds the cold curve e_cold(rho) before it
    sets elat = eCV - eIV, then divides by it.
 2. shock past 2 rho0: on-demand extension asks RK45 for less than one step.
 3. plume below ~rho0/2: RK45 marching backwards takes the whole interval
    as its first step; and below rho_IV the curve stalls on e = eCV. Now
    RK4 in 0.2 % steps, held at eCV below the crossing (cohesive plateau).
 7. t ~ 72 ns: the exact Riemann solver's rarefaction stages test
    rho<=0 || c^2<0, which NaN passes; NaN reaches Tillotson's assert.

exit(-1), i.e. "Exit 255" with the message from one rank only:
 4. step 2: p -> e below rho_IV. Case 3 (e<eCV) and Case 2 (e>eCV) do not
    meet -- for Al at 0.44 rho0, -32 GPa vs +8.3 GPa -- so a reconstructed
    or Riemann pressure in between has no e. Now e = eCV.
 5,6. t ~ 28 ns: the Riemann solver evaluates the EOS at trial densities
    <= 0 before rejecting them; every other M2C EOS returns a number,
    Tillotson exited. Now it returns, and the caller rejects as designed.
 6b. t ~ 81 ns: the blend inverse finds its root on e_IV to round-off with
    the wrong sign and exits for "incorrect inputs". Now takes the endpoint.

--selftest compiles M2C's own VarFcnTillot.h alone (g++, ~10 s) and drives
the cold curve the way a run does: unpatched it aborts as M2C did.

Idempotent per edit (a source patched by an earlier version gets only what
it lacks); keeps *.orig; refuses, changing nothing, if the source differs.
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

# --- 4. p -> e below rho_IV: no e exists for a band of pressures ------------
# Below rho_IV the forward EOS is Case 3 for e < eCV and Case 2 above, and the
# two do not meet: for Al at 0.44 rho0, P3(eCV-) = -32 GPa, P2(eCV+) = +8.3
# GPa. Any (rho, p) with p in between -- a reconstructed or Riemann state, not
# one the update itself produces -- has no inverse, and M2C exits(-1). Under
# MPI only the failing rank prints, so the run just stops with exit code 255.
OLD_4 = r"""      e = GetInternalEnergyPerUnitMass2(rho,p);
      if(e<eCV) {
        fprintf(stdout,"\033[0;31m*** Error: VarFcnTillot::GetInternalEnergyPerUnitMass failed for "
                       "rho = %e, p = %e.\033[0m\n", rho, p);
        exit(-1);
      }"""
NEW_4 = r"""      e = GetInternalEnergyPerUnitMass2(rho,p);
      if(e<eCV) // hvi_emp patch: p lies in the gap between Case 3 at eCV- and
        e = eCV; // Case 2 at eCV+, where no e gives it; take the discontinuity"""

# --- 6b. blend inverse: a root sitting on e_IV within round-off -------------
# GetInternalEnergyPerUnitMass sends p to the Case 1|2 blend when the Case 1
# root is just above eIV. The blend equals Case 1 at eIV, so f(eIV) is zero
# to round-off and can come out with the wrong sign (-1.3e-16 at t = 81 ns),
# and M2C exits(-1) for "incorrect inputs". Take the endpoint nearer zero.
OLD_8 = r"""  if(f_low*f_high>0) {
    fprintf(stdout,"\033[0;31m*** Error: VarFcnTillot::GetInternalEnergyPerUnitMass12 called w. incorrect inputs."
                   " rho = %e, p = %e. f(%e) = %e, f(%e) = %e.\033[0m\n",
            rho, p, e_low, f_low, e_high, f_high);
    exit(-1);
  }"""
NEW_8 = r"""  if(f_low*f_high>0) // hvi_emp patch: no sign change -- the root is at an
    return fabs(f_low)<=fabs(f_high) ? e_low : e_high; // end, to round-off"""

# --- 5. non-positive trial densities are the caller's to reject -------------
# ExactRiemannSolverBase::Rarefaction_OneStepRK4 evaluates e(rho, p) and c^2 at
# each RK stage *before* testing rho > 0, and treats rho <= 0 as "step too
# big, retry smaller". Every other M2C EOS returns a number there; Tillotson
# exits(-1). Seen at t = 28 ns in the Al->Al 32 km/s run.
OLD_5 = r"""    if(rho<=0.0) {
      fprintf(stdout,"\033[0;31m*** Error: VarFcnTillot::GetCaseWithRhoE detected non-positive rho (%e).\033[0m\n",
              rho);
      exit(-1);
    }"""
NEW_5 = r"""    if(!(rho>0.0)) // hvi_emp patch: trial states from the exact Riemann solver
      return 0;    // (rho <= 0 or NaN); it rejects them itself after the call"""
OLD_6 = r"""  if(rho<=0.0) {
    fprintf(stdout,"\033[0;31m*** Error: VarFcnTillot::GetInternalEnergyPerUnitMass detected negative rho (%e).\033[0m\n",
            rho);
    exit(-1);
  }"""
NEW_6 = r"""  if(!(rho>0.0)) // hvi_emp patch: see GetCaseWithRhoE -- the caller rejects it
    return 0.0;"""

EDITS = [("elat before the cold curve", OLD_1, NEW_1),
         ("first step upwards", OLD_2, NEW_2),
         ("downward cold curve", OLD_3, NEW_3),
         ("p -> e gap below rho_IV", OLD_4, NEW_4),
         ("trial rho <= 0 (case)", OLD_5, NEW_5),
         ("trial rho <= 0 (energy)", OLD_6, NEW_6),
         ("blend root at an endpoint", OLD_8, NEW_8)]

# --- 7. ExactRiemannSolverBase.cpp: reject NaN trial states too ------------
# Each RK stage of the rarefaction integration tests `rho_k<=0 || c_k^2<0`.
# Both comparisons are false for NaN, so a NaN stage (c^2 -> 0 or inf
# upstream) is carried on and reaches the EOS -- in Tillotson, an assert
# (rho >= rhoIV) and signal 6. Seen at t = 72 ns. NaN-safe: !(x>0), !(x>=0).
RIEMANN_FILE = "ExactRiemannSolverBase.cpp"
RIEMANN_RE = re.compile(r"if ?\(rho_(\d)<=0 \|\| c_\1_square<0\) ?\{")
RIEMANN_NEW = (r"if(!(rho_\1>0) || !(c_\1_square>=0)) { "
               r"// hvi_emp patch: NaN-safe")
RIEMANN_COUNT = 10

# A line of each NEW block that only the patch writes: lets a source patched
# by an earlier version of this script receive just the edits it lacks.
_SIGNATURE = {
    "elat before the cold curve": "elat = eCV-eIV; // hvi_emp patch",
    "first step upwards": "rhomax - rho_plus[mysize-1]); // hvi_emp patch",
    "downward cold curve": "Classical RK4 in 0.2% density steps",
    "p -> e gap below rho_IV": "e = eCV; // Case 2 at eCV+",
    "trial rho <= 0 (case)": "(rho <= 0 or NaN); it rejects them itself",
    "trial rho <= 0 (energy)": "see GetCaseWithRhoE -- the caller rejects it",
    "blend root at an endpoint": "no sign change -- the root is at an",
}


def find_header(root: Path) -> Path:
    for cand in (root, root.parent, root / "src"):
        p = cand / "VarFcnTillot.h"
        if p.is_file():
            return p
    raise FileNotFoundError(f"VarFcnTillot.h not under {root} (give the M2C "
                            "source directory, or its build/ directory)")


def missing_edits(text: str) -> list:
    """Edits to VarFcnTillot.h not yet applied to `text`."""
    return [name for name, _, _ in EDITS if _SIGNATURE[name] not in text]


def is_patched(text: str) -> bool:
    """VarFcnTillot.h text carries every edit (see also source_patched)."""
    return not missing_edits(text)


def riemann_patched(text: str) -> bool:
    return not RIEMANN_RE.search(text) and "hvi_emp patch: NaN-safe" in text


def source_patched(src: Path) -> bool:
    """Both files of the M2C source tree `src` carry the patch."""
    return (is_patched((src / "VarFcnTillot.h").read_text())
            and riemann_patched((src / RIEMANN_FILE).read_text()))


def patch_riemann_text(text: str) -> str:
    if riemann_patched(text):
        return text
    n = len(RIEMANN_RE.findall(text))
    if n != RIEMANN_COUNT:
        raise ValueError(f"'{RIEMANN_FILE}': expected {RIEMANN_COUNT} "
                         f"rarefaction-stage checks, found {n}. Your M2C "
                         "differs from the version this patch was written "
                         "for (kevinwgy/m2c, Sep 2026).")
    return RIEMANN_RE.sub(RIEMANN_NEW, text)


def _block(old: str) -> "re.Pattern[str]":
    """Match `old` line by line, ignoring trailing whitespace (M2C has some)."""
    lines = [re.escape(ln.rstrip()) + r"[ \t]*" for ln in old.split("\n")]
    return re.compile("\n".join(lines))


def patch_text(text: str) -> str:
    todo = missing_edits(text)
    for name, old, new in EDITS:
        if name not in todo:
            continue
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


def _write(path: Path, new: str) -> None:
    orig = path.with_name(path.name + ".orig")
    if not orig.exists():           # keep the stock file, not a half-patched one
        shutil.copy2(path, orig)
    path.write_text(new)


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
    src = hdr.parent
    rsv = src / RIEMANN_FILE
    if not rsv.is_file():
        print(f"{rsv} not found")
        return 2

    if args.selftest:
        return selftest(hdr, args.cxx)
    htext, rtext = hdr.read_text(), rsv.read_text()
    todo = missing_edits(htext)
    rdo = not riemann_patched(rtext)
    if args.check:
        if not todo and not rdo:
            print(f"{src}: patched")
            return 0
        print(f"{src}: NOT patched -- missing: "
              + ", ".join(todo + ([RIEMANN_FILE] if rdo else [])))
        return 1
    if not todo and not rdo:
        print(f"{src}: already patched; nothing to do")
        return 0
    try:
        hnew = patch_text(htext)
        rnew = patch_riemann_text(rtext)
    except ValueError as e:
        print(f"refusing to patch {src} (nothing changed):\n  {e}")
        return 1
    if todo:
        _write(hdr, hnew)
        print(f"patched {hdr.name}: {', '.join(todo)}")
    if rdo:
        _write(rsv, rnew)
        print(f"patched {RIEMANN_FILE}: NaN-safe rarefaction stages")
    print("stock files kept as *.orig")
    build = src / "build"
    print("Rebuild M2C:")
    print(f"  cd {build if build.is_dir() else '<your M2C build dir>'} "
          f"&& make -j 16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
