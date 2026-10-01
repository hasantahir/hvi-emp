// Stub for scripts/m2c_harness/tillotson_selftest.cpp only -- just enough
// of M2C/boost for VarFcnTillot.h to compile on its own. Not M2C code.
#include <cassert>
#pragma once
#include <cstdio>
#include <cstdlib>
extern int verbose;
struct ClassAssigner;
struct TillotsonModelData {
  double rho0,e0,a,b,A,B,alpha,beta,rhoIV,eIV,eCV,cv;
  enum YesNo {NO=0,YES=1} temperature_depends_on_density; double T0,cp,h0;
};
struct MaterialModelData {
  enum Eos {STIFFENED_GAS=0, NOBLE_ABEL_STIFFENED_GAS=1, MIE_GRUNEISEN=2, EXTENDED_MIE_GRUNEISEN=3, TILLOTSON=4} eos;
  double rhomin,pmin,rhomax,pmax,failsafe_density;
  TillotsonModelData tillotModel;
};
