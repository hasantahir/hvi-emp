// Stub for scripts/m2c_harness/tillotson_selftest.cpp only -- just enough
// of M2C/boost for VarFcnTillot.h to compile on its own. Not M2C code.
#pragma once
#include <utility>
#include <cstdint>
namespace boost { using uintmax_t = std::uintmax_t; namespace math { namespace tools {
template<class F,class T> std::pair<double,double> toms748_solve(F f,double a,double b,double fa,double fb,T tol,uintmax_t& maxit){
  for(uintmax_t i=0;i<maxit && !tol(a,b);i++){double m=0.5*(a+b),fm=f(m); if((fm<0)==(fa<0)){a=m;fa=fm;}else{b=m;fb=fm;}} return {a,b};}
}}}
