// Exercise M2C's VarFcnTillot.h (the real header, from your M2C source) the
// way a hypervelocity-impact run does: Al Tillotson with
// TemperatureDependsOnDensity = Yes, constructed, then temperatures asked for
// at densities from 1e-6 to 5 rho0 in random order. Stubs replace IoData.h,
// Vector3D.h and boost's root finder so it builds without the rest of M2C.
// Built and run by:  python scripts/patch_m2c_tillotson.py <m2c> --selftest
#include <VarFcnTillot.h>
#include <random>
int verbose = 0;
int main() {
  MaterialModelData d;
  d.eos = MaterialModelData::TILLOTSON;
  d.rhomin = 2.7e-9; d.pmin = 1.0; d.rhomax = 1e10; d.pmax = 1e30; d.failsafe_density = 2.7e-3;
  TillotsonModelData &t = d.tillotModel;          // mm-g-s, as the hvi_emp deck writes it
  t.rho0 = 2.7e-3; t.e0 = 5e12; t.a = 0.5; t.b = 1.63; t.A = 7.52e10; t.B = 6.5e10;
  t.alpha = 5; t.beta = 5; t.rhoIV = 1.195062e-3; t.eIV = 3e12; t.eCV = 1.39e13;
  t.cv = 8.97e8; t.T0 = 300; t.cp = 0; t.h0 = 0;
  t.temperature_depends_on_density = TillotsonModelData::YES;

  printf("constructing VarFcnTillot ... "); fflush(stdout);
  VarFcnTillot vf(d);
  printf("ok\n");

  printf("T at rest: %.1f K (expect 300)\n", vf.GetTemperature(2.7e-3, 0.0));
  std::mt19937 g(1);
  std::uniform_real_distribution<double> u(-6.0, 0.7);
  double s = 0.0;
  printf("2e5 temperatures, rho in [1e-6, 5] rho0, random order ... "); fflush(stdout);
  for (int i = 0; i < 200000; i++) s += vf.GetTemperature(2.7e-3 * pow(10.0, u(g)), 2e13);
  printf("ok\n");
  printf("T(0.01 rho0, 20 MJ/kg) = %.1f K\n", vf.GetTemperature(2.7e-5, 2e13));
  printf("PASS\n");
  return std::isfinite(s) ? 0 : 1;
}
