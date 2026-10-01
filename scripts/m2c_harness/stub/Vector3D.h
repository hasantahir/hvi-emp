// Stub for scripts/m2c_harness/tillotson_selftest.cpp only -- just enough
// of M2C/boost for VarFcnTillot.h to compile on its own. Not M2C code.
#pragma once
struct Vec3D { double v[3]; double& operator[](int i){return v[i];} };
