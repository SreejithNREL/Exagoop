// ─────────────────────────────────────────────────────────────────────────────
// ExaGOOP thermal BC UDF template
// ─────────────────────────────────────────────────────────────────────────────
//
// Use this file to define spatially varying convective heat transfer BCs.
// Compile and point the input file at the resulting .so/.dylib.
//
// BUILD:
//   CMake:   exagoop-build-udf my_thermal.cpp ./build
//   GNUmake: make udf UDF_SRC=my_thermal.cpp
//
// INPUT FILE (type 5 = convective UDF):
//   mpm.bc_lower_temp         = 0  5       # face 0: none, face 1 (y-lo): UDF
//   mpm.bc_lower_thermal_udf_1 = /path/to/build/liblevelset_udf.dylib
//
// FUNCTIONS:
//   thermal_h(x,y,z)    — heat transfer coefficient h [W/m^2/K]  (required)
//   thermal_Tinf(x,y,z) — ambient temperature T_inf [K]          (optional)
//
// If thermal_Tinf is not exported, T_inf = 0.0 everywhere.
// The convective flux applied at each boundary node is:
//   Q_node = h(x,y,z) * (T_inf(x,y,z) - T_node) * A_node
// ─────────────────────────────────────────────────────────────────────────────

#include <cmath>

#ifdef EXAGOOP_UDF_WINDOWS_EXPORT
#  define EXAGOOP_API __declspec(dllexport)
#else
#  define EXAGOOP_API
#endif

extern "C" EXAGOOP_API double thermal_h(double x, double y, double z)
{
    // ── Edit below this line ──────────────────────────────────────────────────

    // Example: uniform h = 100 W/m^2/K everywhere on this face
    (void)x; (void)y; (void)z;
    return 100.0;

    // ── Edit above this line ──────────────────────────────────────────────────
}

extern "C" EXAGOOP_API double thermal_Tinf(double x, double y, double z)
{
    // ── Edit below this line ──────────────────────────────────────────────────

    // Example: uniform ambient temperature T_inf = 300 K
    (void)x; (void)y; (void)z;
    return 300.0;

    // ── Edit above this line ──────────────────────────────────────────────────
}
