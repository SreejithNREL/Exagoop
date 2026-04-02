// ─────────────────────────────────────────────────────────────────────────────
// ExaGOOP thermal BC UDF: spatially varying template
// ─────────────────────────────────────────────────────────────────────────────
//
// h(x,y,z) and T_inf(x,y,z) can be any function of position.
// Common use cases:
//   - h varies with distance from a stagnation point
//   - T_inf varies along a wall (e.g. inlet-to-outlet temperature profile)
//   - Prescribed heat flux q(x,y,z) — set T_inf=0 and use h as q directly
//     with bc type 3 (prescribed flux) instead
//
// BUILD:
//   exagoop-build-udf my_thermal_varying.cpp ./build
//
// INPUT FILE:
//   mpm.bc_upper_temp          = 5  0      # x-hi face: UDF, y-hi: none
//   mpm.bc_upper_thermal_udf_0 = /path/to/build/liblevelset_udf.dylib
// ─────────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <algorithm>

#ifdef EXAGOOP_UDF_WINDOWS_EXPORT
#  define EXAGOOP_API __declspec(dllexport)
#else
#  define EXAGOOP_API
#endif

extern "C" EXAGOOP_API double thermal_h(double x, double y, double z)
{
    // ── Edit below this line ──────────────────────────────────────────────────

    // Example: h increases linearly from 50 to 200 W/m^2/K along x
    // Domain assumed [0, 0.4] — adjust to your domain
    (void)z;
    double h_min = 50.0, h_max = 200.0;
    double domain_length = 0.4;
    double frac = std::min(std::max(x / domain_length, 0.0), 1.0);
    return h_min + frac * (h_max - h_min);

    // ── Edit above this line ──────────────────────────────────────────────────
}

extern "C" EXAGOOP_API double thermal_Tinf(double x, double y, double z)
{
    // ── Edit below this line ──────────────────────────────────────────────────

    // Example: T_inf varies sinusoidally along y
    // T_inf = 300 + 50 * sin(pi * y / 0.4)
    (void)x; (void)z;
    const double pi = 3.14159265358979323846;
    return 300.0 + 50.0 * std::sin(pi * y / 0.4);

    // ── Edit above this line ──────────────────────────────────────────────────
}
