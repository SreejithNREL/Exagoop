// ─────────────────────────────────────────────────────────────────────────────
// ExaGOOP level-set UDF: simple template
// ─────────────────────────────────────────────────────────────────────────────
//
// INSTRUCTIONS
// ────────────
// 1. Edit the body of levelset_phi() below to define your geometry.
//    φ(x,y,z) < 0  →  inside the solid / wall
//    φ(x,y,z) > 0  →  outside (material / fluid region)
//    φ(x,y,z) = 0  →  the surface
//
//    For a signed distance field, |φ| equals the distance to the surface.
//    AMReX's FillSignedDistance will re-distance the field automatically,
//    so approximate signed distances are acceptable.
//
// 2. Build the shared library:
//    Linux/macOS:  exagoop-build-udf my_geometry.cpp ./build
//    Windows:      exagoop-build-udf.bat my_geometry.cpp .\build
//    GNUmake:      make -f GNUmakefile.udf UDF_SRC=my_geometry.cpp AMREX_HOME=...
//
// 3. Add to your ExaGOOP input file:
//    eb2.geom_type   = udf_cpp
//    eb2.udf_so_file = ./build/udf_build/levelset_udf.so   # Linux/macOS
//    # eb2.udf_so_file = .\build\udf_build\levelset_udf.dll  # Windows
//    eb2.ls_refinement = 2
//    mpm.lsetbc        = 2     # 1=no-slip  2=free-slip  3=Coulomb
//    mpm.lset_wall_mu  = 0.3   # friction coefficient (lsetbc=3 only)
//
// COMPOSITING PRIMITIVES (pure C++ math, no AMReX headers needed)
// ───────────────────────────────────────────────────────────────
//   Union (inside either):       φ = min(φ_A, φ_B)
//   Intersection (inside both):  φ = max(φ_A, φ_B)
//   Complement (flip solid):     φ = -φ_A
//   Subtraction (A minus B):     φ = max(φ_A, -φ_B)
//
// COMMON PRIMITIVE FORMULAS
// ─────────────────────────
//   Sphere:    sqrt((x-cx)²+(y-cy)²+(z-cz)²) - r
//   Plane:     (x-px)*nx + (y-py)*ny + (z-pz)*nz       (n̂ points outward)
//   Cylinder:  sqrt((x-cx)²+(z-cz)²) - r               (axis along Y)
//   Box:       max(|x-cx|-hx, |y-cy|-hy, |z-cz|-hz)
//   Half-space below y=0.3:  -(y - 0.3)                (solid below the plane)
// ─────────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <algorithm>   // std::min, std::max

// ── Windows DLL export (ignored on Linux/macOS) ───────────────────────────────
#ifdef EXAGOOP_UDF_WINDOWS_EXPORT
#  define EXAGOOP_API __declspec(dllexport)
#else
#  define EXAGOOP_API
#endif

extern "C" EXAGOOP_API double levelset_phi(double x, double y, double z)
{
    // ── Edit below this line ──────────────────────────────────────────────────

    // Example: sphere of radius 0.05 centred at (0.5, 0.5, 0.5)
    const double cx = 0.5, cy = 0.5, cz = 0.5, r = 0.05;
    return std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz)) - r;

    // ── Edit above this line ──────────────────────────────────────────────────
}
