// ─────────────────────────────────────────────────────────────────────────────
// ExaGOOP level-set UDF: advanced template
// ─────────────────────────────────────────────────────────────────────────────
//
// INSTRUCTIONS
// ────────────
// This template gives you access to the full AMReX EB2 implicit-function
// library, including all analytic primitives and compositing operations.
// Use it when you want to combine shapes (union, intersection, subtraction,
// translation, rotation) without writing the SDF math by hand.
//
// 1. Edit the body of levelset_phi() below.
//    Return φ < 0 inside the solid, φ > 0 outside.
//
// 2. Build the shared library:
//    Linux/macOS:  exagoop-build-udf my_geometry.cpp ./build
//    Windows:      exagoop-build-udf.bat my_geometry.cpp .\build
//    GNUmake:      make -f GNUmakefile.udf UDF_SRC=my_geometry.cpp AMREX_HOME=...
//
// 3. Add to your ExaGOOP input file:
//    eb2.geom_type   = udf_cpp
//    eb2.udf_so_file = ./build/udf_build/levelset_udf.so
//    eb2.ls_refinement = 2
//
// AVAILABLE AMReX EB2 PRIMITIVES
// ───────────────────────────────
//   EB2::SphereIF   (center, radius, inside)
//   EB2::PlaneIF    (point_on_plane, outward_normal, inside)
//   EB2::CylinderIF (radius, length, axis_dir, center, inside)
//   EB2::BoxIF      (lo_corner, hi_corner, inside)
//   EB2::EllipsoidIF(radii, center, inside)
//   EB2::TiltedCylinderIF / EB2::PolynomialIF / EB2::SphereIF
//
//   inside=false → solid is the region where primitive < 0 (standard convention)
//   inside=true  → flip: solid is where primitive > 0
//
// COMPOSITING
// ───────────
//   makeUnion(A, B, ...)          union of solids  (min φ)
//   makeIntersection(A, B, ...)   intersection      (max φ)
//   makeComplement(A)             flip inside/outside
//   translate(A, {dx,dy,dz})      rigid translation
//   EB2::rotate(A, angle, axis)   rigid rotation
//
// CALLING CONVENTION
// ──────────────────
// AMReX IF objects are callable as if_obj(RealArray{x,y,z}) → Real.
// Cast to double before returning.
// ─────────────────────────────────────────────────────────────────────────────

// AMReX EB2 implicit-function headers
#include <AMReX_EB2_IF_Sphere.H>
#include <AMReX_EB2_IF_Plane.H>
#include <AMReX_EB2_IF_Cylinder.H>
#include <AMReX_EB2_IF_Box.H>
#include <AMReX_EB2_IF_Ellipsoid.H>
#include <AMReX_EB2_IF_Union.H>
#include <AMReX_EB2_IF_Intersection.H>
#include <AMReX_EB2_IF_Complement.H>
#include <AMReX_EB2_IF_Translation.H>

#include <AMReX_Array.H>
#include <cmath>
#include <algorithm>

using namespace amrex;

// ── Windows DLL export (ignored on Linux/macOS) ───────────────────────────────
#ifdef EXAGOOP_UDF_WINDOWS_EXPORT
#  define EXAGOOP_API __declspec(dllexport)
#else
#  define EXAGOOP_API
#endif

extern "C" EXAGOOP_API double levelset_phi(double x, double y, double z)
{
    // ── Edit below this line ──────────────────────────────────────────────────
    //
    // Example: a wedge hopper
    //   - two angled planes forming a funnel
    //   - a vertical box forming the bin above
    //   Combined with makeUnion, then complemented so the
    //   interior of the hopper is the solid region.
    //
    //   Adjust the geometry parameters to match your domain.

    // Left funnel wall: point (−0.05, 0, 0), outward normal pointing left+up
    EB2::PlaneIF left_wall ({-0.05, 0.0, 0.0},
                             {-1.0,  0.5, 0.0}, false);

    // Right funnel wall: mirror of left
    EB2::PlaneIF right_wall({ 0.05, 0.0, 0.0},
                             { 1.0,  0.5, 0.0}, false);

    // Bin above the funnel (axis-aligned box)
    EB2::BoxIF bin({-0.10,  0.05, -0.10},
                   { 0.10,  0.50,  0.10}, false);

    // Union: a point is inside the solid if it is inside any component
    auto hopper = EB2::makeUnion(left_wall, right_wall, bin);

    // Translate so the exit is at the domain centre
    // (replace with your actual domain centre if needed)
    auto hopper_centred = EB2::translate(hopper, RealArray{0.5, 0.3, 0.5});

    RealArray p{x, y, z};
    return static_cast<double>(hopper_centred(p));

    // ── Edit above this line ──────────────────────────────────────────────────
}
