// clang-format off
/**
 * @file mpm_eb_udf_build.cpp
 *
 * @brief CXX-only translation unit for building EB2 geometry from a UDF.
 *
 * This file MUST be compiled as CXX (never by nvcc/CUDA).
 * Reason: EB2::Build instantiates templates that pull in stub.c via nvcc's
 * device-function scanner when compiled as CUDA, producing an unresolvable
 * link error.  Isolating it here and forcing LANGUAGE CXX in CMake avoids
 * that entirely.
 *
 * In CMake (BuildExaGOOPExe.cmake):
 *   set_source_files_properties(
 *       ${SRC_DIR}/mpm_eb_udf_build.cpp
 *       PROPERTIES LANGUAGE CXX)
 *
 * In GNUmake (Make.package), add:
 *   CEXE_sources += mpm_eb_udf_build.cpp
 * (GNUmake never calls nvcc on .cpp files, so no extra annotation needed.)
 */
// clang-format on

#include <mpm_eb.H>

#if USE_EB
#include <AMReX_EB2.H>
#include <functional>
#include <mpm_udf_loader.H>

/**
 * @brief Builds an EB2 index space and EBFArrayBoxFactory from a UDF
 *        implicit function, and allocates + defines the nodal lsphi MultiFab.
 *
 * The caller is responsible for passing in a valid UDFImplicitFunction (or
 * any callable matching `Real(const RealArray&)`).  This function does NOT
 * fill lsphi with signed-distance values — that is done by the caller via
 * amrex::LoopOnCpu after this returns.
 *
 * @param[in]  udf_if          Implicit function (phi<0 inside obstacle).
 * @param[in]  geom            Coarse-level geometry.
 * @param[in]  ba              BoxArray for the coarse level.
 * @param[in]  dm              DistributionMapping.
 * @param[in]  nghost          Number of ghost cells for factory and lsphi.
 * @param[in]  ls_refinement   Refinement factor for the level-set grid.
 * @param[out] lsphi_out       Newly allocated nodal MultiFab (caller owns).
 * @param[out] ebfactory_out   Newly allocated EBFArrayBoxFactory (caller owns).
 */
void build_udf_eb(UDFImplicitFunction udf_if, // copyable — EB2 will copy it
                  const amrex::Geometry &geom,
                  const amrex::BoxArray &ba,
                  const amrex::DistributionMapping &dm,
                  int nghost,
                  int ls_refinement,
                  amrex::MultiFab *&lsphi_out,
                  amrex::EBFArrayBoxFactory *&ebfactory_out)
{
    // Build a refined geometry for the level-set EB
    amrex::Box dom_ls = geom.Domain();
    dom_ls.refine(ls_refinement);
    amrex::Geometry geom_ls(dom_ls);

    // required_coarsening_level = log2(ls_refinement)
    int required_coarsening_level = 0;
    if (ls_refinement > 1)
    {
        int tmp = ls_refinement;
        while (tmp >>= 1)
            ++required_coarsening_level;
    }

    // EB2::makeShop requires the implicit function by value; wrap in a lambda
    // so the UDFImplicitFunction (which owns the dlopen handle) is captured
    // by reference — safe because udf_if outlives this call.
    auto shop = amrex::EB2::makeShop(udf_if);
    amrex::EB2::Build(shop, geom_ls, required_coarsening_level,
                      /*max_coarsening_level=*/10);

    const amrex::EB2::IndexSpace &ebis = amrex::EB2::IndexSpace::top();
    const amrex::EB2::Level &eblev = ebis.getLevel(geom);

    // Build factory at coarse level
    ebfactory_out = new amrex::EBFArrayBoxFactory(
        eblev, geom, ba, dm, {nghost, nghost, nghost}, amrex::EBSupport::full);

    // Allocate nodal lsphi at refined resolution
    amrex::BoxArray ls_ba = amrex::convert(ba, amrex::IntVect::TheNodeVector());
    ls_ba.refine(ls_refinement);

    lsphi_out = new amrex::MultiFab;
    lsphi_out->define(ls_ba, dm, /*ncomp=*/1, nghost);
}

#endif // USE_EB
