// clang-format off
/**
 * @file mpm_eb_udf_build.cpp
 *
 * @brief CPP file for building EB2 geometry from a UDF.
 *
 * This file is compiled as CUDA (via nvcc) in CUDA-enabled builds, and as
 * plain CXX in CPU-only builds.  The content is host-only code that calls
 * EB2::makeShop / EB2::Build; no device kernels are defined here.
 **
 * In CMake (BuildExaGOOPExe.cmake):
 *   All .cpp files are set to LANGUAGE CUDA when EXAGOOP_ENABLE_CUDA is ON.
 *
 * In GNUmake (Make.package):
 *   CEXE_sources += mpm_eb_udf_build.cpp
 * (GNUmake routes .cpp files through the host compiler, which is correct for
 *  non-CUDA GNUmake builds.  CUDA GNUmake builds use nvcc for all sources.)
 */
// clang-format on

#include <AMReX_Config.H>
#if defined(AMREX_USE_CUDA) && !defined(__CUDACC__)
#  define __host__
#  define __device__
#  define __global__
#  define __shared__
#  define __constant__
#  define __forceinline__ inline
#endif

#define EXAGOOP_INCLUDE_EB2_IF

#include <mpm_eb.H>

#if USE_EB
#include <AMReX_EB2.H>
#include <mpm_udf_loader.H>

// ============================================================
// MultiUDF — union implicit function over multiple UDF bodies.
// ============================================================
struct MultiUDF
{
    using PhiFn = double (*)(double, double, double);

    PhiFn fns[EXAGOOP_MAX_LS_BODIES];
    int n_bodies = 0;

    void add(const UDFImplicitFunction &udf_if)
    {
        AMREX_ALWAYS_ASSERT(n_bodies < EXAGOOP_MAX_LS_BODIES);
        fns[n_bodies++] = udf_if.phi_fn();
    }

    AMREX_FORCE_INLINE amrex::Real operator()(const amrex::RealArray &p) const
    {
        amrex::Real min_phi =
            static_cast<amrex::Real>(fns[0](p[0], p[1], p[2]));
        for (int b = 1; b < n_bodies; ++b)
        {
            amrex::Real phi =
                static_cast<amrex::Real>(fns[b](p[0], p[1], p[2]));
            if (phi < min_phi)
                min_phi = phi;
        }
        return min_phi;
    }
};

/**
 * @brief Builds an EB2 index space and EBFArrayBoxFactory from a UDF
 *        implicit function, and allocates + defines the nodal lsphi MultiFab.
 *
 * The caller is responsible for passing in a valid UDFImplicitFunction.
 * This function does NOT fill lsphi with signed-distance values — that is
 * done by the caller via amrex::LoopOnCpu after this returns.
 *
 * @param[in]  udf_if          Implicit function (phi<0 inside obstacle).
 * @param[in]  geom            Coarse-level geometry.
 * @param[in]  ba              BoxArray for the coarse level.
 * @param[in]  dm              DistributionMapping.
 * @param[in]  nghost          Number of ghost cells for factory and lsphi.
 * @param[in]  ls_refinement   Refinement factor for the level-set grid.
 * @param[out] lsphi_out       Newly allocated nodal MultiFab (caller owns).
 * @param[out] ebfactory_out   Newly allocated EBFArrayBoxFactory (caller owns).
 *                             Any existing factory pointed to by ebfactory_out
 *                             is deleted before the new one is allocated.
 */
void build_udf_eb(UDFImplicitFunction udf_if,
                  const amrex::Geometry &geom,
                  const amrex::BoxArray &ba,
                  const amrex::DistributionMapping &dm,
                  int nghost,
                  int ls_refinement,
                  amrex::MultiFab *&lsphi_out,
                  amrex::EBFArrayBoxFactory *&ebfactory_out)
{
    amrex::Box dom_ls = geom.Domain();
    dom_ls.refine(ls_refinement);
    amrex::Geometry geom_ls(dom_ls);

    int required_coarsening_level = 0;
    if (ls_refinement > 1)
    {
        int tmp = ls_refinement;
        while (tmp >>= 1)
            ++required_coarsening_level;
    }

    auto shop = amrex::EB2::makeShop(udf_if);
    amrex::EB2::Build(shop, geom_ls, required_coarsening_level,
                      /*max_coarsening_level=*/10);

    const amrex::EB2::IndexSpace &ebis = amrex::EB2::IndexSpace::top();
    const amrex::EB2::Level &eblev = ebis.getLevel(geom);

    delete ebfactory_out;
    ebfactory_out = new amrex::EBFArrayBoxFactory(
        eblev, geom, ba, dm, {nghost, nghost, nghost}, amrex::EBSupport::full);

    amrex::BoxArray ls_ba = amrex::convert(ba, amrex::IntVect::TheNodeVector());
    ls_ba.refine(ls_refinement);

    lsphi_out = new amrex::MultiFab;
    lsphi_out->define(ls_ba, dm, /*ncomp=*/1, nghost);
}

#endif // USE_EB
