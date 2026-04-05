#ifndef _MPM_EB_H_
#define _MPM_EB_H_

// clang-format off
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>

#if USE_EB
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EB2_IF_Cylinder.H>
#include <AMReX_EB2_IF_Plane.H>
#include <AMReX_EB2_IF_Polynomial.H>
#include <AMReX_EB2_IF_Intersection.H>
#include <AMReX_EB2_IF_Union.H>
#include <AMReX_EB2_IF_Translation.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_EB_utils.H>
#endif

#include <AMReX_PlotFileUtil.H>
#include <constants.H>
#include <interpolants.H>

// clang-format on

using namespace amrex;

#if USE_EB
// ============================================================
// Forward declarations — full type defined in mpm_udf_loader.H
// Only needed when USE_EB is on.
// ============================================================
class UDFImplicitFunction;
namespace amrex
{
class EBFArrayBoxFactory;
}

/**
 * @brief Builds EB2 index space + EBFArrayBoxFactory + nodal lsphi MultiFab
 *        from a UDF implicit function.
 *
 * Defined in mpm_eb_udf_build.cpp (CXX-only TU — never compiled by nvcc).
 */
void build_udf_eb(UDFImplicitFunction udf_if, // copyable — pass by value
                  const amrex::Geometry &geom,
                  const amrex::BoxArray &ba,
                  const amrex::DistributionMapping &dm,
                  int nghost,
                  int ls_refinement,
                  amrex::MultiFab *&lsphi_out,
                  amrex::EBFArrayBoxFactory *&ebfactory_out);
#endif // USE_EB

#if USE_EB
namespace mpm_ebtools
{
extern EBFArrayBoxFactory *ebfactory;
extern MultiFab *lsphi;
extern int ls_refinement;
extern bool using_levelset_geometry;

void init_eb(const Geometry &geom,
             const BoxArray &ba,
             const DistributionMapping &dm);
} // namespace mpm_ebtools
#endif

// ============================================================
// Level-set interpolation utilities
//
// These functions are called from:
//   - amrex::ParallelFor  (GPU device kernels)
//   - amrex::LoopOnCpu    (CPU host code, e.g. UDF path)
//
// Both call sites require AMREX_GPU_HOST_DEVICE.
// Using AMREX_GPU_DEVICE only would compile fine on CPU builds but fail
// with "calling __device__ from __host__" on CUDA builds.
// ============================================================

/**
 * @brief Evaluates the level-set field at an arbitrary physical point.
 *
 * Multilinear interpolation over the 2^D corners of the containing cell.
 *
 * @param[in] phi     Nodal Array4 containing the level-set field.
 * @param[in] problo  Physical lower bounds of the domain.
 * @param[in] dx      Coarse grid spacing (will be divided by lsref internally).
 * @param[in] ptxyz   Physical coordinates of the query point.
 * @param[in] lsref   Refinement factor used for the level-set grid.
 *
 * @return Interpolated level-set value phi(ptxyz).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
get_levelset_value(amrex::Array4<amrex::Real> phi,
                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo,
                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
                   amrex::Real ptxyz[AMREX_SPACEDIM],
                   amrex::Real lsref)
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> ls_dx;
    const int lo[AMREX_SPACEDIM] = {0};
    const int hi[AMREX_SPACEDIM] = {0};
    const amrex::GpuArray<int, AMREX_SPACEDIM> order_pseudo = {
        AMREX_D_DECL(1, 1, 1)};
    const amrex::GpuArray<int, AMREX_SPACEDIM> periodic_pseudo = {
        AMREX_D_DECL(1, 1, 1)};

    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        ls_dx[d] = dx[d] / lsref;

    IntVect idx;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        idx[d] = static_cast<int>(
            amrex::Math::floor((ptxyz[d] - problo[d] + TINYVAL) / ls_dx[d]));

    amrex::Real lsval = 0.0;
    const int nCorners = 1 << AMREX_SPACEDIM;

    for (int corner = 0; corner < nCorners; ++corner)
    {
        IntVect offset;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
            offset[d] = (corner >> d) & 1;

        int ii = idx[0] + offset[0];
        int jj = (AMREX_SPACEDIM > 1) ? idx[1] + offset[1] : 0;
        int kk = (AMREX_SPACEDIM > 2) ? idx[2] + offset[2] : 0;

        lsval +=
            phi(ii, jj, kk) * basisval(offset, idx, ptxyz, problo, ls_dx,
                                       order_pseudo, periodic_pseudo, lo, hi);
    }
    return lsval;
}

/**
 * @brief Computes the gradient of the level-set field at a physical point.
 *
 * Multilinear basis-function derivatives over the 2^D corners of the cell.
 *
 * @param[in]  phi     Nodal Array4 containing the level-set field.
 * @param[in]  problo  Physical lower bounds of the domain.
 * @param[in]  dx      Coarse grid spacing.
 * @param[in]  ptxyz   Physical coordinates of the query point.
 * @param[in]  lsref   Refinement factor used for the level-set grid.
 * @param[out] lsgrad  Output gradient vector ∇phi at ptxyz.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_levelset_grad(amrex::Array4<amrex::Real> phi,
                  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo,
                  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
                  amrex::Real ptxyz[AMREX_SPACEDIM],
                  amrex::Real lsref,
                  amrex::Real lsgrad[AMREX_SPACEDIM])
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> ls_dx;
    const int lo[AMREX_SPACEDIM] = {AMREX_D_DECL(0, 0, 0)};
    const int hi[AMREX_SPACEDIM] = {AMREX_D_DECL(0, 0, 0)};
    const amrex::GpuArray<int, AMREX_SPACEDIM> order_pseudo = {
        AMREX_D_DECL(1, 1, 1)};
    const amrex::GpuArray<int, AMREX_SPACEDIM> periodic_pseudo = {
        AMREX_D_DECL(1, 1, 1)};

    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        ls_dx[d] = dx[d] / lsref;

    IntVect idx;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        idx[d] = static_cast<int>(
            amrex::Math::floor((ptxyz[d] - problo[d] + TINYVAL) / ls_dx[d]));

    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        lsgrad[d] = 0.0;

    const int nCorners = 1 << AMREX_SPACEDIM;
    for (int corner = 0; corner < nCorners; ++corner)
    {
        IntVect offset;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
            offset[d] = (corner >> d) & 1;

        int ii = idx[0] + offset[0];
        int jj = (AMREX_SPACEDIM > 1) ? idx[1] + offset[1] : 0;
        int kk = (AMREX_SPACEDIM > 2) ? idx[2] + offset[2] : 0;

        for (int d = 0; d < AMREX_SPACEDIM; ++d)
            lsgrad[d] += phi(ii, jj, kk) *
                         basisvalder(d, offset, idx, ptxyz, problo, ls_dx,
                                     order_pseudo, periodic_pseudo, lo, hi);
    }
}

#endif // _MPM_EB_H_
