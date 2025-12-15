#ifndef _MPM_EB_H_
#define _MPM_EB_H_
// clang-format off
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_EB2_IF_Cylinder.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_PlotFileUtil.H>

#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EB2_IF_Plane.H>
#include <AMReX_EB2_IF_Polynomial.H>
#include <AMReX_EB2_IF_Intersection.H>
#include <AMReX_EB2_IF_Union.H>
#include <AMReX_EB2_IF_Intersection.H>
#include <AMReX_EB2_IF_Translation.H>

#include <AMReX_EB_utils.H>
#include <constants.H>
#include <interpolants.H>
// clang-format on

using namespace amrex;

namespace mpm_ebtools
{
extern EBFArrayBoxFactory *ebfactory;
extern MultiFab *lsphi;
extern int ls_refinement;
extern bool using_levelset_geometry;
void init_eb(const Geometry &geom,
             const BoxArray &ba,
             const DistributionMapping &dm);
void make_wedge_hopper_levelset(const Geometry &geom,
                                const BoxArray &ba,
                                const DistributionMapping &dm);
} // namespace mpm_ebtools

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
get_levelset_value(amrex::Array4<amrex::Real> phi,
                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo,
                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
                   amrex::Real ptxyz[AMREX_SPACEDIM],
                   amrex::Real lsref)
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> ls_dx;
    const int lo[AMREX_SPACEDIM] = {0};
    const int hi[AMREX_SPACEDIM] = {0};
    const amrex::GpuArray<int, AMREX_SPACEDIM> order_scheme_pseudo = {
        AMREX_D_DECL(1, 1, 1)};
    const amrex::GpuArray<int, AMREX_SPACEDIM> periodic_pseudo = {
        AMREX_D_DECL(1, 1, 1)};

    // scale dx by lsref
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        ls_dx[d] = dx[d] / lsref;
    }

    // compute cell indices in each dimension
    IntVect idx;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        idx[d] = static_cast<int>(
            amrex::Math::floor((ptxyz[d] - problo[d] + TINYVAL) / ls_dx[d]));
    }

    amrex::Real lsval = 0.0;

    // nested loops over dimensions (generalized)
    // we use recursion or iterative product to cover 2^AMREX_SPACEDIM corners
    int cornerCount = 1 << AMREX_SPACEDIM; // 2^dim
    for (int corner = 0; corner < cornerCount; ++corner)
    {
        IntVect offset;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            offset[d] = (corner >> d) & 1; // 0 or 1
        }

        // gather indices
        int ii = idx[0] + offset[0];
        int jj = (AMREX_SPACEDIM > 1) ? idx[1] + offset[1] : 0;
        int kk = (AMREX_SPACEDIM > 2) ? idx[2] + offset[2] : 0;

        lsval += phi(ii, jj, kk) * basisval(offset, idx, ptxyz, problo, ls_dx,
                                            order_scheme_pseudo,
                                            periodic_pseudo, lo, hi);
    }

    return lsval;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
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
    const amrex::GpuArray<int, AMREX_SPACEDIM> order_scheme_pseudo = {
        AMREX_D_DECL(1, 1, 1)};
    const amrex::GpuArray<int, AMREX_SPACEDIM> periodic_pseudo = {
        AMREX_D_DECL(1, 1, 1)};

    // scale dx by lsref
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        ls_dx[d] = dx[d] / lsref;
    }

    // compute cell indices
    IntVect idx;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        idx[d] = static_cast<int>(
            amrex::Math::floor((ptxyz[d] - problo[d] + TINYVAL) / ls_dx[d]));
    }

    // initialize gradient
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        lsgrad[d] = 0.0;
    }

    // loop over 2^dim corners
    int cornerCount = 1 << AMREX_SPACEDIM;
    for (int corner = 0; corner < cornerCount; ++corner)
    {
        IntVect offset;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            offset[d] = (corner >> d) & 1; // 0 or 1
        }

        // gather indices (safe for lower dims)
        int ii = idx[0] + offset[0];
        int jj = (AMREX_SPACEDIM > 1) ? idx[1] + offset[1] : 0;
        int kk = (AMREX_SPACEDIM > 2) ? idx[2] + offset[2] : 0;

        // accumulate gradient contributions for each dimension
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            lsgrad[d] +=
                phi(ii, jj, kk) * basisvalder(d, offset, idx, ptxyz, problo,
                                              ls_dx, order_scheme_pseudo,
                                              periodic_pseudo, lo, hi);
        }
    }
}

#endif
