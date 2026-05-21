#ifndef _MPM_EB_H_
#define _MPM_EB_H_

// clang-format off
#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>

#if USE_EB
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_EB_utils.H>


#ifdef EXAGOOP_INCLUDE_EB2_IF
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EB2_IF_Cylinder.H>
#include <AMReX_EB2_IF_Plane.H>
#include <AMReX_EB2_IF_Polynomial.H>
#include <AMReX_EB2_IF_Intersection.H>
#include <AMReX_EB2_IF_Union.H>
#include <AMReX_EB2_IF_Translation.H>
#endif // EXAGOOP_INCLUDE_EB2_IF
#endif // USE_EB

#include <AMReX_PlotFileUtil.H>
#include <constants.H>
#include <interpolants.H>
// clang-format on

using namespace amrex;

#if USE_EB

constexpr int EXAGOOP_MAX_LS_BODIES = 8; // Maximum number of level set bodies

// Each body has its own level set data structure. Using a struct to group all
// quantities
struct LevelSetBody
{
    std::string name;                 // name of body
    amrex::MultiFab *lsphi = nullptr; // levelset value
    int ls_refinement = 1;            // levelset refinement factor

    std::string mom_bc_type =
        "noslipwall";          // BC for each level set (1bc for each levelset)
    amrex::Real wall_mu = 0.0; // friction coeff.
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
        wall_vel = // in case wall velocity is specified
        {AMREX_D_DECL(0.0, 0.0, 0.0)};

    /**
     * @brief Maps the string mom_bc_type to the integer BC_* constant
     *        used by applybc() in mpm_kernels.H.
     *
     * Valid strings: "noslipwall", "slipwall", "partialslip".
     * Returns BC_NOSLIPWALL if the string is unrecognised (with Abort).
     */
    int mom_bc_int() const
    {
        if (mom_bc_type == "noslipwall")
            return BC_NOSLIPWALL;
        if (mom_bc_type == "slipwall")
            return BC_SLIPWALL;
        if (mom_bc_type == "partialslip")
            return BC_PARTIALSLIPWALL;
        amrex::Abort("LevelSetBody '" + name +
                     "': unknown levelset_mom type '" + mom_bc_type +
                     "'."
                     " Valid values: noslipwall, slipwall, partialslip.");
        return BC_NOSLIPWALL;
    }

    // Temperature BC — set by init_eb() from <name>.temp_bc_type etc.
    //
    //   "adiabatic"  (default) — no thermal constraint (natural BC, zero flux)
    //   "isothermal"           — Dirichlet: T = T_wall
    //   "heatflux"             — Neumann: T_node = T_nb + heat_flux * dx /
    //   k_node "convective"           — Robin:   T_node = (T_nb + Bi * T_inf) /
    //   (1 + Bi)
    //                                     Bi = h_conv * dx / k_node
    std::string temp_bc_type = "adiabatic";
    amrex::Real T_wall = 0.0;
    amrex::Real heat_flux = 0.0;
    amrex::Real h_conv = 0.0;
    amrex::Real T_inf = 0.0;

    /**
     * @brief Maps temp_bc_type string to an integer code matching the
     *        convention used by nodal_bcs_temperature():
     *          0 = adiabatic (no-op)
     *          1 = isothermal (Dirichlet)
     *          3 = heatflux  (Neumann, prescribed flux)
     *          4 = convective (Robin)
     */
    int temp_bc_int() const
    {
        if (temp_bc_type == "adiabatic")
            return 0;
        if (temp_bc_type == "isothermal")
            return 1;
        if (temp_bc_type == "heatflux")
            return 3;
        if (temp_bc_type == "convective")
            return 4;
        amrex::Abort(
            "LevelSetBody '" + name + "': unknown temp_bc_type '" +
            temp_bc_type +
            "'."
            " Valid values: adiabatic, isothermal, heatflux, convection.");
        return 0;
    }
};

// ============================================================
// Forward declarations — full type defined in mpm_udf_loader.H
// Only needed when USE_EB is true.
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
 * The lsphi MultiFab is allocated but NOT filled — the caller fills it via
 * amrex::LoopOnCpu after this returns.
 */
void build_udf_eb(UDFImplicitFunction udf_if,
                  const amrex::Geometry &geom,
                  const amrex::BoxArray &ba,
                  const amrex::DistributionMapping &dm,
                  int nghost,
                  int ls_refinement,
                  amrex::MultiFab *&lsphi_out,
                  amrex::EBFArrayBoxFactory *&ebfactory_out);

namespace mpm_ebtools
{
// Grouping eb related tools. EBFArrayBoxFactory is used in FillSignedDistance
// (only) but we will retain it for future use

extern std::vector<LevelSetBody> ls_bodies;

extern EBFArrayBoxFactory *ebfactory;
extern bool using_levelset_geometry;

void init_eb(const Geometry &geom,
             const BoxArray &ba,
             const DistributionMapping &dm,
             MPMspecs &specs);
} // namespace mpm_ebtools
#endif // USE_EB

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

        int ii = amrex::Clamp(idx[0] + offset[0], amrex::lbound(phi).x,
                              amrex::ubound(phi).x);
        int jj = (AMREX_SPACEDIM > 1)
                     ? amrex::Clamp(idx[1] + offset[1], amrex::lbound(phi).y,
                                    amrex::ubound(phi).y)
                     : 0;
        int kk = (AMREX_SPACEDIM > 2)
                     ? amrex::Clamp(idx[2] + offset[2], amrex::lbound(phi).z,
                                    amrex::ubound(phi).z)
                     : 0;

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
 * @param[out] lsgrad  Output gradient vector grad(phi) at ptxyz.
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

        int ii = amrex::Clamp(idx[0] + offset[0], amrex::lbound(phi).x,
                              amrex::ubound(phi).x);
        int jj = (AMREX_SPACEDIM > 1)
                     ? amrex::Clamp(idx[1] + offset[1], amrex::lbound(phi).y,
                                    amrex::ubound(phi).y)
                     : 0;
        int kk = (AMREX_SPACEDIM > 2)
                     ? amrex::Clamp(idx[2] + offset[2], amrex::lbound(phi).z,
                                    amrex::ubound(phi).z)
                     : 0;

        for (int d = 0; d < AMREX_SPACEDIM; ++d)
            lsgrad[d] += phi(ii, jj, kk) *
                         basisvalder(d, offset, idx, ptxyz, problo, ls_dx,
                                     order_pseudo, periodic_pseudo, lo, hi);
    }
}

#endif // _MPM_EB_H_
