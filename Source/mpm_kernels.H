#ifndef MPM_KERNELS_H_
#define MPM_KERNELS_H_
// clang-format off
#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <constants.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
// clang-format on

AMREX_GPU_DEVICE AMREX_FORCE_INLINE int
applybc(amrex::Real relvel_in[AMREX_SPACEDIM],
        amrex::Real relvel_out[AMREX_SPACEDIM],
        amrex::Real wall_mu,
        amrex::Real normaldir[AMREX_SPACEDIM],
        int bc)
{
    int modify_pos = false;

    amrex::Real velt[AMREX_SPACEDIM];
    amrex::Real velt_hat[AMREX_SPACEDIM];

    // Compute normal component of velocity
    amrex::Real veln = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        veln += relvel_in[d] * normaldir[d];
    }

    // Tangential component = relvel_in - veln * normal
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        velt[d] = relvel_in[d] - veln * normaldir[d];
    }

    // Magnitude of tangential velocity
    amrex::Real veltmag = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        veltmag += velt[d] * velt[d];
    }
    veltmag = std::sqrt(veltmag);

    // Unit tangential direction
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        velt_hat[d] = velt[d] / (veltmag + TINYVAL);
    }

    if (bc == BC_NOSLIPWALL)
    {
        modify_pos = true;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            relvel_out[d] = shunya;
        }
    }
    else if (bc == BC_SLIPWALL)
    {
        modify_pos = true;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            relvel_out[d] = velt[d];
        }
    }
    else if (bc == BC_PARTIALSLIPWALL)
    {
        if (veln <= shunya)
        {
            modify_pos = true;
            if (veltmag <= -wall_mu * veln)
            {
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    relvel_out[d] = shunya;
                }
            }
            else
            {
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    relvel_out[d] = (veltmag + wall_mu * veln) * velt_hat[d];
                }
            }
        }
    }
    else if (bc == BC_PERIODIC)
    {
        // nothing to do
    }
    else
    {
        amrex::Abort("\nUnknown boundary condition");
    }

    return modify_pos;
}


#endif
