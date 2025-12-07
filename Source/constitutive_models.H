#ifndef CONSTITUTIVE_MODELS_H_
#define CONSTITUTIVE_MODELS_H_
// clang-format off
#include<constants.H>
#include <AMReX.H>
// clang-format on
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
linear_elastic(amrex::Real eps[NCOMP_TENSOR],
               amrex::Real sigma[NCOMP_TENSOR],
               amrex::Real E,
               amrex::Real v)
{
    amrex::Real const1 = E / (eka + v) / (eka - two * v);
    amrex::Real const2 = E / (eka + v);

#if (AMREX_SPACEDIM == 1)
    // 1D case: only XX component matters
    sigma[XX] = E * eps[XX];   // Hooke’s law in 1D
    sigma[YY] = sigma[ZZ] = 0.0;
    sigma[XY] = sigma[XZ] = sigma[YZ] = 0.0;

#elif (AMREX_SPACEDIM == 2)
    // 2D plane stress/strain case
    sigma[XX] = const1 * ((1 - v) * eps[XX] + v * eps[YY]);
    sigma[YY] = const1 * ((1 - v) * eps[YY] + v * eps[XX]);
    sigma[ZZ] = 0.0;  // out-of-plane stress often set to 0 in plane stress
    sigma[XY] = const2 * eps[XY];
    sigma[XZ] = sigma[YZ] = 0.0;

#elif (AMREX_SPACEDIM == 3)
    // Full 3D elasticity
    sigma[XX] = const1 * ((1 - v) * eps[XX] + v * (eps[YY] + eps[ZZ]));
    sigma[YY] = const1 * ((1 - v) * eps[YY] + v * (eps[XX] + eps[ZZ]));
    sigma[ZZ] = const1 * ((1 - v) * eps[ZZ] + v * (eps[XX] + eps[YY]));

    sigma[XY] = const2 * eps[XY];
    sigma[XZ] = const2 * eps[XZ];
    sigma[YZ] = const2 * eps[YZ];
#endif

}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
linear_elastic_delta(amrex::Real delta_eps[NCOMP_TENSOR],
               amrex::Real delta_sigma[NCOMP_TENSOR],
               amrex::Real E,
               amrex::Real v)
{
    amrex::Real const1 = E / (eka + v) / (eka - two * v);
    amrex::Real const2 = E / (eka + v);

#if (AMREX_SPACEDIM == 1)
    // 1D elasticity: only XX component
    delta_sigma[XX] = E * delta_eps[XX];
    delta_sigma[YY] = delta_sigma[ZZ] = 0.0;
    delta_sigma[XY] = delta_sigma[XZ] = delta_sigma[YZ] = 0.0;

#elif (AMREX_SPACEDIM == 2)
    // 2D elasticity (plane stress assumption)
    delta_sigma[XX] = const1 * ((1 - v) * delta_eps[XX] + v * delta_eps[YY]);
    delta_sigma[YY] = const1 * ((1 - v) * delta_eps[YY] + v * delta_eps[XX]);
    delta_sigma[ZZ] = 0.0;  // plane stress: out-of-plane stress neglected
    delta_sigma[XY] = const2 * delta_eps[XY];
    delta_sigma[XZ] = delta_sigma[YZ] = 0.0;

#elif (AMREX_SPACEDIM == 3)
    // Full 3D elasticity
    delta_sigma[XX] = const1 * ((1 - v) * delta_eps[XX] +
                                v * (delta_eps[YY] + delta_eps[ZZ]));
    delta_sigma[YY] = const1 * ((1 - v) * delta_eps[YY] +
                                v * (delta_eps[XX] + delta_eps[ZZ]));
    delta_sigma[ZZ] = const1 * ((1 - v) * delta_eps[ZZ] +
                                v * (delta_eps[XX] + delta_eps[YY]));

    delta_sigma[XY] = const2 * delta_eps[XY];
    delta_sigma[XZ] = const2 * delta_eps[XZ];
    delta_sigma[YZ] = const2 * delta_eps[YZ];
#endif
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
Newtonian_Fluid(amrex::Real epsdot[NCOMP_TENSOR],
                amrex::Real sigma[NCOMP_TENSOR],
                amrex::Real dyn_visc,
                amrex::Real pressure)
{
#if (AMREX_SPACEDIM == 1)
    // 1D Newtonian fluid: only XX component
    sigma[XX] = 2.0 * dyn_visc * epsdot[XX] - pressure;
    sigma[YY] = sigma[ZZ] = 0.0;
    sigma[XY] = sigma[XZ] = sigma[YZ] = 0.0;

#elif (AMREX_SPACEDIM == 2)
    // 2D Newtonian fluid
    amrex::Real trace_epsdot = epsdot[XX] + epsdot[YY];
    amrex::Real one_by_two = 0.5;  // average for 2D trace
	
    sigma[XX] = 2.0 * dyn_visc * (epsdot[XX] - one_by_two * trace_epsdot) - pressure;
    sigma[YY] = 2.0 * dyn_visc * (epsdot[YY] - one_by_two * trace_epsdot) - pressure;
    sigma[ZZ] = 0.0;
    sigma[XY] = 2.0 * dyn_visc * epsdot[XY];
    sigma[XZ] = sigma[YZ] = 0.0;

#elif (AMREX_SPACEDIM == 3)
    // Full 3D Newtonian fluid
    amrex::Real trace_epsdot = epsdot[XX] + epsdot[YY] + epsdot[ZZ];
    amrex::Real one_by_three = 1.0 / 3.0;

    sigma[XX] = 2.0 * dyn_visc * (epsdot[XX] - one_by_three * trace_epsdot) - pressure;
    sigma[YY] = 2.0 * dyn_visc * (epsdot[YY] - one_by_three * trace_epsdot) - pressure;
    sigma[ZZ] = 2.0 * dyn_visc * (epsdot[ZZ] - one_by_three * trace_epsdot) - pressure;

    sigma[XY] = 2.0 * dyn_visc * epsdot[XY];
    sigma[XZ] = 2.0 * dyn_visc * epsdot[XZ];
    sigma[YZ] = 2.0 * dyn_visc * epsdot[YZ];
#endif
}


#endif
