#ifndef CONSTITUTIVE_MODELS_H_
#define CONSTITUTIVE_MODELS_H_
// clang-format off
#include<constants.H>
#include <AMReX.H>
// clang-format on

//Number of max constitutive model parameters
constexpr int MAX_MODEL_PARAMS = 24;

//Namespace for constitutive models (not materials)
namespace ConstitutiveModel
{
enum
{
    ELASTIC = 0,
    FLUID = 1,
    NEOHOOKEAN = 2,
    JOHNSON_COOK = 3
};
}

// Per-model parameter layout within MaterialParams::p[].
namespace ElasticP
{
enum
{
    E = 0,
    nu = 1,
    count
};
}
namespace FluidP
{
enum
{
    bulk = 0,  // bulk modulus
    gama = 1,  // EoS exponent (Gamma)
    visc = 2,  // dynamic viscosity
	p_inf = 3, // free stream pressure
    count
};
}
namespace NeoHookeanP
{
enum
{
    E = 0,
    nu = 1,
    count
};
}
// Johnson-Cook (+ Mie-Gruneisen EOS) parameters
namespace JCP
{
enum
{
    E = 0,
    nu = 1,
    A = 2,         // JC_A yield stress
    B = 3,         // JC_B hardening modulus
    n = 4,         // JC_n hardening exponent
    C = 5,         // JC_C strain-rate sensitivity
    m = 6,         // JC_m thermal softening exponent
    eps_dot_0 = 7, // reference plastic strain rate
    Tr = 8,        // reference temperature
    Tm = 9,        // melting temperature
    chi = 10,      // Taylor-Quinney fraction
    c0 = 11,       // bulk sound speed (EOS)
    Salpha = 12,   // Hugoniot slope (EOS)
    Gamma0 = 13,   // Gruneisen Gamma (EOS)
    rho0 = 14,     // reference density
    // Johnson-Cook damage parameters. Damages disabled
    D1 = 15,       // initial failure strain
    D2 = 16,       // exponential factor
    D3 = 17,       // triaxiality sensitivity
    D4 = 18,       // strain-rate sensitivity
    D5 = 19,       // temperature sensitivity
    count
};
}
// Johnson-Cook per-particle internal state variables, stored in realData::isv.
namespace JC_ISV
{
enum
{
    ep = 0,       // equivalent plastic strain
    sdev = 1,     // un-rotated deviatoric stress, 6 Voigt comps (sdev..sdev+5)
    damage = 7,   // accumulated JC damage (Phase JC-3)
    count
};
}
namespace Fluid_ISV
{
enum
{       
    pressure = 0,   // pressure is unique to fluid, and also evolves with solution. Hence storing it in ISV
    count
};
}

//Each instance of a material used in the simulation: For example, polysulfone and steel are elastic, but will have different structs
struct MaterialParams
{
    int material_id=-1;						//Each material has a unique id
    int model = ConstitutiveModel::ELASTIC;			//Elastic by default
    amrex::Real p[MAX_MODEL_PARAMS] = {0.0};
};


namespace
{
struct ConstitutiveModelInfo
{
    const char *name;
    int id;
    std::vector<const char *> param_names; // in slot order
};
const std::vector<ConstitutiveModelInfo> &Get_Constitutive_Model_Registry()
{
    static const std::vector<ConstitutiveModelInfo> reg =
    {
        {"elastic",      ConstitutiveModel::ELASTIC,      {"E", "nu"}},
        {"fluid",        ConstitutiveModel::FLUID,        {"Bulk_modulus", "Gama_pressure", "Dynamic_viscosity"}},
	{"neohookean",   ConstitutiveModel::NEOHOOKEAN,   {"E", "nu"}},
        {"johnson_cook", ConstitutiveModel::JOHNSON_COOK, {"E", "nu", "JC_A", "JC_B", "JC_n", "JC_C", "JC_m", "JC_eps_dot_0", "JC_Tr", "JC_Tm", "JC_chi", "JC_c0", "JC_Salpha", "JC_Gamma0", "density", "JC_D1", "JC_D2", "JC_D3", "JC_D4", "JC_D5"}},
    };
    return reg;
}
} // namespace

/**
 * @brief Computes Cauchy stress for a linear elastic solid.
 *
 * Implements isotropic Hookean elasticity in 1D, 2D, or 3D depending on
 * AMREX_SPACEDIM. Uses Young’s modulus (E) and Poisson’s ratio (v) to form
 * the constitutive matrix and multiplies it by the strain tensor eps[].
 *
 * @param eps    Strain tensor components
 * @param sigma  Output stress tensor components
 * @param E      Young’s modulus
 * @param v      Poisson’s ratio
 */

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
linear_elastic(amrex::Real eps[NCOMP_TENSOR],
               amrex::Real sigma[NCOMP_TENSOR],
               amrex::Real E,
               amrex::Real v)
{
    [[maybe_unused]] amrex::Real const1 = E / (eka + v) / (eka - two * v);
    [[maybe_unused]] amrex::Real const2 = E / (eka + v);

#if (AMREX_SPACEDIM == 1)
    // 1D case: only XX component matters
    sigma[XX] = E * eps[XX]; // Hooke’s law in 1D
    sigma[YY] = sigma[ZZ] = 0.0;
    sigma[XY] = sigma[XZ] = sigma[YZ] = 0.0;

#elif (AMREX_SPACEDIM == 2)
    // 2D plane stress/strain case
    sigma[XX] = const1 * ((1 - v) * eps[XX] + v * eps[YY]);
    sigma[YY] = const1 * ((1 - v) * eps[YY] + v * eps[XX]);
    sigma[ZZ] = v * (sigma[XX] + sigma[YY]);
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

/**
 * @brief Computes the incremental stress response for linear elasticity.
 *
 * This is the differential form of Hooke’s law: Δσ = C : Δε.
 * Used in implicit or incremental formulations where only the change in
 * strain is known. Behavior depends on AMREX_SPACEDIM.
 *
 * @param delta_eps     Incremental strain tensor
 * @param delta_sigma   Output incremental stress tensor
 * @param E             Young’s modulus
 * @param v             Poisson’s ratio
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
linear_elastic_delta(amrex::Real delta_eps[NCOMP_TENSOR],
                     amrex::Real delta_sigma[NCOMP_TENSOR],
                     amrex::Real E,
                     amrex::Real v)
{
    [[maybe_unused]] amrex::Real const1 = E / (eka + v) / (eka - two * v);
    [[maybe_unused]] amrex::Real const2 = E / (eka + v);

#if (AMREX_SPACEDIM == 1)
    // 1D elasticity: only XX component
    delta_sigma[XX] = E * delta_eps[XX];
    delta_sigma[YY] = delta_sigma[ZZ] = 0.0;
    delta_sigma[XY] = delta_sigma[XZ] = delta_sigma[YZ] = 0.0;

#elif (AMREX_SPACEDIM == 2)
    // 2D elasticity (plane stress assumption)
    delta_sigma[XX] = const1 * ((1 - v) * delta_eps[XX] + v * delta_eps[YY]);
    delta_sigma[YY] = const1 * ((1 - v) * delta_eps[YY] + v * delta_eps[XX]);
    delta_sigma[ZZ] = v * (delta_sigma[XX] + delta_sigma[YY]);
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

/**
 * @brief Computes the Cauchy stress for a Newtonian fluid.
 *
 * Implements σ = 2μ ε̇_dev − p I, where ε̇ is the strain‑rate tensor,
 * μ is the dynamic viscosity, and p is the pressure. Handles 1D, 2D, and 3D
 * cases depending on AMREX_SPACEDIM.
 *
 * @param epsdot    Strain‑rate tensor components
 * @param sigma     Output stress tensor components
 * @param dyn_visc  Dynamic viscosity μ
 * @param pressure  Fluid pressure p
 */
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
    amrex::Real one_by_two = 0.5;

    sigma[XX] =
        2.0 * dyn_visc * (epsdot[XX] - one_by_two * trace_epsdot) - pressure;
    sigma[YY] =
        2.0 * dyn_visc * (epsdot[YY] - one_by_two * trace_epsdot) - pressure;
    sigma[ZZ] = 2 * dyn_visc * (epsdot[ZZ] - trace_epsdot / 3.0) - pressure;
    sigma[XY] = 2.0 * dyn_visc * epsdot[XY];
    sigma[XZ] = sigma[YZ] = 0.0;

#elif (AMREX_SPACEDIM == 3)
    // Full 3D Newtonian fluid
    amrex::Real trace_epsdot = epsdot[XX] + epsdot[YY] + epsdot[ZZ];
    amrex::Real one_by_three = 1.0 / 3.0;

    sigma[XX] =
        2.0 * dyn_visc * (epsdot[XX] - one_by_three * trace_epsdot) - pressure;
    sigma[YY] =
        2.0 * dyn_visc * (epsdot[YY] - one_by_three * trace_epsdot) - pressure;
    sigma[ZZ] =
        2.0 * dyn_visc * (epsdot[ZZ] - one_by_three * trace_epsdot) - pressure;

    sigma[XY] = 2.0 * dyn_visc * epsdot[XY];
    sigma[XZ] = 2.0 * dyn_visc * epsdot[XZ];
    sigma[YZ] = 2.0 * dyn_visc * epsdot[YZ];
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
left_cauchy_green(const amrex::Real F[NCOMP_FULLTENSOR],
                  amrex::Real b[NCOMP_TENSOR])
{
    constexpr int d = AMREX_SPACEDIM; // stride into F, NOT 3

    auto row_dot = [&](int i, int j)
    {
        amrex::Real s = amrex::Real(0.0);
        for (int k = 0; k < d; ++k)
            s += F[i * d + k] * F[j * d + k];
        return s;
    };

    b[XX] = row_dot(0, 0);
#if (AMREX_SPACEDIM >= 2)
    b[YY] = row_dot(1, 1);
    b[XY] = row_dot(0, 1);
#endif
#if (AMREX_SPACEDIM == 3)
    b[ZZ] = row_dot(2, 2);
    b[XZ] = row_dot(0, 2);
    b[YZ] = row_dot(1, 2);
#endif
}

/**
 * @brief Computes Cauchy stress for a neo-hooken solid
 *
 * Implements compressible neo-hookean elasticity in 1D, 2D, or 3D depending on
 * AMREX_SPACEDIM. Uses Young’s modulus (E) and Poisson’s ratio (v) to form
 * the constitutive matrix and multiplies it by the strain tensor eps[].
 *
 * @param eps    Strain tensor components
 * @param sigma  Output stress tensor components
 * @param E      Young’s modulus
 * @param v      Poisson’s ratio
 */

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
neo_hookean(amrex::Real sigma[NCOMP_TENSOR],
            const amrex::Real F[NCOMP_FULLTENSOR],
            amrex::Real E,
            amrex::Real v)
{
    amrex::Real mu = E / (two * (eka + v));                     // shear modulus
    amrex::Real lambda = E * v / ((eka + v) * (eka - two * v)); // first Lamé

    amrex::Real B[NCOMP_TENSOR];
    left_cauchy_green(F, B); // B = F Fᵀ

    // Jacobian
    amrex::Real detF = 0.0;
#if (AMREX_SPACEDIM == 1)
    detF = F[0];
#elif (AMREX_SPACEDIM == 2)
    detF = F[0] * F[3] - F[1] * F[2];
#else
    detF = F[0] * (F[4] * F[8] - F[5] * F[7]) -
           F[1] * (F[3] * F[8] - F[5] * F[6]) +
           F[2] * (F[3] * F[7] - F[4] * F[6]);
#endif
    detF = amrex::max(detF, amrex::Real(1.e-12)); // NaN guard

    const amrex::Real vol = lambda * std::log(detF) / detF; // λ ln(J)/J
    const amrex::Real muJ = mu / detF;                      // μ/J

#if (AMREX_SPACEDIM == 1)
    sigma[XX] = muJ * (B[XX] - 1.0) + vol;
    sigma[YY] = sigma[ZZ] = vol;
    sigma[XY] = sigma[XZ] = sigma[YZ] = 0.0;

#elif (AMREX_SPACEDIM == 2)
    // plane strain: F_zz = 1 implicit, so b_zz = 1 → out-of-plane deviatoric =
    // 0
    sigma[XX] = muJ * (B[XX] - 1.0) + vol;
    sigma[YY] = muJ * (B[YY] - 1.0) + vol;
    sigma[ZZ] = vol; // NOT zero under plane strain
    sigma[XY] = muJ * B[XY];
    sigma[XZ] = sigma[YZ] = 0.0;

#elif (AMREX_SPACEDIM == 3)
    sigma[XX] = muJ * (B[XX] - 1.0) + vol;
    sigma[YY] = muJ * (B[YY] - 1.0) + vol;
    sigma[ZZ] = muJ * (B[ZZ] - 1.0) + vol;
    sigma[XY] = muJ * B[XY];
    sigma[XZ] = muJ * B[XZ];
    sigma[YZ] = muJ * B[YZ];
#endif
}

#endif
