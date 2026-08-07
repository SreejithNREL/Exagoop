#ifndef CONSTITUTIVE_MODELS_H_
#define CONSTITUTIVE_MODELS_H_
// clang-format off
#include<constants.H>
#include <AMReX.H>
// clang-format on

// ============================================================
// Material-parameter table (ADR-0001, Phase 1).
//
// Material *parameters* are constant per material and are stored ONCE per
// constitutive-model instance in a small device-resident table, indexed by the
// particle's intData::constitutive_model id -- not duplicated on every particle.
// Each model interprets its own parameter slots (see the *P namespaces below).
// ============================================================
constexpr int MAX_MODEL_PARAMS = 24;

// Constitutive-model type ids (must match the cm_id / intData::constitutive_model
// convention used in the input: 0 = elastic, 1 = Newtonian fluid,
// 2 = Johnson-Cook).
namespace CModel
{
enum
{
    ELASTIC = 0,
    FLUID = 1,
    JOHNSON_COOK = 2
};
}

// Per-model parameter slot layout within MaterialParams::p[].
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
    count
};
}
// Johnson-Cook (+ Mie-Gruneisen EOS) parameter slots.
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
    // Johnson-Cook damage parameters (JC-3). All zero -> damage disabled.
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

// POD, trivially copyable -> safe to place in a Gpu::DeviceVector and capture
// by pointer in device kernels.
struct MaterialParams
{
    int model = CModel::ELASTIC;			//Elastic by default
    amrex::Real p[MAX_MODEL_PARAMS] = {0.0};
};

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
    sigma[ZZ] = 0.0; // out-of-plane stress often set to 0 in plane stress
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
    delta_sigma[ZZ] = 0.0;
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
    sigma[ZZ] = 0.0;
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

// ============================================================
// Johnson-Cook elasto-plasticity + Mie-Gruneisen EOS
// (ported from the JCModel branch; Nguyen et al. 2023, Sec 4.3).
// ============================================================

/**
 * @brief Polar decomposition F = R U via Newton iteration (row-major 3x3).
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
polar_decomp_3x3(const amrex::Real F[9], amrex::Real R[9])
{
    amrex::Real Q[9];
    amrex::Real Qinv[9];

    amrex::Real normF = amrex::Real(0.0);
    for (int k = 0; k < 9; ++k) normF += F[k] * F[k];
    normF = std::sqrt(normF);
    amrex::Real inv_normF = (normF > amrex::Real(0.0)) ?
                            amrex::Real(1.0) / normF : amrex::Real(1.0);
    for (int k = 0; k < 9; ++k) Q[k] = F[k] * inv_normF;

    for (int iter = 0; iter < 20; ++iter)
    {
        amrex::Real det =
            Q[0] * (Q[4]*Q[8] - Q[5]*Q[7]) -
            Q[1] * (Q[3]*Q[8] - Q[5]*Q[6]) +
            Q[2] * (Q[3]*Q[7] - Q[4]*Q[6]);

        amrex::Real inv_det = (std::abs(det) > amrex::Real(1e-30)) ?
                              amrex::Real(1.0) / det : amrex::Real(0.0);

        Qinv[0] =  (Q[4]*Q[8] - Q[5]*Q[7]) * inv_det;
        Qinv[1] = -(Q[1]*Q[8] - Q[2]*Q[7]) * inv_det;
        Qinv[2] =  (Q[1]*Q[5] - Q[2]*Q[4]) * inv_det;
        Qinv[3] = -(Q[3]*Q[8] - Q[5]*Q[6]) * inv_det;
        Qinv[4] =  (Q[0]*Q[8] - Q[2]*Q[6]) * inv_det;
        Qinv[5] = -(Q[0]*Q[5] - Q[2]*Q[3]) * inv_det;
        Qinv[6] =  (Q[3]*Q[7] - Q[4]*Q[6]) * inv_det;
        Qinv[7] = -(Q[0]*Q[7] - Q[1]*Q[6]) * inv_det;
        Qinv[8] =  (Q[0]*Q[4] - Q[1]*Q[3]) * inv_det;

        amrex::Real err = amrex::Real(0.0);
        amrex::Real Qnew[9];
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                amrex::Real val = amrex::Real(0.5) *
                                  (Q[r*3+c] + Qinv[c*3+r]);
                Qnew[r*3+c] = val;
                err += (val - Q[r*3+c]) * (val - Q[r*3+c]);
            }
        }
        for (int k = 0; k < 9; ++k) Q[k] = Qnew[k];
        if (err < amrex::Real(1e-24)) break;
    }

    for (int k = 0; k < 9; ++k) R[k] = Q[k];
}

/**
 * @brief Mie-Gruneisen equation of state (compression-positive pressure).
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
mie_gruneisen_eos(amrex::Real eta, amrex::Real rho0, amrex::Real c0,
                  amrex::Real Salpha, amrex::Real Gamma0, amrex::Real e)
{
    amrex::Real phat;
    if (eta >= amrex::Real(1.0))
    {
        amrex::Real denom = eta - Salpha * (eta - amrex::Real(1.0));
        denom = denom * denom;
        if (std::abs(denom) < amrex::Real(1e-30))
            denom = amrex::Real(1e-30);
        phat = rho0 * c0 * c0 * (eta - amrex::Real(1.0)) *
               (eta - amrex::Real(0.5) * Gamma0 * (eta - amrex::Real(1.0))) /
               denom + Gamma0 * e;
    }
    else
    {
        phat = rho0 * c0 * c0 * (eta - amrex::Real(1.0)) + Gamma0 * e;
    }
    return phat;
}

/**
 * @brief Johnson-Cook elasto-plastic stress update (hypoelastic, un-rotated
 *        config; von Mises + radial return; Mie-Gruneisen pressure; adiabatic
 *        heating). sdev_ur and ep are per-particle state (ISV), updated in
 *        place. sigma uses tensile-positive convention.
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
johnson_cook_stress_update(
    const amrex::Real F[9],
    const amrex::Real D_voigt[NCOMP_TENSOR],
    amrex::Real       sdev_ur[NCOMP_TENSOR],
    amrex::Real      &ep,
    amrex::Real       sigma[NCOMP_TENSOR],
    amrex::Real      &pressure,
    amrex::Real      &heat_src_out,
    amrex::Real rho,       amrex::Real rho0,
    amrex::Real E,         amrex::Real nu,
    amrex::Real JC_A,      amrex::Real JC_B,  amrex::Real JC_n,
    amrex::Real JC_C,      amrex::Real JC_m,
    amrex::Real eps_dot_0,
    amrex::Real T,         amrex::Real Tr,    amrex::Real Tm,
    amrex::Real chi,
    amrex::Real c0,        amrex::Real Salpha, amrex::Real Gamma0,
    amrex::Real Dd1,       amrex::Real Dd2,   amrex::Real Dd3,
    amrex::Real Dd4,       amrex::Real Dd5,
    amrex::Real       &damage,
    amrex::Real dt)
{
    heat_src_out = amrex::Real(0.0);

    amrex::Real G = E / (amrex::Real(2.0) * (amrex::Real(1.0) + nu));

    amrex::Real D3[3][3];
    D3[0][0] = D_voigt[XX]; D3[1][1] = D_voigt[YY]; D3[2][2] = D_voigt[ZZ];
    D3[0][1] = D3[1][0] = D_voigt[XY];
    D3[0][2] = D3[2][0] = D_voigt[XZ];
    D3[1][2] = D3[2][1] = D_voigt[YZ];

    amrex::Real R[9];
    polar_decomp_3x3(F, R);

    amrex::Real d[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    d[i][j] += R[k*3+i] * D3[k][l] * R[l*3+j];

    amrex::Real tr_d = d[0][0] + d[1][1] + d[2][2];
    amrex::Real one_third = amrex::Real(1.0) / amrex::Real(3.0);
    amrex::Real dd[NCOMP_TENSOR];
    dd[XX] = d[0][0] - one_third * tr_d;
    dd[YY] = d[1][1] - one_third * tr_d;
    dd[ZZ] = d[2][2] - one_third * tr_d;
    dd[XY] = d[0][1];
    dd[XZ] = d[0][2];
    dd[YZ] = d[1][2];

    amrex::Real strial[NCOMP_TENSOR];
    for (int c = 0; c < NCOMP_TENSOR; ++c)
        strial[c] = sdev_ur[c] + amrex::Real(2.0) * G * dt * dd[c];

    amrex::Real seq2 =
        strial[XX]*strial[XX] + strial[YY]*strial[YY] + strial[ZZ]*strial[ZZ]
        + amrex::Real(2.0) * (strial[XY]*strial[XY] +
                              strial[XZ]*strial[XZ] +
                              strial[YZ]*strial[YZ]);
    amrex::Real seq_trial = std::sqrt(amrex::Real(1.5) * seq2);

    amrex::Real strain_fac =
        JC_A + JC_B * std::pow(amrex::max(ep, amrex::Real(0.0)), JC_n);

    amrex::Real rate_fac = amrex::Real(1.0);
    if (JC_C > amrex::Real(0.0) && eps_dot_0 > amrex::Real(0.0))
    {
        amrex::Real epd_norm = seq_trial /
            (amrex::Real(3.0) * G * dt * eps_dot_0);
        epd_norm = amrex::max(epd_norm, amrex::Real(1.0));
        rate_fac = amrex::Real(1.0) + JC_C * std::log(epd_norm);
    }

    amrex::Real temp_fac = amrex::Real(1.0);
    if (JC_m > amrex::Real(0.0) && (Tm - Tr) > amrex::Real(0.0))
    {
        amrex::Real Tstar = (T - Tr) / (Tm - Tr);
        Tstar = amrex::max(amrex::Real(0.0),
                amrex::min(amrex::Real(1.0), Tstar));
        temp_fac = amrex::max(amrex::Real(0.0),
                   amrex::Real(1.0) - std::pow(Tstar, JC_m));
    }

    amrex::Real sigma_f = amrex::max(
        strain_fac * rate_fac * temp_fac, amrex::Real(0.0));

    amrex::Real dep = amrex::Real(0.0);   // plastic strain increment this step
    amrex::Real ep_dot = amrex::Real(0.0);
    if (seq_trial <= sigma_f || seq_trial < amrex::Real(1e-30))
    {
        for (int c = 0; c < NCOMP_TENSOR; ++c) sdev_ur[c] = strial[c];
    }
    else
    {
        dep = amrex::max(
            (seq_trial - sigma_f) / (amrex::Real(3.0) * G), amrex::Real(0.0));
        ep += dep;
        amrex::Real scale = sigma_f / seq_trial;
        for (int c = 0; c < NCOMP_TENSOR; ++c)
            sdev_ur[c] = scale * strial[c];
        ep_dot = (dt > amrex::Real(0.0)) ? dep / dt : amrex::Real(0.0);
        heat_src_out = chi * sigma_f * ep_dot;
    }

    amrex::Real eta = (rho0 > amrex::Real(0.0)) ?
                      rho / rho0 : amrex::Real(1.0);
    pressure = mie_gruneisen_eos(eta, rho0, c0, Salpha, Gamma0,
                                 amrex::Real(0.0));

    // --- Johnson-Cook damage (JC-3) ---------------------------------------
    // Opt-in: only active when the failure model is parameterised (D1 or D2).
    // Failure strain  eps_f = [D1 + D2 exp(D3 sigma*)][1 + D4 ln(edot*)]
    //                          [1 + D5 T*], with triaxiality sigma* = sigma_m/svm.
    // Once accumulated damage D = sum(dep/eps_f) reaches 1 the particle fails:
    // deviatoric stress is removed and only compressive pressure is retained,
    // so failed material separates instead of carrying tension/shear.
    if ((Dd1 > amrex::Real(0.0) || Dd2 > amrex::Real(0.0)) &&
        damage < amrex::Real(1.0))
    {
        if (dep > amrex::Real(0.0))
        {
            // von Mises after return ~ sigma_f (flow stress); guard tiny values.
            amrex::Real svm = amrex::max(sigma_f, amrex::Real(1e-12));
            // Mean (tensile-positive) stress = -pressure (pressure is
            // compression-positive here). Triaxiality sigma* = sigma_m/svm.
            amrex::Real triax = (-pressure) / svm;
            amrex::Real fac_t = Dd1 + Dd2 * std::exp(Dd3 * triax);

            amrex::Real fac_r = amrex::Real(1.0);
            if (Dd4 != amrex::Real(0.0) && eps_dot_0 > amrex::Real(0.0))
            {
                amrex::Real edn = amrex::max(ep_dot / eps_dot_0, amrex::Real(1e-12));
                fac_r = amrex::Real(1.0) + Dd4 * std::log(edn);
            }
            amrex::Real fac_T = amrex::Real(1.0);
            if (Dd5 != amrex::Real(0.0) && (Tm - Tr) > amrex::Real(0.0))
            {
                amrex::Real Ts = (T - Tr) / (Tm - Tr);
                Ts = amrex::max(amrex::Real(0.0), amrex::min(amrex::Real(1.0), Ts));
                fac_T = amrex::Real(1.0) + Dd5 * Ts;
            }
            amrex::Real eps_f = fac_t * fac_r * fac_T;
            if (eps_f > amrex::Real(1e-12))
                damage += dep / eps_f;
        }
    }

    if (damage >= amrex::Real(1.0))
    {
        damage = amrex::Real(1.0);
        for (int c = 0; c < NCOMP_TENSOR; ++c) sdev_ur[c] = amrex::Real(0.0);
        // Retain compression only (no tensile stress) so chips separate.
        if (pressure < amrex::Real(0.0)) pressure = amrex::Real(0.0);
    }

    amrex::Real sigma_ur[3][3];
    sigma_ur[0][0] = sdev_ur[XX] - pressure;
    sigma_ur[1][1] = sdev_ur[YY] - pressure;
    sigma_ur[2][2] = sdev_ur[ZZ] - pressure;
    sigma_ur[0][1] = sigma_ur[1][0] = sdev_ur[XY];
    sigma_ur[0][2] = sigma_ur[2][0] = sdev_ur[XZ];
    sigma_ur[1][2] = sigma_ur[2][1] = sdev_ur[YZ];

    amrex::Real sigma3[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    sigma3[i][j] += R[i*3+k] * sigma_ur[k][l] * R[j*3+l];

    sigma[XX] = sigma3[0][0];
    sigma[YY] = sigma3[1][1];
    sigma[ZZ] = sigma3[2][2];
    sigma[XY] = sigma3[0][1];
    sigma[XZ] = sigma3[0][2];
    sigma[YZ] = sigma3[1][2];
}

#endif
