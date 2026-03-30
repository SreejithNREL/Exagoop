// clang-format off
#include <mpm_particle_container.H>
#include <interpolants.H>
// clang-format on

/**
 * @brief Computes the total kinetic and strain energies of all material points.
 *
 * Performs a parallel reduction over all particles to compute:
 *  - Total kinetic energy:  TKE = 0.5 * m * |v|²
 *  - Total strain energy:   TSE = 0.5 * V * (σ : ε)
 *
 * The function launches two AMReX reductions, one for kinetic energy and one
 * for strain energy. The results are summed across MPI ranks if applicable.
 *
 * @param[out] TKE  Total kinetic energy of all particles (Joules)
 * @param[out] TSE  Total strain energy of all particles (Joules)
 *
 * @note This routine does not modify particle data. It only reads particle
 *       mass, velocity, stress, strain, and volume.
 */

void MPMParticleContainer::Calculate_Total_Energies(Real &TKE, Real &TSE)
{
    TKE = 0.0;
    TSE = 0.0;

    using PType = typename MPMParticleContainer::SuperParticleType;

    TKE = amrex::ReduceSum(*this,
                           [] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                           {
                               Real v2 = 0.0;
                               for (int d = 0; d < AMREX_SPACEDIM; ++d)
                               {
                                   v2 += p.rdata(realData::xvel + d) *
                                         p.rdata(realData::xvel + d);
                               }
                               return 0.5 * p.rdata(realData::mass) * v2;
                           });

    TSE = amrex::ReduceSum(*this,
                           [] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                           {
                               Real se = 0.0;

#if (AMREX_SPACEDIM == 1)
                               se = p.rdata(realData::stress + XX) *
                                    p.rdata(realData::strain + XX);

#elif (AMREX_SPACEDIM == 2)
            se = p.rdata(realData::stress + XX) * p.rdata(realData::strain + XX)
               + p.rdata(realData::stress + YY) * p.rdata(realData::strain + YY)
               + 2.0 * p.rdata(realData::stress + XY) * p.rdata(realData::strain + XY);

#elif (AMREX_SPACEDIM == 3)
            se = p.rdata(realData::stress + XX) * p.rdata(realData::strain + XX)
               + p.rdata(realData::stress + YY) * p.rdata(realData::strain + YY)
               + p.rdata(realData::stress + ZZ) * p.rdata(realData::strain + ZZ)
               + 2.0 * (p.rdata(realData::stress + XY) * p.rdata(realData::strain + XY)
                      + p.rdata(realData::stress + YZ) * p.rdata(realData::strain + YZ)
                      + p.rdata(realData::stress + XZ) * p.rdata(realData::strain + XZ));
#endif

                               return 0.5 * p.rdata(realData::volume) * se;
                           });

#ifdef BL_USE_MPI
    ParallelDescriptor::ReduceRealSum(TKE);
    ParallelDescriptor::ReduceRealSum(TSE);
#endif
}

/**
 * @brief Computes the mass‑weighted average velocity components of all
 * particles.
 *
 * Performs AMReX reductions to compute:
 *   momentum_tot[d] = Σ (m * v_d)
 *   mass_tot        = Σ m
 *
 * The center‑of‑mass velocity is then:
 *   Vcm[d] = momentum_tot[d] / mass_tot
 *
 * @param[out] Vcm  Array of size AMREX_SPACEDIM containing the mass‑weighted
 *                  average velocity components.
 *
 * @note MPI reductions are performed when running in parallel.
 */

void MPMParticleContainer::Calculate_MWA_VelocityComponents(
    amrex::GpuArray<Real, AMREX_SPACEDIM> &Vcm)
{
    using PType = typename MPMParticleContainer::SuperParticleType;

    amrex::GpuArray<Real, AMREX_SPACEDIM> momentum_tot;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        momentum_tot[d] = 0.0;
    }
    Real mass_tot = 0.0;

    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        momentum_tot[d] = amrex::ReduceSum(
            *this, [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
            { return p.rdata(realData::mass) * p.rdata(realData::xvel + d); });
    }

    mass_tot =
        amrex::ReduceSum(*this, [] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                         { return p.rdata(realData::mass); });

#ifdef BL_USE_MPI
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        ParallelDescriptor::ReduceRealSum(momentum_tot[d]);
    }
    ParallelDescriptor::ReduceRealSum(mass_tot);
#endif

    // Mass-weighted average velocity components
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        Vcm[d] = momentum_tot[d] / mass_tot;
    }
}

/**
 * @brief Computes the mass‑weighted average velocity magnitude of all
 * particles.
 *
 * Performs reductions to compute:
 *   massvelmag = Σ (m * |v|)
 *   mass_tot   = Σ m
 *
 * The mass‑weighted average velocity magnitude is:
 *   Vcm = massvelmag / mass_tot
 *
 * @param[out] Vcm  Scalar mass‑weighted average velocity magnitude.
 *
 * @note Uses AMReX parallel reductions and MPI reductions when enabled.
 */

void MPMParticleContainer::Calculate_MWA_VelocityMagnitude(amrex::Real &Vcm)
{
    using PType = typename MPMParticleContainer::SuperParticleType;

    Real massvelmag = 0.0;
    Real mass_tot = 0.0;

    massvelmag =
        amrex::ReduceSum(*this,
                         [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                         {
                             Real vmag = 0.0;
                             for (int d = 0; d < AMREX_SPACEDIM; ++d)
                             {
                                 vmag += p.rdata(realData::xvel + d) *
                                         p.rdata(realData::xvel + d);
                             }
                             return p.rdata(realData::mass) * sqrt(vmag);
                         });

    mass_tot =
        amrex::ReduceSum(*this, [] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                         { return p.rdata(realData::mass); });

#ifdef BL_USE_MPI
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        ParallelDescriptor::ReduceRealSum(massvelmag);
    }
    ParallelDescriptor::ReduceRealSum(mass_tot);
#endif

    Vcm = massvelmag / mass_tot;
}

/**
 * @brief Computes the minimum and maximum particle positions in each dimension.
 *
 * Performs AMReX ReduceMin and ReduceMax operations over all particles to find
 * the bounding box of the particle cloud.
 *
 * @param[out] minpos  Minimum particle position per dimension.
 * @param[out] maxpos  Maximum particle position per dimension.
 *
 * @note This function does not modify particle data.
 */

void MPMParticleContainer::Calculate_MinMaxPos(
    amrex::GpuArray<Real, AMREX_SPACEDIM> &minpos,
    amrex::GpuArray<Real, AMREX_SPACEDIM> &maxpos)
{
    using PType = typename MPMParticleContainer::SuperParticleType;

    for (int dim = 0; dim < AMREX_SPACEDIM; dim++)
    {
        minpos[dim] = amrex::ReduceMin(
            *this, [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
            { return p.pos(dim); });
    }

    for (int dim = 0; dim < AMREX_SPACEDIM; dim++)
    {
        maxpos[dim] = amrex::ReduceMax(
            *this, [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
            { return p.pos(dim); });
    }

#ifdef BL_USE_MPI
    // for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        amrex::ParallelDescriptor::ReduceRealMax(maxpos.data(), AMREX_SPACEDIM);
        amrex::ParallelDescriptor::ReduceRealMin(minpos.data(), AMREX_SPACEDIM);
        // amrex::ParallelDescriptor::ReduceRealMax(minpos);
    }
#endif
}

/**
 * @brief Estimates the normal contact force at the top and bottom surfaces.
 *
 * Computes Fy_top = Σ(m * a_y) + |Σ(m * g_y)| + |Fy_bottom| using a
 * GPU‑parallel reduction over all particles. The vertical acceleration
 * a_y is read from @c realData::yacceleration and gravity from the
 * passed‑in array. Fy_bottom is set to zero (placeholder for a future
 * boundary integral).
 *
 * @param[in]  gravity     Gravitational acceleration vector.
 * @param[out] Fy_top      Estimated upward reaction force at the top surface.
 * @param[out] Fy_bottom   Estimated downward reaction force at the bottom
 *                         surface (currently always zero).
 *
 * @return None.
 */
void MPMParticleContainer::CalculateSurfaceIntegralTop(
    Array<Real, AMREX_SPACEDIM> gravity, Real &Fy_top, Real &Fy_bottom)
{
    Real Mvy = 0.0;
    Real Fg = 0.0;
    Fy_bottom = 0.0;

    using PType = typename MPMParticleContainer::SuperParticleType;
    Mvy = amrex::ReduceSum(*this,
                           [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                           {
                               return (p.rdata(realData::mass) *
                                       p.rdata(realData::yacceleration));
                           });

    Fg = amrex::ReduceSum(
        *this, [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
        { return (p.rdata(realData::mass) * gravity[YDIR]); });

#ifdef BL_USE_MPI
    ParallelDescriptor::ReduceRealSum(Mvy);
    ParallelDescriptor::ReduceRealSum(Fg);
#endif

    Fy_top = Mvy + fabs(Fg) + fabs(Fy_bottom);
}
