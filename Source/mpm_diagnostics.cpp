// clang-format off
#include <mpm_particle_container.H>
#include <interpolants.H>
// clang-format on

/**
 * @brief Calculates the total kinetic and strain energies of all material
 * points in the domain at a given time
 *
 * Performs a reduction over all particles to compute:
 * - TKE = 0.5 * m * v²
 * - TSE = 0.5 * V * (σ:ε)
 *
 * @param[out] TKE Total kinetic energy of all particles (units: Joules).
 * @param[out] TSE Total strain energy of all particles (units: Joules).
 *
 * @note This function loops over all particles and performs a parallel
 * reduction.
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
    /*
        tmpExtremasFile.open(
                tempFileName.c_str(),
                std::ios::out | std::ios::app | std::ios_base::binary);
              tmpExtremasFile.precision(12);
              tmpExtremasFile << "iter,time";
        std::ofstream tmpExtremasFile;

        tmpStateFile << m_nstep << "," << m_cur_time << "," << m_dt // Time
                       << "," << kinenergy_int                        // Kinetic
       energy
                       << "," << enstrophy_int                        //
       Enstrophy
                       << "," << m_pNew             // Thermo. pressure
                       << "," << fuelConsumptionInt // Integ fuel burning rate
                       << "," << heatReleaseRateInt // Integ heat release rate
                       << "\n";
          tmpStateFile.flush();

          // Get min/max for state components
          auto stateMax =
            (m_incompressible) != 0
              ? MLmax(GetVecOfConstPtrs(getStateVect(AmrNewTime)), 0,
       AMREX_SPACEDIM) : MLmax(GetVecOfConstPtrs(getStateVect(AmrNewTime)), 0,
       NVAR); auto stateMin = (m_incompressible) != 0 ?
       MLmin(GetVecOfConstPtrs(getStateVect(AmrNewTime)), 0, AMREX_SPACEDIM) :
       MLmin(GetVecOfConstPtrs(getStateVect(AmrNewTime)), 0, NVAR);

          tmpExtremasFile << m_nstep << "," << m_cur_time; // Time
          for (int n = 0; n < stateMax.size();
               ++n) { // Min & max of each state variable
            tmpExtremasFile << "," << stateMin[n] << "," << stateMax[n];
          }
          tmpExtremasFile << "\n";
          tmpExtremasFile.flush();
          tmpExtremasFile.flush();
          tmpExtremasFile.close();*/
}

/**
 * @brief Calculates the mass weighted average of all velocity components
 *
 * Performs a reduction over all particles to compute:
 * - momentum_tot = the total momentum in dimension d
 * - mass_tot = total mass of material points
 *
 * @param[out] Vcm Vector of mass weighted averaged (MWA) material point
 * velocities
 *
 * @note This function loops over all particles and performs a parallel
 * reduction.
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
 * @brief Calculates the mass weighted average of velocity magnitude of material
 * points
 *
 * Performs a reduction over all particles to compute:
 * - massvelmag = the total momentum magnitude of mps
 * - mass_tot = total mass of material points
 *
 * @param[out] Vcm scalar value of mass weighted averaged (MWA) material point
 * velocity magnitude
 *
 * @note This function loops over all particles and performs a parallel
 * reduction.
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
 * @brief Calculates the mass weighted average of velocity magnitude of material
 * points
 *
 * Performs a reduction over all particles to compute:
 * - massvelmag = the total momentum magnitude of mps
 * - mass_tot = total mass of material points
 *
 * @param[out] Vcm scalar value of mass weighted averaged (MWA) material point
 * velocity magnitude
 *
 * @note This function loops over all particles and performs a parallel
 * reduction.
 */

void MPMParticleContainer::Calculate_MinMaxPos(amrex::GpuArray<Real, AMREX_SPACEDIM> &minpos,amrex::GpuArray<Real, AMREX_SPACEDIM> &maxpos)
{
    using PType = typename MPMParticleContainer::SuperParticleType;

    for(int dim=0;dim<AMREX_SPACEDIM;dim++)
      {
	minpos[dim] =
	        amrex::ReduceMin(*this,
	                         [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
	                         {

	                             return p.pos(dim);
	                         });
      }

    for(int dim=0;dim<AMREX_SPACEDIM;dim++)
          {
    	maxpos[dim] =
    	        amrex::ReduceMax(*this,
    	                         [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
    	                         {

    	                             return p.pos(dim);
    	                         });
          }
}

void MPMParticleContainer::CalculateSurfaceIntegralTop(
    Array<Real, AMREX_SPACEDIM> gravity, Real &Fy_top, Real &Fy_bottom)
{
    // const int lev = 0;
    // const Geometry &geom = Geom(lev);
    /*auto &plev = GetParticles(lev);
    const auto dxi = geom.InvCellSizeArray();
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto domain = geom.Domain();*/

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

amrex::Real
MPMParticleContainer::CalculateEffectiveSpringConstant(amrex::Real Area,
                                                       amrex::Real L0)
{
    // First calculate the total strain energy
    // const int lev = 0;
    // const Geometry &geom = Geom(lev);
    /*auto &plev = GetParticles(lev);
    const auto dxi = geom.InvCellSizeArray();
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto domain = geom.Domain();*/

    amrex::Real TSE = 0.0;
    amrex::Real Total_vol = 0.0;
    amrex::Real deflection = 0.0;
    amrex::Real Restoring_force = 0.0;
    amrex::Real smallval = 1e-10;
    amrex::Real Calculated_Spring_Const = 0.0;

    using PType = typename MPMParticleContainer::SuperParticleType;

    TSE = amrex::ReduceSum(*this,
                           [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                           {
                               return (
                                   0.5 * p.rdata(realData::volume) *
                                   (p.rdata(realData::stress + XX) *
                                        p.rdata(realData::strain + XX) +
                                    p.rdata(realData::stress + YY) *
                                        p.rdata(realData::strain + YY) +
                                    p.rdata(realData::stress + ZZ) *
                                        p.rdata(realData::strain + ZZ) +
                                    p.rdata(realData::stress + XY) *
                                        p.rdata(realData::strain + XY) * 2.0 +
                                    p.rdata(realData::stress + YZ) *
                                        p.rdata(realData::strain + YZ) * 2.0 +
                                    p.rdata(realData::stress + XZ) *
                                        p.rdata(realData::strain + XZ) * 2.0));
                           });

#ifdef BL_USE_MPI
    ParallelDescriptor::ReduceRealSum(TSE);
#endif

    // Then Calculate the total volume at this instant
    Total_vol = amrex::ReduceSum(
        *this, [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
        { return (p.rdata(realData::volume)); });

    // Calculate the deflection
    deflection = L0 - Total_vol / Area;

    // Calculate the spring constant
    if (fabs(deflection) <= smallval)
    {
        Restoring_force = 0.0;
    }
    else
    {
        Restoring_force = 2 * TSE / deflection;
        Calculated_Spring_Const = 2 * TSE / (deflection * deflection);
    }

    PrintToFile("SpringConst.out") << Calculated_Spring_Const << "\n";

    // Calculate and return the restoring force
    return (Restoring_force);
}
