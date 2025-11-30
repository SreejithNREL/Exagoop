// clang-format off
#include <mpm_particle_container.H>
#include <interpolants.H>
#include <constitutive_models.H>
// clang-format on

using namespace amrex;

void MPMParticleContainer::apply_constitutive_model(
    const amrex::Real &dt, amrex::Real applied_strainrate /*=0.0*/)
{
    const int lev = 0;
    auto &plev = GetParticles(lev);

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        int gid = mfi.index();
        int tid = mfi.LocalTileIndex();
        auto index = std::make_pair(gid, tid);

        auto &ptile = plev[index];
        auto &aos = ptile.GetArrayOfStructs();

        int np = aos.numRealParticles();
        int ng = aos.numNeighborParticles();
        int nt = np + ng;

        ParticleType *pstruct = aos().dataPtr();

        amrex::ParallelFor(
            nt,
            [=] AMREX_GPU_DEVICE(int i) noexcept
            {
                ParticleType &p = pstruct[i];
                if (p.idata(intData::phase) == 0)
                {
                    amrex::Real strainrate[NCOMP_TENSOR];
                    amrex::Real strain[NCOMP_TENSOR];
                    amrex::Real stress[NCOMP_TENSOR];

                    // Update strain from strainrate
                    for (int d = 0; d < NCOMP_TENSOR; ++d)
                    {
                        p.rdata(realData::strain + d) +=
                            dt * p.rdata(realData::strainrate + d);
                    }

                // Apply external axial strainrate in active spatial dims
                // Assumes XX, YY, ZZ are defined consistently with
                // AMREX_SPACEDIM
#if (AMREX_SPACEDIM >= 1)
                    p.rdata(realData::strain + XX) += dt * applied_strainrate;
#endif
#if (AMREX_SPACEDIM >= 2)
                    p.rdata(realData::strain + YY) += dt * applied_strainrate;
#endif
#if (AMREX_SPACEDIM >= 3)
                    p.rdata(realData::strain + ZZ) += dt * applied_strainrate;
#endif

                    // Copy strain/strainrate into local arrays
                    for (int d = 0; d < NCOMP_TENSOR; ++d)
                    {
                        strainrate[d] = p.rdata(realData::strainrate + d);
                        strain[d] = p.rdata(realData::strain + d);
                    }

                    if (p.idata(intData::constitutive_model) == 0)
                    {
                        // Elastic solid
                        linear_elastic(strain, strainrate, stress,
                                       p.rdata(realData::E),
                                       p.rdata(realData::nu));
                    }
                    else if (p.idata(intData::constitutive_model) == 1)
                    {
                        // Viscous fluid with approximate EoS
                        const amrex::Real p_inf = 0.0;
                        p.rdata(realData::pressure) =
                            p.rdata(realData::Bulk_modulus) *
                                (std::pow(1.0 / p.rdata(realData::jacobian),
                                          p.rdata(realData::Gama_pressure)) -
                                 1.0) +
                            p_inf;



                        Newtonian_Fluid(strainrate, stress,
                                        p.rdata(realData::Dynamic_viscosity),
                                        p.rdata(realData::pressure));
                    }

                    // Write back stress
                    for (int d = 0; d < NCOMP_TENSOR; ++d)
                    {
                        p.rdata(realData::stress + d) = stress[d];
                    }
                }
            });
    }
}

void MPMParticleContainer::apply_constitutive_model_delta(
    const amrex::Real &dt, amrex::Real applied_strainrate /*= 0.0*/)
{
    const int lev = 0;
    auto &plev = GetParticles(lev);

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        const int gid = mfi.index();
        const int tid = mfi.LocalTileIndex();
        auto index = std::make_pair(gid, tid);

        auto &ptile = plev[index];
        auto &aos = ptile.GetArrayOfStructs();

        const int np = aos.numRealParticles();
        const int ng = aos.numNeighborParticles();
        const int nt = np + ng;

        ParticleType *pstruct = aos().dataPtr();

        amrex::ParallelFor(
            nt,
            [=] AMREX_GPU_DEVICE(int i) noexcept
            {
                ParticleType &p = pstruct[i];

                if (p.idata(intData::phase) == 0)
                {
                    amrex::Real delta_strain[NCOMP_TENSOR];
                    amrex::Real delta_stress[NCOMP_TENSOR];

                    // Accumulate strain from current strainrate
                    for (int c = 0; c < NCOMP_TENSOR; ++c)
                    {
                        p.rdata(realData::strain + c) +=
                            dt * p.rdata(realData::strainrate + c);
                    }

                // Apply external axial strainrate in active spatial dims
#if (AMREX_SPACEDIM >= 1)
                    p.rdata(realData::strain + XX) += dt * applied_strainrate;
#endif
#if (AMREX_SPACEDIM >= 2)
                    p.rdata(realData::strain + YY) += dt * applied_strainrate;
#endif
#if (AMREX_SPACEDIM >= 3)
                    p.rdata(realData::strain + ZZ) += dt * applied_strainrate;
#endif

                    // Build delta_strain from strainrate
                    for (int c = 0; c < NCOMP_TENSOR; ++c)
                    {
                        delta_strain[c] =
                            dt * p.rdata(realData::strainrate + c);
                    }

                // Add external axial delta in active dims
#if (AMREX_SPACEDIM >= 1)
                    delta_strain[XX] += dt * applied_strainrate;
#endif
#if (AMREX_SPACEDIM >= 2)
                    delta_strain[YY] += dt * applied_strainrate;
#endif
#if (AMREX_SPACEDIM >= 3)
                    delta_strain[ZZ] += dt * applied_strainrate;
#endif

                    // Constitutive response for delta update
                    if (p.idata(intData::constitutive_model) == 0)
                    {
                        // Elastic solid: linear operator on delta_strain
                        linear_elastic(delta_strain, delta_stress,
                                       p.rdata(realData::E),
                                       p.rdata(realData::nu));
                    }
                    else if (p.idata(intData::constitutive_model) == 1)
                    {
                        amrex::Abort(
                            "\nDelta strain model for weakly compressible "
                            "fluids not implemented yet.");
                    }

                    // Accumulate stress with delta contribution
                    for (int c = 0; c < NCOMP_TENSOR; ++c)
                    {
                        p.rdata(realData::stress + c) += delta_stress[c];
                    }
                }
            });
    }
}
