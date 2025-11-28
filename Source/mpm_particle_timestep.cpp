// clang-format off
#include <mpm_particle_container.H>
#include <interpolants.H>
#include <mpm_eb.H>
#include <mpm_kernels.H>
#include <aesthetics.H>
// clang-format on

amrex::Real MPMParticleContainer::Calculate_time_step(MPMspecs &specs)
{
    if (specs.fixed_timestep == 1)
    {
        return specs.timestep;
    }

    const int lev = 0;
    const Geometry &geom = Geom(lev);
    const auto dx = geom.CellSizeArray();

    using PType = typename MPMParticleContainer::SuperParticleType;
    amrex::Real dt = amrex::ReduceMin(
        *this,
        [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> amrex::Real
        {
            if (p.idata(intData::phase) == 0)
            {
                amrex::Real Cs = 0.0;
                if (p.idata(intData::constitutive_model) == 1)
                {
                    Cs = std::sqrt(p.rdata(realData::Bulk_modulus) /
                                   p.rdata(realData::density));
                }
                else if (p.idata(intData::constitutive_model) == 0)
                {
                    amrex::Real lambda = p.rdata(realData::E) *
                                         p.rdata(realData::nu) /
                                         ((1 + p.rdata(realData::nu)) *
                                          (1 - 2.0 * p.rdata(realData::nu)));
                    amrex::Real mu = p.rdata(realData::E) /
                                     (2.0 * (1 + p.rdata(realData::nu)));
                    Cs = std::sqrt((lambda + 2.0 * mu) /
                                   p.rdata(realData::density));
                }

                // Dimension‑aware velocity magnitude
                amrex::Real velmag = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    velmag += p.rdata(realData::xvel + d) *
                              p.rdata(realData::xvel + d);
                }
                velmag = std::sqrt(velmag);

                // Minimum cell size across dimensions
                amrex::Real dxmin = dx[0];
                for (int d = 1; d < AMREX_SPACEDIM; ++d)
                {
                    dxmin = amrex::min(dxmin, dx[d]);
                }

                return dxmin / (Cs + velmag);
            }
            return std::numeric_limits<amrex::Real>::max();
        });

#ifdef BL_USE_MPI
    ParallelDescriptor::ReduceRealMin(dt);
#endif

    dt = specs.CFL * dt;
    dt = std::max(std::min(dt, specs.dt_max_limit), specs.dt_min_limit);
    return dt;
}


void MPMParticleContainer::updateVolume()
{
    BL_PROFILE("MPMParticleContainer::updateVolume");
    const int lev = 0;
    auto &plev = GetParticles(lev);

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        auto &ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto &aos = ptile.GetArrayOfStructs();
        const size_t np = aos.numParticles();
        ParticleType *pstruct = aos().dataPtr();

        amrex::ParallelFor(
            np,
            [=] AMREX_GPU_DEVICE(int i) noexcept
            {
                ParticleType &p = pstruct[i];
                if (p.idata(intData::phase) == 0)
                {
                    // Build deformation gradient matrix
                    amrex::Real F[AMREX_SPACEDIM][AMREX_SPACEDIM];
                    for (int r = 0; r < AMREX_SPACEDIM; ++r)
                    {
                        for (int c = 0; c < AMREX_SPACEDIM; ++c)
                        {
                            F[r][c] = p.rdata(realData::deformation_gradient +
                                              r * AMREX_SPACEDIM + c);
                        }
                    }

                    // Compute determinant (1D, 2D, or 3D)
                    amrex::Real detF = 0.0;
#if (AMREX_SPACEDIM == 1)
                    detF = F[0][0];
#elif (AMREX_SPACEDIM == 2)
                    detF = F[0][0] * F[1][1] - F[0][1] * F[1][0];
#else
                    detF = F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1]) -
                           F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0]) +
                           F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);
#endif

                    p.rdata(realData::jacobian) = detF;
                    p.rdata(realData::volume) =
                        p.rdata(realData::vol_init) * detF;
                    p.rdata(realData::density) =
                        p.rdata(realData::mass) / p.rdata(realData::volume);
                }
            });
    }
}


void MPMParticleContainer::moveParticles(
    const amrex::Real &dt,
    int bclo[AMREX_SPACEDIM],
    int bchi[AMREX_SPACEDIM],
    int lsetbc,
    amrex::Real wall_mu_lo[AMREX_SPACEDIM],
    amrex::Real wall_mu_hi[AMREX_SPACEDIM],
    amrex::Real wall_vel_lo[AMREX_SPACEDIM * AMREX_SPACEDIM],
    amrex::Real wall_vel_hi[AMREX_SPACEDIM * AMREX_SPACEDIM],
    amrex::Real lset_wall_mu)
{
    BL_PROFILE("MPMParticleContainer::moveParticles");

    const int lev = 0;
    const auto plo = Geom(lev).ProbLoArray();
    const auto phi = Geom(lev).ProbHiArray();
    const auto dx = Geom(lev).CellSizeArray();
    auto &plev = GetParticles(lev);

    bool using_levsets = mpm_ebtools::using_levelset_geometry;
    int lsref = mpm_ebtools::ls_refinement;

    GpuArray<int, AMREX_SPACEDIM> bc_lo_arr, bc_hi_arr;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        bc_lo_arr[d] = bclo[d];
        bc_hi_arr[d] = bchi[d];
    }

    GpuArray<amrex::Real, AMREX_SPACEDIM * AMREX_SPACEDIM> wall_vel_lo_arr,
        wall_vel_hi_arr;
    for (int d = 0; d < AMREX_SPACEDIM * AMREX_SPACEDIM; ++d)
    {
        wall_vel_lo_arr[d] = wall_vel_lo[d];
        wall_vel_hi_arr[d] = wall_vel_hi[d];
    }

    GpuArray<amrex::Real, AMREX_SPACEDIM> wall_mu_lo_arr, wall_mu_hi_arr;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        wall_mu_lo_arr[d] = wall_mu_lo[d];
        wall_mu_hi_arr[d] = wall_mu_hi[d];
    }

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        auto &ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto &aos = ptile.GetArrayOfStructs();
        const size_t np = aos.numParticles();
        ParticleType *pstruct = aos().dataPtr();

        amrex::Array4<amrex::Real> lsetarr;
        if (using_levsets)
        {
            lsetarr = mpm_ebtools::lsphi->array(mfi);
        }

        amrex::ParallelFor(
            np,
            [=] AMREX_GPU_DEVICE(int i) noexcept
            {
                ParticleType &p = pstruct[i];

                // Update positions for all dimensions
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    p.pos(d) += p.rdata(realData::xvel_prime + d) * dt;
                }

                // Build relvel arrays
                amrex::Real relvel_in[AMREX_SPACEDIM];
                amrex::Real relvel_out[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    relvel_in[d] = p.rdata(realData::xvel + d);
                    relvel_out[d] = p.rdata(realData::xvel + d);
                }

                // Levelset BC
                if (using_levsets && p.idata(intData::phase) == 0)
                {
                    amrex::Real xp[AMREX_SPACEDIM];
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        xp[d] = p.pos(d);

                    amrex::Real dist =
                        get_levelset_value(lsetarr, plo, dx, xp, lsref);
                    if (dist < TINYVAL)
                    {
                        amrex::Real normaldir[AMREX_SPACEDIM];
                        get_levelset_grad(lsetarr, plo, dx, xp, lsref,
                                          normaldir);

                        // normalize
                        amrex::Real gradmag = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            gradmag += normaldir[d] * normaldir[d];
                        gradmag = std::sqrt(gradmag);
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            normaldir[d] /= (gradmag + TINYVAL);

                        int modify_pos =
                            applybc(relvel_in, relvel_out, lset_wall_mu,
                                    normaldir, lsetbc);
                        if (modify_pos)
                        {
                            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            {
                                p.pos(d) +=
                                    2.0 * amrex::Math::abs(dist) * normaldir[d];
                            }
                        }
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            p.rdata(realData::xvel + d) = relvel_out[d];
                        }
                    }
                }

                // Domain boundary BCs
                amrex::Real wallvel[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    wallvel[d] = 0.0;

                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    if (p.pos(dir) < plo[dir])
                    {
                        // subtract wall velocity
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            wallvel[d] =
                                wall_vel_lo_arr[dir * AMREX_SPACEDIM + d];
                            relvel_in[d] -= wallvel[d];
                        }
                        amrex::Real normaldir[AMREX_SPACEDIM] = {0};
                        normaldir[dir] = 1.0;
                        int modify_pos =
                            applybc(relvel_in, relvel_out, wall_mu_lo_arr[dir],
                                    normaldir, bc_lo_arr[dir]);
                        if (modify_pos)
                        {
                            p.pos(dir) = 2.0 * plo[dir] - p.pos(dir);
                        }
                    }
                    else if (p.pos(dir) > phi[dir])
                    {
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            wallvel[d] =
                                wall_vel_hi_arr[dir * AMREX_SPACEDIM + d];
                            relvel_in[d] -= wallvel[d];
                        }
                        amrex::Real normaldir[AMREX_SPACEDIM] = {0};
                        normaldir[dir] = -1.0;
                        int modify_pos =
                            applybc(relvel_in, relvel_out, wall_mu_hi_arr[dir],
                                    normaldir, bc_hi_arr[dir]);
                        if (modify_pos)
                        {
                            p.pos(dir) = 2.0 * phi[dir] - p.pos(dir);
                        }
                    }
                }

                // Update velocities with BC corrections
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    p.rdata(realData::xvel + d) = relvel_out[d] + wallvel[d];
                }
            });
    }
}


amrex::Real MPMParticleContainer::GetPosSpring()
{
    //const int lev = 0;
    /*const Geometry &geom = Geom(lev);
    const auto plo = Geom(lev).ProbLoArray();
    const auto phi = Geom(lev).ProbHiArray();
    const auto dx = Geom(lev).CellSizeArray();
    auto &plev = GetParticles(lev);*/
    amrex::Real ymin = 0.0;

    using PType = typename MPMParticleContainer::SuperParticleType;
    ymin = amrex::ReduceMax(*this,
                            [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                            {
                                Real yscale;
                                yscale = p.pos(YDIR);
                                return (yscale);
                            });
    return (ymin);
}

amrex::Real MPMParticleContainer::GetPosPiston()
{
    //const int lev = 0;
    /*const Geometry &geom = Geom(lev);
    const auto plo = Geom(lev).ProbLoArray();
    const auto phi = Geom(lev).ProbHiArray();
    const auto dx = Geom(lev).CellSizeArray();
    auto &plev = GetParticles(lev);*/
    amrex::Real ymin = std::numeric_limits<amrex::Real>::max();

    using PType = typename MPMParticleContainer::SuperParticleType;
    ymin = amrex::ReduceMin(*this,
                            [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                            {
                                Real yscale;
                                if (p.idata(intData::phase) == 1 and
                                    p.idata(intData::rigid_body_id) == 0)
                                {
                                    yscale = p.pos(YDIR);
                                }
                                else
                                {
                                    yscale =
                                        std::numeric_limits<amrex::Real>::max();
                                }
                                return (yscale);
                            });
    return (ymin);
}

void MPMParticleContainer::UpdateRigidParticleVelocities(
    int rigid_body_id, Array<amrex::Real, AMREX_SPACEDIM> velocity)
{
    BL_PROFILE("MPMParticleContainer::GetVelPiston");

    const int lev = 0;
    /*const Geometry &geom = Geom(lev);
    const auto plo = Geom(lev).ProbLoArray();
    const auto phi = Geom(lev).ProbHiArray();
    const auto dx = Geom(lev).CellSizeArray();*/
    auto &plev = GetParticles(lev);

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        int gid = mfi.index();
        int tid = mfi.LocalTileIndex();
        auto index = std::make_pair(gid, tid);

        auto &ptile = plev[index];
        auto &aos = ptile.GetArrayOfStructs();
        const size_t np = aos.numParticles();
        ParticleType *pstruct = aos().dataPtr();

        // now we move the particles
        amrex::ParallelFor(np,
                           [=] AMREX_GPU_DEVICE(int i) noexcept
                           {
                               ParticleType &p = pstruct[i];
                               if (p.idata(intData::phase) == 1 and
                                   p.idata(intData::rigid_body_id) ==
                                       rigid_body_id)
                               {
                                   p.rdata(realData::xvel_prime) = velocity[0];
                                   p.rdata(realData::yvel_prime) = velocity[1];
                                   p.rdata(realData::zvel_prime) = velocity[2];
                               }
                           });
    }
}
