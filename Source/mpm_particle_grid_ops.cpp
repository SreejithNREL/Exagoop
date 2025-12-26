// clang-format off
#include "mpm_specs.H"
#include <mpm_particle_container.H>
#include <interpolants.H>
// clang-format on

int MPMParticleContainer::checkifrigidnodespresent()
{
    int rigidnodespresent = 0;

    using PType = typename MPMParticleContainer::SuperParticleType;
    rigidnodespresent = static_cast<int>(
        amrex::ReduceMax(*this,
                         [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                         {
                             if (p.idata(intData::phase) == 1)
                             {
                                 int rigidnodespresenttmp = 1;
                                 return (rigidnodespresenttmp);
                             }
                             else
                             {
                                 int rigidnodespresenttmp = 0;
                                 return (rigidnodespresenttmp);
                             }
                         }));

#ifdef BL_USE_MPI
    ParallelDescriptor::ReduceIntMax(rigidnodespresent);
#endif
    return (rigidnodespresent);
}

void MPMParticleContainer::Calculate_Total_Number_of_rigid_particles(
    int body_id, int &total_num)
{
    total_num = 0;

    using PType = typename MPMParticleContainer::SuperParticleType;
    total_num = static_cast<int>(
        amrex::ReduceSum(*this,
                         [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                         {
                             if (p.idata(intData::phase) == 1 and
                                 p.idata(intData::rigid_body_id) == body_id)
                             {
                                 return (1);
                             }
                             else
                             {
                                 return (0);
                             }
                         }));

#ifdef BL_USE_MPI
    ParallelDescriptor::ReduceIntSum(total_num);
#endif
}

void MPMParticleContainer::Calculate_Total_Number_of_MaterialParticles(
    int &total_num)
{
    total_num = 0;

    using PType = typename MPMParticleContainer::SuperParticleType;
    total_num = static_cast<int>(
        amrex::ReduceSum(*this,
                         [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                         {
                             if (p.idata(intData::phase) == 0)
                             {
                                 return (1);
                             }
                             else
                             {
                                 return (0);
                             }
                         }));

#ifdef BL_USE_MPI
    ParallelDescriptor::ReduceIntSum(total_num);
#endif
}

void MPMParticleContainer::Calculate_Total_Mass_RigidParticles(int body_id,
                                                               Real &total_mass)
{
    total_mass = 0.0;

    using PType = typename MPMParticleContainer::SuperParticleType;
    total_mass =
        amrex::ReduceSum(*this,
                         [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                         {
                             if (p.idata(intData::phase) == 1 and
                                 p.idata(intData::rigid_body_id) == body_id)
                             {
                                 return (p.rdata(realData::mass));
                             }
                             else
                             {
                                 return (0.0);
                             }
                         });

#ifdef BL_USE_MPI
    ParallelDescriptor::ReduceRealSum(total_mass);
#endif
}

void MPMParticleContainer::Calculate_Total_Mass_MaterialPoints(Real &total_mass)
{

    total_mass = 0.0;

    using PType = typename MPMParticleContainer::SuperParticleType;
    total_mass =
        amrex::ReduceSum(*this,
                         [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                         {
                             if (p.idata(intData::phase) == 0)
                             {
                                 return (p.rdata(realData::mass));
                             }
                             else
                             {
                                 return (0.0);
                             }
                         });

#ifdef BL_USE_MPI
    ParallelDescriptor::ReduceRealSum(total_mass);
#endif
}

void MPMParticleContainer::Calculate_Total_Vol_MaterialPoints(Real &total_vol)
{

    total_vol = 0.0;

    using PType = typename MPMParticleContainer::SuperParticleType;
    total_vol =
        amrex::ReduceSum(*this,
                         [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                         {
                             if (p.idata(intData::phase) == 0)
                             {
                                 return (p.rdata(realData::volume));
                             }
                             else
                             {
                                 return (0.0);
                             }
                         });

#ifdef BL_USE_MPI
    ParallelDescriptor::ReduceRealSum(total_vol);
#endif
}

amrex::Real
MPMParticleContainer::Calculate_Total_Vol_RigidParticles(int body_id)
{

    amrex::Real total_vol = 0.0;

    using PType = typename MPMParticleContainer::SuperParticleType;
    total_vol =
        amrex::ReduceSum(*this,
                         [=] AMREX_GPU_HOST_DEVICE(const PType &p) -> Real
                         {
                             if (p.idata(intData::phase) == 1 and
                                 p.idata(intData::rigid_body_id) == body_id)
                             {
                                 return (p.rdata(realData::volume));
                             }
                             else
                             {
                                 return (0.0);
                             }
                         });

#ifdef BL_USE_MPI
    ParallelDescriptor::ReduceRealSum(total_vol);
#endif
    return (total_vol);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE std::pair<int, int>
compute_bounds(int ivd, int lod, int hid, int scheme, bool is_periodic)
{
    int bmin = 0, bmax = 0;

    if (scheme == 1)
    {
        // Linear: 2-point support
        bmin = 0;
        bmax = 2;
    }
    else if (scheme == 3 or scheme == 2)
    {
        if (is_periodic)
        {
            bmin = -1;
            bmax = 3; // symmetric
        }
        else
        {
            if (ivd == lod)
            {
                bmin = 0;
                bmax = 3; // one-sided at lo
            }
            else if (ivd == hid)
            {
                bmin = -1;
                bmax = bmin + 3; // one-sided at hi
            }
            else
            {
                bmin = -1;
                bmax = 3; // interior symmetric
            }
        }
    }
    else
    {
        amrex::Abort("Unsupported interpolation scheme");
    }

    return {bmin, bmax};
}

void MPMParticleContainer::deposit_onto_grid_momentum(
    MultiFab &nodaldata,
    amrex::Array<Real, AMREX_SPACEDIM> gravity,
    int external_loads_present,
    amrex::Array<Real, AMREX_SPACEDIM> force_slab_lo,
    amrex::Array<Real, AMREX_SPACEDIM> force_slab_hi,
    amrex::Array<Real, AMREX_SPACEDIM> extforce,
    int update_mass,
    int update_vel,
    int update_forces,
    amrex::Real mass_tolerance,
    amrex::GpuArray<int, AMREX_SPACEDIM> order_scheme_directional,
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic)
{
    // if(testing==1) amrex::Print()<<"\n Entered deposit_onto_grid_momentum";

    const int lev = 0;
    const Geometry &geom = Geom(lev);
    auto &plev = GetParticles(lev);
    const auto dxi = geom.InvCellSizeArray();
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto domain = geom.Domain();

    const int *loarr = domain.loVect();
    const int *hiarr = domain.hiVect();

    Real grav[] = {AMREX_D_DECL(gravity[XDIR], gravity[YDIR], gravity[ZDIR])};
    Real extpforce[] = {
        AMREX_D_DECL(extforce[XDIR], extforce[YDIR], extforce[ZDIR])};

    int lo[] = {AMREX_D_DECL(loarr[0], loarr[1], loarr[2])};
    int hi[] = {AMREX_D_DECL(hiarr[0], hiarr[1], hiarr[2])};

    int extloads = external_loads_present;

    // Zero out nodal data
    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {

        const Box &nodalbox = mfi.validbox();
        auto nodal_data_arr = nodaldata.array(mfi);
        amrex::ParallelFor(
            nodalbox,
            [=] AMREX_GPU_DEVICE(AMREX_D_DECL(int i, int j, int k)) noexcept
            {
                IntVect nodeindex(AMREX_D_DECL(i, j, k));
                if (update_mass)
                {
                    nodal_data_arr(nodeindex, MASS_INDEX) = 0.0;
                }
                if (update_vel)
                {

                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        nodal_data_arr(nodeindex, VELX_INDEX + d) = 0.0;
                    }
                }
                if (update_forces)
                {
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        nodal_data_arr(nodeindex, FRCX_INDEX + d) = 0.0;
                    }
                }
                if (update_forces == 2)
                {
                    nodal_data_arr(nodeindex, STRESS_INDEX) = 0.0;
                }
            });
    }

    // Deposit particle contributions
    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {

        Box nodalbox = convert(mfi.tilebox(), {AMREX_D_DECL(1, 1, 1)});
        auto &ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto &aos = ptile.GetArrayOfStructs();
        int nt = aos.numRealParticles() + aos.numNeighborParticles();
        auto nodal_data_arr = nodaldata.array(mfi);
        ParticleType *pstruct = aos().dataPtr();

        amrex::ParallelFor(
            nt,
            [=] AMREX_GPU_DEVICE(int ip) noexcept
            {
                ParticleType &p = pstruct[ip];
                if (p.idata(intData::phase) != 0)
                    return;

                amrex::Real xp[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    xp[d] = p.pos(d);

                auto iv = getParticleCell(p, plo, dxi, domain);

                // Compute stencil extents per dimension

                int min_idx[AMREX_SPACEDIM], max_idx[AMREX_SPACEDIM];

                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    auto bounds = compute_bounds(iv[d], lo[d], hi[d],
                                                 order_scheme_directional[d],
                                                 periodic[d]);
                    min_idx[d] = bounds.first;
                    max_idx[d] = bounds.second;
                }

                // Nested loops over stencil (specialized per dimension)
                amrex::Real basisvalue = 0.0;
                amrex::Real basisval_grad[AMREX_SPACEDIM] = {
                    AMREX_D_DECL(0.0, 0.0, 0.0)};
                IntVect ivlocal = {AMREX_D_DECL(0, 0, 0)};

#if (AMREX_SPACEDIM == 3)
                for (int n = min_idx[2]; n < max_idx[2]; n++)
                {
#endif
#if (AMREX_SPACEDIM >= 2)
                    for (int m = min_idx[1]; m < max_idx[1]; m++)
                    {
#endif
                        for (int l = min_idx[0]; l < max_idx[0]; l++)
                        {
                            ivlocal = {
                                AMREX_D_DECL(iv[0] + l, iv[1] + m, iv[2] + n)};
                            IntVect stencil(AMREX_D_DECL(l, m, n));
                            if (nodalbox.contains(ivlocal))
                            {

                                // amrex::Print()<<"\n Inisde P2G "<<xp[0]<<"
                                // "<<xp[1]<<" "<<iv[0]<<" "<<iv[1]<<"
                                // "<<stencil[0]<<" "<<stencil[1];
                                basisvalue = basisval(stencil, iv, xp, plo, dx,
                                                      order_scheme_directional,
                                                      periodic, lo, hi);

                                if (update_mass)
                                {
                                    amrex::Real mass_contrib =
                                        p.rdata(realData::mass) * basisvalue;
                                    amrex::Gpu::Atomic::AddNoRet(
                                        &nodal_data_arr(ivlocal, MASS_INDEX),
                                        mass_contrib);
                                }
                                if (update_vel)
                                {

                                    amrex::Real p_contrib[AMREX_SPACEDIM] = {
                                        AMREX_D_DECL(
                                            p.rdata(realData::mass) *
                                                p.rdata(realData::xvel) *
                                                basisvalue,
                                            p.rdata(realData::mass) *
                                                p.rdata(realData::yvel) *
                                                basisvalue,
                                            p.rdata(realData::mass) *
                                                p.rdata(realData::zvel) *
                                                basisvalue)};

                                    for (int dim = 0; dim < AMREX_SPACEDIM;
                                         dim++)
                                    {
                                        amrex::Gpu::Atomic::AddNoRet(
                                            &nodal_data_arr(ivlocal,
                                                            VELX_INDEX + dim),
                                            p_contrib[dim]);
                                    }
                                }

                                if (update_forces)
                                {
                                    for (int dim = 0; dim < AMREX_SPACEDIM;
                                         dim++)
                                    {
                                        basisval_grad[dim] = basisvalder(
                                            dim, stencil, iv, xp, plo, dx,
                                            order_scheme_directional, periodic,
                                            lo, hi);
                                    }

                                    amrex::Real stress_tens[AMREX_SPACEDIM *
                                                            AMREX_SPACEDIM];
                                    get_tensor(p, realData::stress,
                                               stress_tens);

                                    amrex::Real bforce_contrib[AMREX_SPACEDIM] =
                                        {AMREX_D_DECL(
                                            p.rdata(realData::mass) *
                                                grav[XDIR] * basisvalue,
                                            p.rdata(realData::mass) *
                                                grav[YDIR] * basisvalue,
                                            p.rdata(realData::mass) *
                                                grav[ZDIR] * basisvalue)};

                                    if (extloads &&
                                        xp[XDIR] > force_slab_lo[XDIR] &&
                                        xp[XDIR] < force_slab_hi[XDIR]
#if (AMREX_SPACEDIM >= 2)
                                        && xp[YDIR] > force_slab_lo[YDIR] &&
                                        xp[YDIR] < force_slab_hi[YDIR]
#endif
#if (AMREX_SPACEDIM == 3)
                                        && xp[ZDIR] > force_slab_lo[ZDIR] &&
                                        xp[ZDIR] < force_slab_hi[ZDIR]
#endif
                                    )
                                    {
                                        for (int dim = 0; dim < AMREX_SPACEDIM;
                                             dim++)
                                        {
                                            bforce_contrib[dim] +=
                                                extpforce[dim] * basisvalue;
                                        }
                                    }

                                    amrex::Real tensvect[AMREX_SPACEDIM];
                                    tensor_vector_pdt(stress_tens,
                                                      basisval_grad, tensvect);

                                    amrex::Real
                                        intforce_contrib[AMREX_SPACEDIM] = {
                                            AMREX_D_DECL(
                                                -p.rdata(realData::volume) *
                                                    tensvect[XDIR],
                                                -p.rdata(realData::volume) *
                                                    tensvect[YDIR],
                                                -p.rdata(realData::volume) *
                                                    tensvect[ZDIR])};

                                    for (int dim = 0; dim < AMREX_SPACEDIM;
                                         dim++)
                                    {
                                        IntVect nodeindex(AMREX_D_DECL(
                                            iv[0] + l, iv[1] + m, iv[2] + n));
                                        amrex::Gpu::Atomic::AddNoRet(
                                            &nodal_data_arr(nodeindex,
                                                            FRCX_INDEX + dim),
                                            bforce_contrib[dim] +
                                                intforce_contrib[dim]);
                                    }
                                }
                            }
                        }
#if (AMREX_SPACEDIM >= 2)
                    }
#endif
#if (AMREX_SPACEDIM == 3)
                }
#endif
            });
    }

    // if(testing==1)	amrex::Print()<<"\n B4 Normalizing nodal data at node ";
    // Normalize velocities and stresses
    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        Box nodalbox = convert(mfi.tilebox(), {AMREX_D_DECL(1, 1, 1)});
        auto nodal_data_arr = nodaldata.array(mfi);
        amrex::ParallelFor(
            nodalbox,
            [=] AMREX_GPU_DEVICE(AMREX_D_DECL(int i, int j, int k)) noexcept
            {
                IntVect nodeindex(AMREX_D_DECL(i, j, k));
                // if(testing==1)	amrex::Print()<<"\n Normalizing nodal
                // data at node "<<i<<" "<<j<<" "<<nodal_data_arr(nodeindex,
                // MASS_INDEX);
                if (update_vel && nodal_data_arr(nodeindex, MASS_INDEX) > 0.0)
                {
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        if (nodal_data_arr(nodeindex, MASS_INDEX) >=
                            mass_tolerance)
                        {
                            nodal_data_arr(nodeindex, VELX_INDEX + d) /=
                                nodal_data_arr(nodeindex, MASS_INDEX);
                        }

                        else
                            nodal_data_arr(nodeindex, VELX_INDEX + d) = 0.0;
                    }
                }
                if (update_forces == 2 &&
                    nodal_data_arr(nodeindex, MASS_INDEX) > 0.0)
                {
                    if (nodal_data_arr(nodeindex, MASS_INDEX) >= mass_tolerance)
                        nodal_data_arr(nodeindex, STRESS_INDEX) /=
                            nodal_data_arr(nodeindex, MASS_INDEX);
                    else
                        nodal_data_arr(nodeindex, STRESS_INDEX) = 0.0;
                }
            });
    }
}

#if USE_TEMP
void MPMParticleContainer::deposit_onto_grid_temperature(
    MultiFab &nodaldata,
    bool resetnodaldata_to_zero,
    bool update_mass_temp,
    bool update_source,
    amrex::Real mass_tolerance,
    GpuArray<int, AMREX_SPACEDIM> order_scheme_directional,
    GpuArray<int, AMREX_SPACEDIM> periodic)
{
    // if(testing==1) amrex::Print()<<"\n Entered deposit_onto_grid_momentum";

    const int lev = 0;
    const Geometry &geom = Geom(lev);
    auto &plev = GetParticles(lev);
    const auto dxi = geom.InvCellSizeArray();
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto domain = geom.Domain();

    const int *loarr = domain.loVect();
    const int *hiarr = domain.hiVect();

    int lo[] = {AMREX_D_DECL(loarr[0], loarr[1], loarr[2])};
    int hi[] = {AMREX_D_DECL(hiarr[0], hiarr[1], hiarr[2])};

    // Zero out nodal data
    if (resetnodaldata_to_zero)
    {
        for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
        {

            const Box &nodalbox = mfi.validbox();
            auto nodal_data_arr = nodaldata.array(mfi);
            amrex::ParallelFor(
                nodalbox,
                [=] AMREX_GPU_DEVICE(AMREX_D_DECL(int i, int j, int k)) noexcept
                {
                    IntVect nodeindex(AMREX_D_DECL(i, j, k));
                    nodal_data_arr(nodeindex, MASS_SPHEAT) = shunya;
                    nodal_data_arr(nodeindex, MASS_SPHEAT_TEMP) = shunya;
                    // nodal_data_arr(nodeindex, TEMPERATURE) = shunya;
                    nodal_data_arr(nodeindex, SOURCE_TEMP_INDEX) = shunya;
                });
        }
    }

    // Deposit particle contributions
    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {

        Box nodalbox = convert(mfi.tilebox(), {AMREX_D_DECL(1, 1, 1)});
        auto &ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto &aos = ptile.GetArrayOfStructs();
        int nt = aos.numRealParticles() + aos.numNeighborParticles();
        auto nodal_data_arr = nodaldata.array(mfi);
        ParticleType *pstruct = aos().dataPtr();

        amrex::ParallelFor(
            nt,
            [=] AMREX_GPU_DEVICE(int ip) noexcept
            {
                ParticleType &p = pstruct[ip];
                if (p.idata(intData::phase) != 0)
                    return;

                amrex::Real xp[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    xp[d] = p.pos(d);

                auto iv = getParticleCell(p, plo, dxi, domain);

                // Compute stencil extents per dimension

                int min_idx[AMREX_SPACEDIM], max_idx[AMREX_SPACEDIM];

                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    auto bounds = compute_bounds(iv[d], lo[d], hi[d],
                                                 order_scheme_directional[d],
                                                 periodic[d]);
                    min_idx[d] = bounds.first;
                    max_idx[d] = bounds.second;
                }

                // Nested loops over stencil (specialized per dimension)
                amrex::Real basisvalue = 0.0;
                amrex::Real basisval_grad[AMREX_SPACEDIM] = {
                    AMREX_D_DECL(0.0, 0.0, 0.0)};
                IntVect ivlocal = {AMREX_D_DECL(0, 0, 0)};

#if (AMREX_SPACEDIM == 3)
                for (int n = min_idx[2]; n < max_idx[2]; n++)
                {
#endif
#if (AMREX_SPACEDIM >= 2)
                    for (int m = min_idx[1]; m < max_idx[1]; m++)
                    {
#endif
                        for (int l = min_idx[0]; l < max_idx[0]; l++)
                        {
                            ivlocal = {
                                AMREX_D_DECL(iv[0] + l, iv[1] + m, iv[2] + n)};
                            IntVect stencil(AMREX_D_DECL(l, m, n));
                            if (nodalbox.contains(ivlocal))
                            {
                                basisvalue = basisval(stencil, iv, xp, plo, dx,
                                                      order_scheme_directional,
                                                      periodic, lo, hi);

                                if (update_mass_temp)
                                {
                                    amrex::Real mass_spheat_contrib =
                                        p.rdata(realData::mass) *
                                        p.rdata(realData::specific_heat) *
                                        basisvalue;
                                    amrex::Real mass_spheat_temp_contrib =
                                        p.rdata(realData::mass) *
                                        p.rdata(realData::specific_heat) *
                                        p.rdata(realData::temperature) *
                                        basisvalue;
                                    amrex::Gpu::Atomic::AddNoRet(
                                        &nodal_data_arr(ivlocal, MASS_SPHEAT),
                                        mass_spheat_contrib);
                                    amrex::Gpu::Atomic::AddNoRet(
                                        &nodal_data_arr(ivlocal,
                                                        MASS_SPHEAT_TEMP),
                                        mass_spheat_temp_contrib);
                                }

                                if (update_source)
                                {
                                    for (int dim = 0; dim < AMREX_SPACEDIM;
                                         dim++)
                                    {
                                        basisval_grad[dim] = basisvalder(
                                            dim, stencil, iv, xp, plo, dx,
                                            order_scheme_directional, periodic,
                                            lo, hi);
                                    }

                                    amrex::Real heat_flux_vect[AMREX_SPACEDIM];
                                    amrex::Real net_heatflux = 0.0;
                                    amrex::Real int_source = 0.0;
                                    amrex::Real ext_source = 0.0;
                                    get_vector(p, realData::heat_flux,
                                               heat_flux_vect);

                                    for (int dim = 0; dim < AMREX_SPACEDIM;
                                         dim++)
                                    {
                                        net_heatflux += heat_flux_vect[dim] *
                                                        basisval_grad[dim];
                                    }

                                    int_source = net_heatflux *
                                                 p.rdata(realData::volume);
                                    ext_source =
                                        p.rdata(realData::volume) *
                                        p.rdata(realData::heat_source) *
                                        basisvalue;

                                    IntVect nodeindex(AMREX_D_DECL(
                                        iv[0] + l, iv[1] + m, iv[2] + n));
                                    amrex::Gpu::Atomic::AddNoRet(
                                        &nodal_data_arr(nodeindex,
                                                        SOURCE_TEMP_INDEX),
                                        int_source + ext_source);
                                }
                            }
                        }
#if (AMREX_SPACEDIM >= 2)
                    }
#endif
#if (AMREX_SPACEDIM == 3)
                }
#endif
            });
    }

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        Box nodalbox = convert(mfi.tilebox(), {AMREX_D_DECL(1, 1, 1)});
        auto nodal_data_arr = nodaldata.array(mfi);
        amrex::ParallelFor(
            nodalbox,
            [=] AMREX_GPU_DEVICE(AMREX_D_DECL(int i, int j, int k)) noexcept
            {
                IntVect nodeindex(AMREX_D_DECL(i, j, k));
                if (update_mass_temp &&
                    nodal_data_arr(nodeindex, MASS_SPHEAT) > 0.0)
                {

                    if (nodal_data_arr(nodeindex, MASS_SPHEAT) >=
                        mass_tolerance)
                    {
                        nodal_data_arr(nodeindex, TEMPERATURE) =
                            nodal_data_arr(nodeindex, MASS_SPHEAT_TEMP) /
                            nodal_data_arr(nodeindex, MASS_SPHEAT);
                    }

                    else
                    {
                        nodal_data_arr(nodeindex, TEMPERATURE) = 0.0;
                    }
                }
            });
    }
}

#endif

void MPMParticleContainer::deposit_onto_grid_rigidnodesonly(
    MultiFab & /*nodaldata*/,
    Array<Real, AMREX_SPACEDIM> /*gravity*/,
    int /*external_loads_present*/,
    Array<Real, AMREX_SPACEDIM> /*force_slab_lo*/,
    Array<Real, AMREX_SPACEDIM> /*force_slab_hi*/,
    Array<Real, AMREX_SPACEDIM> /*extforce*/,
    int /*update_massvel*/,
    int /*update_forces*/,
    amrex::Real /*mass_tolerance*/,
    GpuArray<int, AMREX_SPACEDIM> /*order_scheme_directional*/,
    GpuArray<int, AMREX_SPACEDIM> /*periodic*/)
{ /*
     const int lev = 0;
     const Geometry &geom = Geom(lev);
     auto &plev = GetParticles(lev);
     const auto dxi = geom.InvCellSizeArray();
     const auto dx = geom.CellSizeArray();
     const auto plo = geom.ProbLoArray();
     const auto domain = geom.Domain();

     const int *loarr = domain.loVect();
     const int *hiarr = domain.hiVect();

     int lo[] = {loarr[0], loarr[1], loarr[2]};
     int hi[] = {hiarr[0], hiarr[1], hiarr[2]};

     for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
     {
         // already nodal as mfi is from nodaldata
         const Box &nodalbox = mfi.validbox();

         Array4<Real> nodal_data_arr = nodaldata.array(mfi);

         amrex::ParallelFor(nodalbox,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                            { nodal_data_arr(i, j, k, RIGID_BODY_ID) = -1; });
     }

     for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
     {
         const amrex::Box &box = mfi.tilebox();
         Box nodalbox = convert(box, {AMREX_D_DECL(1,1,1)});

         int gid = mfi.index();
         int tid = mfi.LocalTileIndex();
         auto index = std::make_pair(gid, tid);

         auto &ptile = plev[index];
         auto &aos = ptile.GetArrayOfStructs();
         int np = aos.numRealParticles();
         int ng = aos.numNeighborParticles();
         int nt = np + ng;

         Array4<Real> nodal_data_arr = nodaldata.array(mfi);

         ParticleType *pstruct = aos().dataPtr();

         amrex::ParallelFor(
             nt,
             [=] AMREX_GPU_DEVICE(int i) noexcept
             {
                 int lmin, lmax, nmin, nmax, mmin, mmax;

                 ParticleType &p = pstruct[i];

                 if (p.idata(intData::phase) ==
                     1) // Compute only for rigid particles with phase=1
                 {
                     amrex::Real xp[AMREX_SPACEDIM];

                     xp[XDIR] = p.pos(XDIR);
                     xp[YDIR] = p.pos(YDIR);
                     xp[ZDIR] = p.pos(ZDIR);

                     auto iv = getParticleCell(p, plo, dxi, domain);

                     lmin = (order_scheme_directional[0] == 1)
                                ? 0
                                : ((order_scheme_directional[0] == 3)
                                       ? (iv[XDIR] == lo[XDIR])
                                             ? 0
                                             : ((iv[XDIR] == hi[XDIR]) ? -1 :
     -1) : -1000); lmax = (order_scheme_directional[0] == 1) ? 2 :
     ((order_scheme_directional[0] == 3) ? (iv[XDIR] == lo[XDIR]) ? lmin + 3 :
     ((iv[XDIR] == hi[XDIR]) ? lmin + 3 : lmin + 4) : -1000);

                     mmin = (order_scheme_directional[1] == 1)
                                ? 0
                                : ((order_scheme_directional[1] == 3)
                                       ? (iv[YDIR] == lo[YDIR])
                                             ? 0
                                             : ((iv[YDIR] == hi[YDIR]) ? -1 :
     -1) : -1000); mmax = (order_scheme_directional[1] == 1) ? 2 :
     ((order_scheme_directional[1] == 3) ? (iv[YDIR] == lo[YDIR]) ? mmin + 3 :
     ((iv[YDIR] == hi[YDIR]) ? mmin + 3 : mmin + 4) : -1000);

                     nmin = (order_scheme_directional[2] == 1)
                                ? 0
                                : ((order_scheme_directional[2] == 3)
                                       ? (iv[ZDIR] == lo[ZDIR])
                                             ? 0
                                             : ((iv[ZDIR] == hi[ZDIR]) ? -1 :
     -1) : -1000); nmax = (order_scheme_directional[2] == 1) ? 2 :
     ((order_scheme_directional[2] == 3) ? (iv[ZDIR] == lo[ZDIR]) ? nmin + 3 :
     ((iv[ZDIR] == hi[ZDIR]) ? nmin + 3 : nmin + 4) : -1000);

                     if (lmin == -1000 or lmax == -1000 or mmin == -1000 or
                         mmax == -1000 or nmin == -1000 or nmax == -1000)
                     {
                         amrex::Abort("\nError. Something wrong with min/max "
                                      "index values in "
                                      "deposit onto grid");
                     }

                     for (int n = nmin; n < nmax; n++)
                     {
                         for (int m = mmin; m < mmax; m++)
                         {
                             for (int l = lmin; l < lmax; l++)
                             {
                                 IntVect ivlocal({AMREX_D_DECL(iv[XDIR] + l,
     iv[YDIR] + m, iv[ZDIR] + n)});

                                 if (nodalbox.contains(ivlocal))
                                 {

                                     amrex::Real basisvalue = basisval(
                                         l, m, n, iv[XDIR], iv[YDIR], iv[ZDIR],
                                         xp, plo, dx, order_scheme_directional,
                                         periodic, lo, hi);

                                     amrex::Real mass_contrib =
                                         p.rdata(realData::mass) * basisvalue;
                                     amrex::Real p_contrib[AMREX_SPACEDIM] = {
                                         p.rdata(realData::mass) *
                                             p.rdata(realData::xvel) *
                                             basisvalue,
                                         p.rdata(realData::mass) *
                                             p.rdata(realData::yvel) *
                                             basisvalue,
                                         p.rdata(realData::mass) *
                                             p.rdata(realData::zvel) *
                                             basisvalue};

                                     amrex::Gpu::Atomic::AddNoRet(
                                         &nodal_data_arr(ivlocal,
                                                         MASS_RIGID_INDEX),
                                         mass_contrib);
                                     nodal_data_arr(ivlocal, RIGID_BODY_ID) =
                                         p.idata(intData::rigid_body_id);

                                     for (int dim = 0; dim < AMREX_SPACEDIM;
                                          dim++)
                                     {
                                         amrex::Gpu::Atomic::AddNoRet(
                                             &nodal_data_arr(ivlocal,
                                                             VELX_RIGID_INDEX +
                                                                 dim),
                                             p_contrib[dim]);
                                     }
                                 }
                             }
                         }
                     }
                 }
             });
     }
     // nodaldata.FillBoundary(geom.periodicity());
     // nodaldata.SumBoundary(geom.periodicity());
     for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
     {
         const amrex::Box &box = mfi.tilebox();
         Box nodalbox = convert(box, {1, 1, 1});

         Array4<Real> nodal_data_arr = nodaldata.array(mfi);

         amrex::ParallelFor(
             nodalbox,
             [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
             {
                 if (update_massvel)
                 {
                     // amrex::Print()<<"\n Nodal mass values for i = "<<i<<" j
     =
                     // "<<j<<" k =
                     // "<<k<<" is "<<nodal_data_arr(i,j,k,MASS_INDEX);
                     if (nodal_data_arr(i, j, k, MASS_RIGID_INDEX) > 0.0)
                     {
                         for (int dim = 0; dim < AMREX_SPACEDIM; dim++)
                         {
                             if (nodal_data_arr(i, j, k, MASS_RIGID_INDEX) >=
                                 mass_tolerance)
                             {
                                 nodal_data_arr(i, j, k,
                                                VELX_RIGID_INDEX + dim) /=
                                     nodal_data_arr(i, j, k, MASS_RIGID_INDEX);
                             }
                             else
                             {
                                 nodal_data_arr(i, j, k,
                                                VELX_RIGID_INDEX + dim) = 0.0;
                             }
                         }
                     }
                 }
             });
     }*/
}

void MPMParticleContainer::interpolate_from_grid(
    MultiFab &nodaldata,
    int update_vel,
    int update_strainrate,
    GpuArray<int, AMREX_SPACEDIM> order_scheme_directional,
    GpuArray<int, AMREX_SPACEDIM> periodic,
    amrex::Real alpha_pic_flip,
    amrex::Real dt)
{
    const int lev = 0;
    const Geometry &geom = Geom(lev);
    auto &plev = GetParticles(lev);
    const auto dxi = geom.InvCellSizeArray();
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto domain = geom.Domain();

    const int *loarr = domain.loVect();
    const int *hiarr = domain.hiVect();

    int lo[AMREX_SPACEDIM];
    int hi[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        lo[d] = loarr[d];
        hi[d] = hiarr[d];
    }

    nodaldata.FillBoundary(geom.periodicity());

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        int gid = mfi.index();
        int tid = mfi.LocalTileIndex();
        auto index = std::make_pair(gid, tid);

        auto &ptile = plev[index];
        auto &aos = ptile.GetArrayOfStructs();
        const int np = aos.numRealParticles();

        Array4<Real> nodal_data_arr = nodaldata.array(mfi);
        ParticleType *pstruct = aos().dataPtr();

        amrex::ParallelFor(
            np,
            [=] AMREX_GPU_DEVICE(int i) noexcept
            {
                ParticleType &p = pstruct[i];
                if (p.idata(intData::phase) != 0)
                    return;

                amrex::Real xp[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    xp[d] = p.pos(d);
                }

                auto iv = getParticleCell(p, plo, dxi, domain);

                // Dimension‑aware min/max indices
                IntVect min_index = {AMREX_D_DECL(0, 0, 0)};
                IntVect max_index = {AMREX_D_DECL(0, 0, 0)};

                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    auto bounds = compute_bounds(iv[d], lo[d], hi[d],
                                                 order_scheme_directional[d],
                                                 periodic[d]);
                    min_index[d] = bounds.first;
                    max_index[d] = bounds.second;
                }

                if (update_vel)
                {
                    for (int dim = 0; dim < AMREX_SPACEDIM; dim++)
                    {

                        if (order_scheme_directional[dim] == 1)
                        {
                            p.rdata(realData::xvel_prime + dim) =
                                bilin_interp(xp, iv, plo, dx, nodal_data_arr,
                                             VELX_INDEX + dim);
                            p.rdata(realData::xvel + dim) =
                                alpha_pic_flip * p.rdata(realData::xvel + dim) +
                                alpha_pic_flip *
                                    bilin_interp(xp, iv, plo, dx,
                                                 nodal_data_arr,
                                                 DELTA_VELX_INDEX + dim) +
                                (1 - alpha_pic_flip) *
                                    p.rdata(realData::xvel_prime + dim);
                        }
                        else if (order_scheme_directional[dim] == 2)
                        {
                            p.rdata(realData::xvel_prime + dim) =
                                quadratic_interp(xp, iv, min_index, max_index,
                                                 plo, dx, nodal_data_arr,
                                                 VELX_INDEX + dim, lo, hi);
                            p.rdata(realData::xvel + dim) =
                                alpha_pic_flip * p.rdata(realData::xvel + dim) +
                                alpha_pic_flip *
                                    quadratic_interp(
                                        xp, iv, min_index, max_index, plo, dx,
                                        nodal_data_arr, DELTA_VELX_INDEX + dim,
                                        lo, hi) +
                                (1 - alpha_pic_flip) *
                                    p.rdata(realData::xvel_prime + dim);
                        }
                        else if (order_scheme_directional[dim] == 3)
                        {
                            p.rdata(realData::xvel_prime + dim) = cubic_interp(
                                xp, iv, min_index, max_index, plo, dx,
                                nodal_data_arr, VELX_INDEX + dim, lo, hi);
                            p.rdata(realData::xvel + dim) =
                                alpha_pic_flip * p.rdata(realData::xvel + dim) +
                                alpha_pic_flip *
                                    cubic_interp(xp, iv, min_index, max_index,
                                                 plo, dx, nodal_data_arr,
                                                 DELTA_VELX_INDEX + dim, lo,
                                                 hi) +
                                (1 - alpha_pic_flip) *
                                    p.rdata(realData::xvel_prime + dim);
                        }
                    }
                }

                // Strain rate update
                if (update_strainrate)
                {
                    amrex::Real gradvp[AMREX_SPACEDIM][AMREX_SPACEDIM] = {};
#if (AMREX_SPACEDIM == 3)
                    for (int n = min_index[2]; n < max_index[2]; ++n)
                    {
#endif
#if (AMREX_SPACEDIM != 1)
                        for (int m = min_index[1]; m < max_index[1]; ++m)
                        {
#endif
                            for (int l = min_index[0]; l < max_index[0]; ++l)
                            {
                                amrex::Real basisval_grad[AMREX_SPACEDIM];
                                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                {
                                    IntVect stencil(AMREX_D_DECL(l, m, n));
                                    IntVect celliv(
                                        AMREX_D_DECL(iv[0], iv[1], iv[2]));
                                    basisval_grad[d] = basisvalder(
                                        d, stencil, celliv, xp, plo, dx,
                                        order_scheme_directional, periodic, lo,
                                        hi);
                                }
                                for (int d1 = 0; d1 < AMREX_SPACEDIM; ++d1)
                                {
                                    for (int d2 = 0; d2 < AMREX_SPACEDIM; ++d2)
                                    {
                                        IntVect nodeindex(AMREX_D_DECL(
                                            iv[0] + l, iv[1] + m, iv[2] + n));
                                        gradvp[d1][d2] +=
                                            nodal_data_arr(nodeindex,
                                                           VELX_INDEX + d1) *
                                            basisval_grad[d2];
                                    }
                                }
                            }
#if (AMREX_SPACEDIM != 1)
                        }
#endif
#if (AMREX_SPACEDIM == 3)
                    }
#endif

                    // Calculate deformation gradient tensor. F_p^{t+dt} at time
                    // t+dt
                    get_deformation_gradient_tensor(
                        p, realData::deformation_gradient, gradvp, dt);

                    // Calculate strain rate tensor. D_p^{t+dt} at time t+dt
                    int ind = 0;
                    for (int d1 = 0; d1 < AMREX_SPACEDIM; ++d1)
                    {
                        for (int d2 = d1; d2 < AMREX_SPACEDIM; ++d2)
                        {
                            p.rdata(realData::strainrate + ind) =
                                0.5 * (gradvp[d1][d2] + gradvp[d2][d1]);
                            ind++;
                        }
                    }
                }
            });
    }
}

#if USE_TEMP
void MPMParticleContainer::interpolate_from_grid_temperature(
    MultiFab &nodaldata,
    bool update_temperature,
    bool update_heatflux,
    GpuArray<int, AMREX_SPACEDIM> order_scheme_directional,
    GpuArray<int, AMREX_SPACEDIM> periodic,
    amrex::Real alpha_pic_flip)
{
    const int lev = 0;
    const Geometry &geom = Geom(lev);
    auto &plev = GetParticles(lev);
    const auto dxi = geom.InvCellSizeArray();
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto domain = geom.Domain();

    const int *loarr = domain.loVect();
    const int *hiarr = domain.hiVect();

    int lo[AMREX_SPACEDIM];
    int hi[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        lo[d] = loarr[d];
        hi[d] = hiarr[d];
    }

    nodaldata.FillBoundary(geom.periodicity());

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        int gid = mfi.index();
        int tid = mfi.LocalTileIndex();
        auto index = std::make_pair(gid, tid);

        auto &ptile = plev[index];
        auto &aos = ptile.GetArrayOfStructs();
        const int np = aos.numRealParticles();

        Array4<Real> nodal_data_arr = nodaldata.array(mfi);
        ParticleType *pstruct = aos().dataPtr();

        amrex::ParallelFor(
            np,
            [=] AMREX_GPU_DEVICE(int i) noexcept
            {
                ParticleType &p = pstruct[i];
                if (p.idata(intData::phase) != 0)
                    return;

                amrex::Real xp[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    xp[d] = p.pos(d);
                }

                auto iv = getParticleCell(p, plo, dxi, domain);

                // Dimension‑aware min/max indices
                IntVect min_index = {AMREX_D_DECL(0, 0, 0)};
                IntVect max_index = {AMREX_D_DECL(0, 0, 0)};

                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    auto bounds = compute_bounds(iv[d], lo[d], hi[d],
                                                 order_scheme_directional[d],
                                                 periodic[d]);
                    min_index[d] = bounds.first;
                    max_index[d] = bounds.second;
                }

                if (update_temperature)
                {
                    if (order_scheme_directional[0] == 1)
                    {
                        p.rdata(realData::temperature) += bilin_interp(
                            xp, iv, plo, dx, nodal_data_arr, DELTA_TEMPERATURE);
                    }
                    else if (order_scheme_directional[0] == 2)
                    {
                        p.rdata(realData::temperature) += quadratic_interp(
                            xp, iv, min_index, max_index, plo, dx,
                            nodal_data_arr, DELTA_TEMPERATURE, lo, hi);
                    }
                    else if (order_scheme_directional[0] == 3)
                    {
                        p.rdata(realData::temperature) += cubic_interp(
                            xp, iv, min_index, max_index, plo, dx,
                            nodal_data_arr, DELTA_TEMPERATURE, lo, hi);
                    }
                }

                // Strain rate update
                if (update_heatflux)
                {
                    amrex::Real gradT[AMREX_SPACEDIM] = {};
#if (AMREX_SPACEDIM == 3)
                    for (int n = min_index[2]; n < max_index[2]; ++n)
                    {
#endif
#if (AMREX_SPACEDIM != 1)
                        for (int m = min_index[1]; m < max_index[1]; ++m)
                        {
#endif
                            for (int l = min_index[0]; l < max_index[0]; ++l)
                            {
                                amrex::Real basisval_grad[AMREX_SPACEDIM];
                                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                {
                                    IntVect stencil(AMREX_D_DECL(l, m, n));
                                    IntVect celliv(
                                        AMREX_D_DECL(iv[0], iv[1], iv[2]));
                                    basisval_grad[d] = basisvalder(
                                        d, stencil, celliv, xp, plo, dx,
                                        order_scheme_directional, periodic, lo,
                                        hi);
                                }
                                for (int d1 = 0; d1 < AMREX_SPACEDIM; ++d1)
                                {
                                    IntVect nodeindex(AMREX_D_DECL(
                                        iv[0] + l, iv[1] + m, iv[2] + n));
                                    gradT[d1] +=
                                        nodal_data_arr(nodeindex, TEMPERATURE) *
                                        basisval_grad[d1];
                                    // amrex::Print()<<"\n Temp = "<<d1<<"
                                    // "<<nodal_data_arr(nodeindex,TEMPERATURE)<<"
                                    // "<<basisval_grad[d1];
                                }
                            }
#if (AMREX_SPACEDIM != 1)
                        }
#endif
#if (AMREX_SPACEDIM == 3)
                    }
#endif
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        p.rdata(realData::heat_flux + d) =
                            -p.rdata(realData::thermal_conductivity) * gradT[d];
                    }
                }
            });
    }
}
#endif

void MPMParticleContainer::calculate_nodal_normal(
    MultiFab & /*nodaldata*/,
    amrex::Real /*mass_tolerance*/,
    GpuArray<int, AMREX_SPACEDIM> /*order_scheme_directional*/,
    GpuArray<int, AMREX_SPACEDIM> /*periodic*/)
{ /*
     const int lev = 0;
     const Geometry &geom = Geom(lev);
     auto &plev = GetParticles(lev);
     const auto dxi = geom.InvCellSizeArray();
     const auto dx = geom.CellSizeArray();
     const auto plo = geom.ProbLoArray();
     const auto domain = geom.Domain();

     const int *loarr = domain.loVect();
     const int *hiarr = domain.hiVect();

     int lo[] = {loarr[0], loarr[1], loarr[2]};
     int hi[] = {hiarr[0], hiarr[1], hiarr[2]};

     for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
     {
         const Box &nodalbox = mfi.validbox();

         Array4<Real> nodal_data_arr = nodaldata.array(mfi);

         amrex::ParallelFor(nodalbox,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                            {
                                nodal_data_arr(i, j, k, NORMALX) = shunya;
                                nodal_data_arr(i, j, k, NORMALY) = shunya;
                                nodal_data_arr(i, j, k, NORMALZ) = shunya;
                            });
     }

     for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
     {
         const amrex::Box &box = mfi.tilebox();
         Box nodalbox = convert(box, {AMREX_D_DECL(1, 1, 1)});

         int gid = mfi.index();
         int tid = mfi.LocalTileIndex();
         auto index = std::make_pair(gid, tid);

         auto &ptile = plev[index];
         auto &aos = ptile.GetArrayOfStructs();
         int np = aos.numRealParticles();
         int ng = aos.numNeighborParticles();
         int nt = np + ng;

         Array4<Real> nodal_data_arr = nodaldata.array(mfi);

         ParticleType *pstruct = aos().dataPtr();

         amrex::ParallelFor(
             nt,
             [=] AMREX_GPU_DEVICE(int i) noexcept
             {
                 int lmin, lmax, nmin, nmax, mmin, mmax;

                 ParticleType &p = pstruct[i];

                 if (p.idata(intData::phase) ==
                     0) // Compute only for standard particles
                        // and not rigid particles with phase=1
                 {

                     amrex::Real xp[AMREX_SPACEDIM];

                     xp[XDIR] = p.pos(XDIR);
                     xp[YDIR] = p.pos(YDIR);
                     xp[ZDIR] = p.pos(ZDIR);

                     auto iv = getParticleCell(p, plo, dxi, domain);

                     lmin = (order_scheme_directional[0] == 1)
                                ? 0
                                : ((order_scheme_directional[0] == 3)
                                       ? (iv[XDIR] == lo[XDIR])
                                             ? 0
                                             : ((iv[XDIR] == hi[XDIR]) ? -1 :
     -1) : -1000); lmax = (order_scheme_directional[0] == 1) ? 2 :
     ((order_scheme_directional[0] == 3) ? (iv[XDIR] == lo[XDIR]) ? lmin + 3 :
     ((iv[XDIR] == hi[XDIR]) ? lmin + 3 : lmin + 4) : -1000);

                     mmin = (order_scheme_directional[1] == 1)
                                ? 0
                                : ((order_scheme_directional[1] == 3)
                                       ? (iv[YDIR] == lo[YDIR])
                                             ? 0
                                             : ((iv[YDIR] == hi[YDIR]) ? -1 :
     -1) : -1000); mmax = (order_scheme_directional[1] == 1) ? 2 :
     ((order_scheme_directional[1] == 3) ? (iv[YDIR] == lo[YDIR]) ? mmin + 3 :
     ((iv[YDIR] == hi[YDIR]) ? mmin + 3 : mmin + 4) : -1000);

                     nmin = (order_scheme_directional[2] == 1)
                                ? 0
                                : ((order_scheme_directional[2] == 3)
                                       ? (iv[ZDIR] == lo[ZDIR])
                                             ? 0
                                             : ((iv[ZDIR] == hi[ZDIR]) ? -1 :
     -1) : -1000); nmax = (order_scheme_directional[2] == 1) ? 2 :
     ((order_scheme_directional[2] == 3) ? (iv[ZDIR] == lo[ZDIR]) ? nmin + 3 :
     ((iv[ZDIR] == hi[ZDIR]) ? nmin + 3 : nmin + 4) : -1000);

                     if (lmin == -1000 or lmax == -1000 or mmin == -1000 or
                         mmax == -1000 or nmin == -1000 or nmax == -1000)
                     {
                         amrex::Abort("\nError. Something wrong with min/max "
                                      "index values in "
                                      "deposit onto grid");
                     }

                     for (int n = nmin; n < nmax; n++)
                     {
                         for (int m = mmin; m < mmax; m++)
                         {
                             for (int l = lmin; l < lmax; l++)
                             {
                                 IntVect ivlocal(iv[XDIR] + l, iv[YDIR] + m,
                                                 iv[ZDIR] + n);
                                 if (nodalbox.contains(ivlocal))
                                 {

                                     amrex::Real basisval_grad[AMREX_SPACEDIM];
                                     for (int d = 0; d < AMREX_SPACEDIM; d++)
                                     {
                                         basisval_grad[d] = basisvalder(
                                             d, l, m, n, iv[XDIR], iv[YDIR],
                                             iv[ZDIR], xp, plo, dx,
                                             order_scheme_directional, periodic,
                                             lo, hi);
                                     }
                                     amrex::Real normal[AMREX_SPACEDIM] = {
                                         p.rdata(realData::mass) *
                                             basisval_grad[XDIR],
                                         p.rdata(realData::mass) *
                                             basisval_grad[YDIR],
                                         p.rdata(realData::mass) *
                                             basisval_grad[ZDIR]};
                                     for (int dim = 0; dim < AMREX_SPACEDIM;
                                          dim++)
                                     {
                                         amrex::Gpu::Atomic::AddNoRet(
                                             &nodal_data_arr(
                                                 iv[XDIR] + l, iv[YDIR] + m,
                                                 iv[ZDIR] + n, NORMALX + dim),
                                             normal[dim]);
                                     }
                                 }
                             }
                         }
                     }
                 }
             });
     }

     for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
     {
         const amrex::Box &box = mfi.tilebox();
         Box nodalbox = convert(box, {AMREX_D_DECL(1, 1, 1)});

         Array4<Real> nodal_data_arr = nodaldata.array(mfi);

         amrex::ParallelFor(
             nodalbox,
             [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
             {
                 amrex::Real nmag = pow((nodal_data_arr(i, j, k, NORMALX) *
     nodal_data_arr(i, j, k, NORMALX) + nodal_data_arr(i, j, k, NORMALY) *
     nodal_data_arr(i, j, k, NORMALY) + nodal_data_arr(i, j, k, NORMALZ) *
                                             nodal_data_arr(i, j, k, NORMALZ)),
                                        0.5);

                 if (nmag > mass_tolerance)
                 {
                     for (int d = 0; d < AMREX_SPACEDIM; d++)
                     {
                         nodal_data_arr(i, j, k, NORMALX + d) =
                             nodal_data_arr(i, j, k, NORMALX + d) / nmag;
                     }
                 }
                 else
                 {
                     nodal_data_arr(i, j, k, NORMALX) = 0.0;
                     nodal_data_arr(i, j, k, NORMALY) = 0.0;
                     nodal_data_arr(i, j, k, NORMALZ) = 0.0;
                 }
             });
     }*/
}
