// clang-format off
#include <mpm_particle_container.H>
#include <interpolants.H>
#include <mpm_eb.H>
#include <mpm_kernels.H>
#include <aesthetics.H>
// clang-format on

/**
 * @brief Computes the stable time step for the MPM update using a CFL
 * condition.
 *
 * If a fixed time step is requested in the input specs, that value is returned.
 * Otherwise, this routine performs a parallel reduction over all material
 * particles (phase = 0) to compute:
 *
 *   - The elastic or acoustic wave speed:
 *        Cs = sqrt( (λ + 2μ) / ρ )     for solids
 *        Cs = sqrt( K / ρ )           for fluids
 *
 *   - The particle velocity magnitude |v|
 *
 *   - The minimum cell size dx_min across dimensions
 *
 * The local stable time step is:
 *        dt_p = dx_min / (Cs + |v|)
 *
 * The global time step is:
 *        dt = CFL * min_p(dt_p)
 * and is clamped to [dt_min_limit, dt_max_limit].
 *
 * @param[in] specs  Simulation specification structure containing CFL and
 * limits.
 *
 * @return amrex::Real  The stable time step for the next update.
 */

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

                amrex::Real dt_p = dxmin / (Cs + velmag);
                return dt_p;
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

/**
 * @brief Updates particle volume, density, and Jacobian from the deformation
 * gradient.
 *
 * For each material particle (phase = 0), this routine:
 *
 *   1. Reconstructs the deformation gradient F from particle storage.
 *   2. Computes det(F) in 1D, 2D, or 3D.
 *   3. Updates:
 *        J = det(F)
 *        V = V₀ * J
 *        ρ = m / V
 *
 * This is the standard MPM kinematic update for compressible materials.
 *
 * @return None.
 */

void MPMParticleContainer::updateVolume()
{
    BL_PROFILE("MPMParticleContainer::updateVolume");
    const int lev = 0;
    auto &plev = GetParticles(lev);

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        auto &ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto &aos = ptile.GetArrayOfStructs();
        const int np = aos.numRealParticles();
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

/**
 * @brief Advances particle positions and applies boundary conditions.
 *
 * For each particle:
 *
 *   1. **Position update**:
 *        xᵖ ← xᵖ + vᵖ' dt
 *
 *   2. **Builds relative velocity** for boundary condition evaluation.
 *
 *   3. **Embedded boundary (level‑set) handling** (if enabled):
 *        - Computes φ(xᵖ) and ∇φ(xᵖ)
 *        - Applies applybc() using the level‑set normal
 *        - Optionally projects particle out of the EB surface
 *
 *   4. **Domain boundary conditions**:
 *        - Detects crossings at low/high faces
 *        - Subtracts wall velocity
 *        - Applies applybc() with appropriate wall friction μ
 *        - Reflects position if required
 *        - Restores wall velocity to outgoing velocity
 *
 * Boundary types include:
 *   - No‑slip
 *   - Slip
 *   - Partial slip (Coulomb friction)
 *   - Periodic
 *
 * @param[in] dt              Time step.
 * @param[in] bclo            Boundary condition types at low faces.
 * @param[in] bchi            Boundary condition types at high faces.
 * @param[in] wall_mu_lo      Friction coefficients at low faces.
 * @param[in] wall_mu_hi      Friction coefficients at high faces.
 * @param[in] wall_vel_lo     Wall velocities at low faces (flattened array).
 * @param[in] wall_vel_hi     Wall velocities at high faces (flattened array).
 *
 * Level-set BC type and friction coefficient are read per-body from
 * mpm_ebtools::ls_bodies (set by init_eb() from the input file).
 *
 * @return None.
 */

void MPMParticleContainer::moveParticles(
    const amrex::Real &dt,
    int bclo[AMREX_SPACEDIM],
    int bchi[AMREX_SPACEDIM],
    amrex::Real wall_mu_lo[AMREX_SPACEDIM],
    amrex::Real wall_mu_hi[AMREX_SPACEDIM],
    amrex::Real wall_vel_lo[AMREX_SPACEDIM * AMREX_SPACEDIM],
    amrex::Real wall_vel_hi[AMREX_SPACEDIM * AMREX_SPACEDIM],
    amrex::GpuArray<const amrex::Real *, AMREX_SPACEDIM> udf_wall_vel_lo_dev,
    amrex::GpuArray<const amrex::Real *, AMREX_SPACEDIM> udf_wall_vel_hi_dev)
{
    BL_PROFILE("MPMParticleContainer::moveParticles");

    const int lev = 0;
    const auto plo = Geom(lev).ProbLoArray();
    const auto p_hi = Geom(lev).ProbHiArray();
#if (AMREX_SPACEDIM >= 2)
    const auto dx = Geom(lev).CellSizeArray();
#endif
    auto &plev = GetParticles(lev);

#if USE_EB
    bool using_levsets = mpm_ebtools::using_levelset_geometry;
    const int num_bodies = static_cast<int>(mpm_ebtools::ls_bodies.size());

    amrex::GpuArray<int, EXAGOOP_MAX_LS_BODIES> body_refs;
    amrex::GpuArray<int, EXAGOOP_MAX_LS_BODIES> body_bcs;
    amrex::GpuArray<amrex::Real, EXAGOOP_MAX_LS_BODIES> body_mus;
    for (int b = 0; b < num_bodies; ++b)
    {
        body_refs[b] = 1;
        body_bcs[b] = mpm_ebtools::ls_bodies[b].mom_bc_int();
        body_mus[b] = mpm_ebtools::ls_bodies[b].wall_mu;
    }

    amrex::Vector<amrex::MultiFab> lsphi_coarse(num_bodies);
    if (using_levsets)
    {
        for (int b = 0; b < num_bodies; ++b)
        {
            int lsref = mpm_ebtools::ls_bodies[b].ls_refinement;
            amrex::BoxArray coarse_ba =
                mpm_ebtools::ls_bodies[b].lsphi->boxArray();
            coarse_ba.coarsen(lsref);
            lsphi_coarse[b].define(
                coarse_ba,
                mpm_ebtools::ls_bodies[b].lsphi->DistributionMap(),
                1, 1);
            amrex::average_down_nodal(
                *mpm_ebtools::ls_bodies[b].lsphi,
                lsphi_coarse[b],
                amrex::IntVect(lsref));
            lsphi_coarse[b].FillBoundary(Geom(lev).periodicity());
        }
    }
#endif

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

    const auto domain = Geom(lev).Domain();
    GpuArray<int, AMREX_SPACEDIM> ncells_arr;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        ncells_arr[d] = domain.length(d);

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        auto &ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto &aos = ptile.GetArrayOfStructs();
        const size_t np = aos.numRealParticles();
        ParticleType *pstruct = aos().dataPtr();

#if USE_EB
        amrex::GpuArray<amrex::Array4<amrex::Real>, EXAGOOP_MAX_LS_BODIES>
            body_arrs;
        if (using_levsets)
        {
            for (int b = 0; b < num_bodies; ++b)
                body_arrs[b] = lsphi_coarse[b].array(mfi);
        }
#endif

        const amrex::GpuArray<const amrex::Real *, AMREX_SPACEDIM> udf_lo_ptrs =
            udf_wall_vel_lo_dev;
        const amrex::GpuArray<const amrex::Real *, AMREX_SPACEDIM> udf_hi_ptrs =
            udf_wall_vel_hi_dev;

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

#if USE_EB

                if (using_levsets && p.idata(intData::phase) == 0)
                {
                    amrex::Real xp[AMREX_SPACEDIM];
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        xp[d] = p.pos(d);

                    int hit_body = -1;
                    amrex::Real min_phi = TINYVAL;

                    for (int b = 0; b < num_bodies; ++b)
                    {
                        amrex::Real phi = get_levelset_value(
                            body_arrs[b], plo, dx, xp, body_refs[b]);
                        if (phi < min_phi)
                        {
                            min_phi = phi;
                            hit_body = b;
                        }
                    }

                    if (hit_body >= 0)
                    {
                        amrex::Real normaldir[AMREX_SPACEDIM];
                        get_levelset_grad(body_arrs[hit_body], plo, dx, xp,
                                          body_refs[hit_body], normaldir);

                        amrex::Real gradmag = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            gradmag += normaldir[d] * normaldir[d];
                        gradmag = std::sqrt(gradmag);
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            normaldir[d] /= (gradmag + TINYVAL);

                        int modify_pos =
                            applybc(relvel_in, relvel_out, body_mus[hit_body],
                                    normaldir, body_bcs[hit_body]);
                        if (modify_pos)
                        {
                            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            {
                                p.pos(d) += 2.0 * amrex::Math::abs(min_phi) *
                                            normaldir[d];
                            }
                        }
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            p.rdata(realData::xvel + d) = relvel_out[d];
                        }
                    }
                }
#endif

                // Domain boundary BCs
                amrex::Real wallvel[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    wallvel[d] = 0.0;

                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    if (p.pos(dir) < plo[dir])
                    {
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            wallvel[d] =
                                wall_vel_lo_arr[dir * AMREX_SPACEDIM + d];

                        if (udf_lo_ptrs[dir] != nullptr)
                        {
#if (AMREX_SPACEDIM >= 2)
                            int p0 = (dir == 0) ? 1 : 0;
#endif
#if (AMREX_SPACEDIM == 2)
                            amrex::Real frac = (p.pos(p0) - plo[p0]) / dx[p0];
                            frac = amrex::max(
                                amrex::Real(0.0),
                                amrex::min(amrex::Real(ncells_arr[p0]), frac));
                            int j0 = (int)frac;
                            int j1 = amrex::min(j0 + 1, ncells_arr[p0]);
                            amrex::Real wt1 = frac - j0;
                            amrex::Real wt0 = amrex::Real(1.0) - wt1;
                            for (int c = 0; c < AMREX_SPACEDIM; ++c)
                                wallvel[c] =
                                    wt0 * udf_lo_ptrs[dir]
                                                     [j0 * AMREX_SPACEDIM + c] +
                                    wt1 * udf_lo_ptrs[dir]
                                                     [j1 * AMREX_SPACEDIM + c];
#elif (AMREX_SPACEDIM == 3)
                            int p1 = (dir == 2) ? 1 : 2;
                            amrex::Real frac0 = (p.pos(p0) - plo[p0]) / dx[p0];
                            amrex::Real frac1 = (p.pos(p1) - plo[p1]) / dx[p1];
                            frac0 = amrex::max(
                                amrex::Real(0.0),
                                amrex::min(amrex::Real(ncells_arr[p0]), frac0));
                            frac1 = amrex::max(
                                amrex::Real(0.0),
                                amrex::min(amrex::Real(ncells_arr[p1]), frac1));
                            int j0 = (int)frac0;
                            int j1 = amrex::min(j0 + 1, ncells_arr[p0]);
                            int k0 = (int)frac1;
                            int k1 = amrex::min(k0 + 1, ncells_arr[p1]);
                            amrex::Real wx1 = frac0 - j0;
                            amrex::Real wx0 = amrex::Real(1.0) - wx1;
                            amrex::Real wy1 = frac1 - k0;
                            amrex::Real wy0 = amrex::Real(1.0) - wy1;
                            int n1 = ncells_arr[p1] + 1;
                            for (int c = 0; c < AMREX_SPACEDIM; ++c)
                                wallvel[c] =
                                    wx0 * wy0 *
                                        udf_lo_ptrs[dir][(j0 * n1 + k0) *
                                                             AMREX_SPACEDIM +
                                                         c] +
                                    wx0 * wy1 *
                                        udf_lo_ptrs[dir][(j0 * n1 + k1) *
                                                             AMREX_SPACEDIM +
                                                         c] +
                                    wx1 * wy0 *
                                        udf_lo_ptrs[dir][(j1 * n1 + k0) *
                                                             AMREX_SPACEDIM +
                                                         c] +
                                    wx1 * wy1 *
                                        udf_lo_ptrs[dir][(j1 * n1 + k1) *
                                                             AMREX_SPACEDIM +
                                                         c];
#endif
                        }

                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            relvel_in[d] -= wallvel[d];

                        amrex::Real normaldir[AMREX_SPACEDIM] = {0};
                        normaldir[dir] = 1.0;
                        int modify_pos =
                            applybc(relvel_in, relvel_out, wall_mu_lo_arr[dir],
                                    normaldir, bc_lo_arr[dir]);
                        if (modify_pos)
                        {
                            p.pos(dir) = 2.0 * plo[dir] - p.pos(dir);
                        }
                        for (int c = 0; c < AMREX_SPACEDIM; ++c)
                            p.rdata(realData::xvel + c) =
                                relvel_out[c] + wallvel[c];
                    }
                    else if (p.pos(dir) > p_hi[dir])
                    {
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            wallvel[d] =
                                wall_vel_hi_arr[dir * AMREX_SPACEDIM + d];

                        if (udf_hi_ptrs[dir] != nullptr)
                        {
#if (AMREX_SPACEDIM >= 2)
                            int p0 = (dir == 0) ? 1 : 0;
#endif
#if (AMREX_SPACEDIM == 2)
                            amrex::Real frac = (p.pos(p0) - plo[p0]) / dx[p0];
                            frac = amrex::max(
                                amrex::Real(0.0),
                                amrex::min(amrex::Real(ncells_arr[p0]), frac));
                            int j0 = (int)frac;
                            int j1 = amrex::min(j0 + 1, ncells_arr[p0]);
                            amrex::Real wt1 = frac - j0;
                            amrex::Real wt0 = amrex::Real(1.0) - wt1;
                            for (int c = 0; c < AMREX_SPACEDIM; ++c)
                                wallvel[c] =
                                    wt0 * udf_hi_ptrs[dir]
                                                     [j0 * AMREX_SPACEDIM + c] +
                                    wt1 * udf_hi_ptrs[dir]
                                                     [j1 * AMREX_SPACEDIM + c];
#elif (AMREX_SPACEDIM == 3)
                            int p1 = (dir == 2) ? 1 : 2;
                            amrex::Real frac0 = (p.pos(p0) - plo[p0]) / dx[p0];
                            amrex::Real frac1 = (p.pos(p1) - plo[p1]) / dx[p1];
                            frac0 = amrex::max(
                                amrex::Real(0.0),
                                amrex::min(amrex::Real(ncells_arr[p0]), frac0));
                            frac1 = amrex::max(
                                amrex::Real(0.0),
                                amrex::min(amrex::Real(ncells_arr[p1]), frac1));
                            int j0 = (int)frac0;
                            int j1 = amrex::min(j0 + 1, ncells_arr[p0]);
                            int k0 = (int)frac1;
                            int k1 = amrex::min(k0 + 1, ncells_arr[p1]);
                            amrex::Real wx1 = frac0 - j0;
                            amrex::Real wx0 = amrex::Real(1.0) - wx1;
                            amrex::Real wy1 = frac1 - k0;
                            amrex::Real wy0 = amrex::Real(1.0) - wy1;
                            int n1 = ncells_arr[p1] + 1;
                            for (int c = 0; c < AMREX_SPACEDIM; ++c)
                                wallvel[c] =
                                    wx0 * wy0 *
                                        udf_hi_ptrs[dir][(j0 * n1 + k0) *
                                                             AMREX_SPACEDIM +
                                                         c] +
                                    wx0 * wy1 *
                                        udf_hi_ptrs[dir][(j0 * n1 + k1) *
                                                             AMREX_SPACEDIM +
                                                         c] +
                                    wx1 * wy0 *
                                        udf_hi_ptrs[dir][(j1 * n1 + k0) *
                                                             AMREX_SPACEDIM +
                                                         c] +
                                    wx1 * wy1 *
                                        udf_hi_ptrs[dir][(j1 * n1 + k1) *
                                                             AMREX_SPACEDIM +
                                                         c];
#endif
                        }

                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            relvel_in[d] -= wallvel[d];

                        amrex::Real normaldir[AMREX_SPACEDIM] = {0};
                        normaldir[dir] = -1.0;
                        int modify_pos =
                            applybc(relvel_in, relvel_out, wall_mu_hi_arr[dir],
                                    normaldir, bc_hi_arr[dir]);
                        if (modify_pos)
                        {
                            p.pos(dir) = 2.0 * p_hi[dir] - p.pos(dir);
                        }
                        for (int c = 0; c < AMREX_SPACEDIM; ++c)
                            p.rdata(realData::xvel + c) =
                                relvel_out[c] + wallvel[c];
                    }
                }
            });
    }
}

/**
 * @brief Assigns a uniform velocity to all particles belonging to a rigid body.
 *
 * For each particle with:
 *      phase = 1  (rigid)
 *      rigid_body_id = rigid_body_id
 * the routine sets:
 *      vᵖ' = prescribed velocity
 *
 * This is used to drive rigid pistons, platens, or moving boundaries.
 *
 * @param[in] rigid_body_id  ID of the rigid body to update.
 * @param[in] velocity       Prescribed velocity vector.
 *
 * @return None.
 */

void MPMParticleContainer::UpdateRigidParticleVelocities(
    int rigid_body_id, Array<amrex::Real, AMREX_SPACEDIM> velocity)
{
    BL_PROFILE("MPMParticleContainer::GetVelPiston");

    const int lev = 0;
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
