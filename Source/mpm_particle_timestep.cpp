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

#ifndef __CUDA_ARCH__
                // Step 2 diagnostic: print first 5 particles evaluated
                {
                    static int print_count = 0;
                    if (print_count < 5)
                    {
                        amrex::AllPrint()
                            << "DEBUG particle: Cs=" << Cs
                            << " velmag=" << velmag
                            << " density=" << p.rdata(realData::density)
                            << " Bulk_modulus="
                            << p.rdata(realData::Bulk_modulus)
                            << " constitutive_model="
                            << p.idata(intData::constitutive_model) << "\n";
                        ++print_count;
                    }
                }
#endif
                amrex::Real dt_p = dxmin / (Cs + velmag);
                // Fix 6: guard against NaN/Inf/zero from degenerate particles
                if (dt_p <= 0.0)
                {
#ifndef __CUDA_ARCH__
                    amrex::AllPrint()
                        << "WARNING: bad dt_p=" << dt_p << " Cs=" << Cs
                        << " velmag=" << velmag
                        << " density=" << p.rdata(realData::density)
                        << " constitutive_model="
                        << p.idata(intData::constitutive_model) << " pos=("
                        << p.pos(0) << "," << p.pos(1) << ")\n";
#endif
                    return amrex::Real(1.0e10); // exclude from min reduction
                }
                // Rogue particle diagnostic: fires from inside the GPU kernel
                // (CUDA printf is device-safe; amrex::AllPrint for CPU builds).
                // Threshold on raw dt_p (pre-CFL): 5e-5 → Cs+v > dxmin/5e-5.
                if (dt_p < 5.0e-5)
                {
#ifdef __CUDA_ARCH__
                    printf("ROGUE GPU: dt_p=%.6e Cs=%.6e velmag=%.6e"
                           " density=%.6e Bulk_modulus=%.6e pressure=%.6e"
                           " jacobian=%.6e volume=%.6e vol_init=%.6e"
                           " cm=%d pos=(%.6f,%.6f)\n",
                           (double)dt_p,
                           (double)Cs,
                           (double)velmag,
                           (double)p.rdata(realData::density),
                           (double)p.rdata(realData::Bulk_modulus),
                           (double)p.rdata(realData::pressure),
                           (double)p.rdata(realData::jacobian),
                           (double)p.rdata(realData::volume),
                           (double)p.rdata(realData::vol_init),
                           (int)p.idata(intData::constitutive_model),
                           (double)p.pos(0),
                           (double)p.pos(1));
#else
                    amrex::AllPrint()
                        << "ROGUE CPU: dt_p=" << dt_p
                        << " Cs=" << Cs
                        << " velmag=" << velmag
                        << " density=" << p.rdata(realData::density)
                        << " Bulk_modulus=" << p.rdata(realData::Bulk_modulus)
                        << " pressure=" << p.rdata(realData::pressure)
                        << " jacobian=" << p.rdata(realData::jacobian)
                        << " volume=" << p.rdata(realData::volume)
                        << " vol_init=" << p.rdata(realData::vol_init)
                        << " cm=" << p.idata(intData::constitutive_model)
                        << " pos=(" << p.pos(0) << "," << p.pos(1) << ")\n";
#endif
                }
                return dt_p;
            }
            return std::numeric_limits<amrex::Real>::max();
        });

#ifdef BL_USE_MPI
    ParallelDescriptor::ReduceRealMin(dt);
#endif

    // Step 1 diagnostics: host-side, printed once per Calculate_time_step call
    {
        amrex::Real dxmin_host = dx[0];
        for (int d = 1; d < AMREX_SPACEDIM; ++d)
            dxmin_host = amrex::min(dxmin_host, dx[d]);
        amrex::Print() << "DEBUG dxmin=" << dxmin_host << "\n";
        amrex::Print() << "DEBUG CFL=" << specs.CFL << "\n";
        amrex::Print() << "DEBUG dt_before_clamp=" << specs.CFL * dt << "\n";
        amrex::Print() << "DEBUG dt_max_limit=" << specs.dt_max_limit << "\n";
        amrex::Print() << "DEBUG dt_min_limit=" << specs.dt_min_limit << "\n";
    }

    // Host-side anomaly marker: the per-particle ROGUE lines are printed
    // directly from the GPU kernel above via CUDA printf (device-safe).
    // The host-side particle loop was removed — it segfaulted because this
    // AMReX build uses device (non-managed) memory for particle arrays.
    if (dt < 5.0e-5)
    {
        amrex::Gpu::streamSynchronize(); // flush CUDA printf buffer
        amrex::Print() << "ANOMALOUS dt_raw=" << dt
                       << " (ROGUE GPU lines printed above)\n";
    }

    dt = specs.CFL * dt;
    dt = std::max(std::min(dt, specs.dt_max_limit), specs.dt_min_limit);
    amrex::Print() << "DEBUG dt_final=" << dt << "\n";
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
        // Fix 2: iterate real particles only — neighbor slots have stale F
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

                    // Fix 1: guard against inverted/degenerate particles
                    if (detF > 1.0e-14)
                    {
                        p.rdata(realData::jacobian) = detF;
                        p.rdata(realData::volume) =
                            p.rdata(realData::vol_init) * detF;
                        p.rdata(realData::density) =
                            p.rdata(realData::mass) / p.rdata(realData::volume);
                    }
                    else
                    {
                        // Degenerate particle — clamp to avoid Inf/NaN
                        // propagation. Keep previous volume/density unchanged.
#ifndef __CUDA_ARCH__
                        amrex::AllPrint()
                            << "WARNING: detF <= 0 at pos=(" << p.pos(0) << ","
                            << p.pos(1) << ") detF=" << detF << "\n";
#endif
                    }
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
 * @param[in] lsetbc          Boundary condition type for level‑set EB.
 * @param[in] wall_mu_lo      Friction coefficients at low faces.
 * @param[in] wall_mu_hi      Friction coefficients at high faces.
 * @param[in] wall_vel_lo     Wall velocities at low faces (flattened array).
 * @param[in] wall_vel_hi     Wall velocities at high faces (flattened array).
 * @param[in] lset_wall_mu    Friction coefficient for level‑set EB.
 *
 * @return None.
 */

void MPMParticleContainer::moveParticles(
    const amrex::Real &dt,
    int bclo[AMREX_SPACEDIM],
    int bchi[AMREX_SPACEDIM],
    [[maybe_unused]] int lsetbc,
    amrex::Real wall_mu_lo[AMREX_SPACEDIM],
    amrex::Real wall_mu_hi[AMREX_SPACEDIM],
    amrex::Real wall_vel_lo[AMREX_SPACEDIM * AMREX_SPACEDIM],
    amrex::Real wall_vel_hi[AMREX_SPACEDIM * AMREX_SPACEDIM],
    [[maybe_unused]] amrex::Real lset_wall_mu)
{
    BL_PROFILE("MPMParticleContainer::moveParticles");

    const int lev = 0;
    const auto plo = Geom(lev).ProbLoArray();
    const auto phi = Geom(lev).ProbHiArray();
    const auto dx = Geom(lev).CellSizeArray();
    auto &plev = GetParticles(lev);

#if USE_EB
    bool using_levsets = mpm_ebtools::using_levelset_geometry;
    int lsref = mpm_ebtools::ls_refinement;

    // ------------------------------------------------------------------
    // Bug 1 + Bug 4 fix: average_down_nodal + FillBoundary
    //
    // lsphi is stored at ls_refinement * coarse resolution.
    // Indexing it directly with a coarse particle MFIter (Bug 4) gives
    // the wrong tile.  Ghost cells are uninitialised without FillBoundary
    // (Bug 1) => garbage phi/grad values near tile boundaries.
    //
    // Fix: coarsen onto a nodal MultiFab matching the particle iterator,
    // fill ghost cells, pass lsref=1 to get_levelset_value/grad.
    // ------------------------------------------------------------------
    amrex::MultiFab lsphi_coarse;
    if (using_levsets)
    {
        // Derive coarse nodal BA/DM directly from lsphi (which is refined).
        // Coarsen lsphi's BA by lsref to get the coarse nodal layout.
        BoxArray nodal_ba_coarse = mpm_ebtools::lsphi->boxArray();
        nodal_ba_coarse.coarsen(lsref);
        lsphi_coarse.define(nodal_ba_coarse,
                            mpm_ebtools::lsphi->DistributionMap(),
                            1,  // ncomp
                            1); // nghost — must be >= 1
        amrex::average_down_nodal(*mpm_ebtools::lsphi, lsphi_coarse,
                                  amrex::IntVect(lsref));
        lsphi_coarse.FillBoundary(Geom(lev).periodicity());
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

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        auto &ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto &aos = ptile.GetArrayOfStructs();
        // Fix 7: iterate real particles only — writing positions / xvel back to
        // neighbor slots corrupts wall-BC reflections and CUDA ReduceMin when
        // stale xvel_prime of a neighbor particle near a wall triggers an
        // incorrect reflection (same class of bug as Fix 2 / Fix 3).
        const size_t np = aos.numRealParticles();
        ParticleType *pstruct = aos().dataPtr();

#if USE_EB
        amrex::Array4<amrex::Real> lsetarr;
        if (using_levsets)
        {
            // Bug 4 fix: use lsphi_coarse, NOT mpm_ebtools::lsphi->array(mfi)
            lsetarr = lsphi_coarse.array(mfi);
        }
#endif

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
                // Levelset BC

                if (using_levsets && p.idata(intData::phase) == 0)
                {
                    amrex::Real xp[AMREX_SPACEDIM];
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        xp[d] = p.pos(d);

                    // lsphi_coarse is at coarse resolution: lsref=1
                    amrex::Real dist =
                        get_levelset_value(lsetarr, plo, dx, xp, /*lsref=*/1);

                    if (dist < 0.0)
                    {
                        amrex::Real normaldir[AMREX_SPACEDIM];
                        get_levelset_grad(lsetarr, plo, dx, xp, /*lsref=*/1,
                                          normaldir);

                        amrex::Real gradmag = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            gradmag += normaldir[d] * normaldir[d];
                        gradmag = std::sqrt(gradmag);

                        // Bug 3 fix: skip degenerate nodes deep inside
                        // obstacle. Dividing by TINYVAL~1e-20 gives ~1e20
                        // normals.
                        if (gradmag < 1.0e-10)
                            return;

                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            normaldir[d] /= gradmag;

                        // Only apply BC when particle moves into the wall.
                        // Level-set gradient points outward: veln > 0 means
                        // particle is already leaving — leave it alone.
                        amrex::Real veln = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            veln += relvel_in[d] * normaldir[d];

                        if (veln <= 0.0)
                        {
                            int modify_pos =
                                applybc(relvel_in, relvel_out, lset_wall_mu,
                                        normaldir, lsetbc);
                            if (modify_pos)
                            {
                                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                    p.pos(d) += 2.0 * amrex::Math::abs(dist) *
                                                normaldir[d];
                            }
                            for (int d = 0; d < AMREX_SPACEDIM; ++d)
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
                    p.rdata(realData::xvel + dir) =
                        relvel_out[dir] + wallvel[dir];
                }
            });
    }
}

/**
 * @brief Returns the maximum Y‑coordinate among all particles.
 *
 * This is typically used to track the top surface of a deforming body
 * (e.g., for spring compression diagnostics).
 *
 * @return amrex::Real  Maximum particle y‑position.
 */

amrex::Real MPMParticleContainer::GetPosSpring()
{

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

/**
 * @brief Returns the minimum Y‑coordinate among all rigid‑body‑0 particles.
 *
 * Performs a GPU‑parallel reduction over all particles with phase = 1 and
 * rigid_body_id = 0 (the piston), returning the lowest Y position. All
 * other particles contribute +∞ to the reduction so they are ignored.
 *
 * @return amrex::Real  Minimum Y‑position of the piston rigid body.
 */
amrex::Real MPMParticleContainer::GetPosPiston()
{
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
                          