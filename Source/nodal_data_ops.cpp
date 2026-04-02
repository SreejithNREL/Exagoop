// clang-format off
#include <nodal_data_ops.H>
#include <mpm_eb.H>
#include <mpm_kernels.H>
// clang-format on

using namespace amrex;

/**
 * @brief Writes nodal grid data to a single‑level plotfile.
 *
 * Converts nodal (node‑centered) data to cell‑centered form using
 * average_node_to_cellcenter(), then writes the resulting MultiFab
 * to a plotfile directory.
 *
 * @param[in] fname        Output plotfile name.
 * @param[in] nodaldata    Nodal MultiFab to be written.
 * @param[in] fieldnames   Names of each component in nodaldata.
 * @param[in] geom         Geometry describing the domain.
 * @param[in] ba           BoxArray for the output MultiFab.
 * @param[in] dm           DistributionMapping for parallel layout.
 * @param[in] time         Simulation time for the plotfile.
 *
 * @return None.
 */

void write_grid_file(std::string fname,
                     MultiFab &nodaldata,
                     Vector<std::string> fieldnames,
                     Geometry geom,
                     BoxArray ba,
                     DistributionMapping dm,
                     Real time)
{
    MultiFab plotmf(ba, dm, nodaldata.nComp(), 0);
    average_node_to_cellcenter(plotmf, 0, nodaldata, 0, nodaldata.nComp());
    WriteSingleLevelPlotfile(fname, plotmf, fieldnames, geom, time, 0);
}

/**
 * @brief Stores the current nodal mass and velocity for later PIC/FLIP updates.
 *
 * For each node with positive mass:
 *   - MASS_OLD_INDEX ← MASS_INDEX
 *   - DELTA_VELX_INDEX[d] ← VELX_INDEX[d]
 *
 * This is used to compute Δv during the next P2G/G2P cycle.
 *
 * @param[in,out] nodaldata  Nodal MultiFab containing mass and velocity fields.
 *
 * @return None.
 */

void backup_current_velocity(MultiFab &nodaldata)
{
    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        // const Box &bx = mfi.validbox();
        //  Box nodalbox = convert(bx, {AMREX_D_DECL(1, 1, 1)});
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));

        Array4<Real> nodal_data_arr = nodaldata.array(mfi);

        amrex::ParallelFor(
            nodalbox,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                if (nodal_data_arr(i, j, k, MASS_INDEX) > shunya)
                {
                    nodal_data_arr(i, j, k, MASS_OLD_INDEX) =
                        nodal_data_arr(i, j, k, MASS_INDEX);
                    for (int d = 0; d < AMREX_SPACEDIM; d++)
                    {
                        nodal_data_arr(i, j, k, DELTA_VELX_INDEX + d) =
                            nodal_data_arr(i, j, k, VELX_INDEX + d);
                    }
                }
            });
    }
}

#if USE_TEMP
/**
 * @brief Stores the current nodal temperature for later ΔT updates.
 *
 * For each node with positive thermal mass (MASS_SPHEAT):
 *   DELTA_TEMPERATURE ← TEMPERATURE
 *
 * @param[in,out] nodaldata  Nodal MultiFab containing thermal fields.
 *
 * @return None.
 */

void backup_current_temperature(MultiFab &nodaldata)
{

    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));

        Array4<Real> nodal_data_arr = nodaldata.array(mfi);
        amrex::ParallelFor(nodalbox,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           {
                               if (nodal_data_arr(i, j, k, MASS_SPHEAT) >
                                   shunya)
                               {
                                   nodal_data_arr(i, j, k, DELTA_TEMPERATURE) =
                                       nodal_data_arr(i, j, k, TEMPERATURE);
                               }
                           });
    }
}
#endif

#if USE_EB
/**
 * @brief Applies level‑set‑based boundary conditions to nodal velocities.
 *
 * For each node with mass > 0 whose level‑set value φ < 0 (i.e. inside or on
 * the EB surface):
 *   1. Maps the node to its location on the refined level‑set grid.
 *   2. Computes the physical position of the node.
 *   3. Evaluates ∇φ via multilinear interpolation and normalises to get the
 *      outward wall normal n̂.
 *   4. Subtracts the prescribed wall velocity to form the relative velocity.
 *   5. Calls applybc() with lsetbc and lset_wall_mu — exactly mirroring the
 *      treatment in nodal_bcs() — to enforce one of:
 *        - lsetbc == 0 : no BC (pass-through, useful for debugging)
 *        - lsetbc == 1 : no-slip  (kill all relative velocity)
 *        - lsetbc == 2 : no-penetration / free-slip
 *        - lsetbc == 3 : Coulomb friction (μ = lset_wall_mu)
 *   6. Restores the wall velocity to give the final nodal velocity.
 *
 * The wall velocity is currently zero (static EB). For a moving EB extend
 * lset_wall_vel[] via ParmParse and pass it in — the restore step already
 * handles it correctly.
 *
 * @param[in,out] nodaldata     Nodal MultiFab containing velocity fields.
 * @param[in]     geom          Geometry describing the domain.
 * @param[in]     dt            Time step (reserved for future use).
 * @param[in]     lsetbc        BC type integer
 * (0=none,1=noslip,2=slip,3=Coulomb).
 * @param[in]     lset_wall_mu  Friction coefficient μ (used when lsetbc == 3).
 *
 * @return None.
 */
void nodal_levelset_bcs(MultiFab &nodaldata,
                        const Geometry geom,
                        amrex::Real & /*dt*/,
                        int lsetbc,
                        amrex::Real lset_wall_mu)
{
    int lsref = mpm_ebtools::ls_refinement;
    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();

    // Wall velocity for a static EB is zero.
    // For a moving EB, replace with the prescribed surface velocity.
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> wall_vel = {
        AMREX_D_DECL(0.0, 0.0, 0.0)};

    // ── Coarsen lsphi to match nodaldata's BoxArray
    // ─────────────────────────── lsphi lives on a refined nodal BoxArray
    // (coarse × lsref). nodaldata lives on the coarse nodal BoxArray. Calling
    // lsphi->array(mfi) with a coarse MFIter gives the wrong tile. Fix:
    // average_down_nodal produces a coarse copy that shares the same BoxArray
    // as nodaldata, making array(mfi) correct and safe. get_levelset_value/grad
    // are then called with lsref=1 since the array is already at coarse
    // resolution.
    MultiFab lsphi_coarse(nodaldata.boxArray(), nodaldata.DistributionMap(),
                          1,  // ncomp
                          1); // nghost

    amrex::average_down_nodal(*mpm_ebtools::lsphi, lsphi_coarse,
                              amrex::IntVect(lsref));

    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));

        Array4<Real> nodal_data_arr = nodaldata.array(mfi);

        // lsphi_coarse shares BoxArray with nodaldata — array(mfi) is correct
        Array4<Real> lsarr = lsphi_coarse.array(mfi);

        amrex::ParallelFor(
            nodalbox,
            [=] AMREX_GPU_DEVICE(AMREX_D_DECL(int i, int j, int k)) noexcept
            {
                IntVect nodeid(AMREX_D_DECL(i, j, k));

                // Physical position of this node on the coarse grid
                amrex::Real xp[AMREX_SPACEDIM] = {AMREX_D_DECL(
                    plo[XDIR] + i * dx[XDIR], plo[YDIR] + j * dx[YDIR],
                    plo[ZDIR] + k * dx[ZDIR])};

                // Sample phi at this node using lsref=1 (coarse array)
                amrex::Real lsval = get_levelset_value(lsarr, plo, dx, xp, 1);

                // Only act on nodes with material mass that are inside/on EB
                if (lsval >= 0.0 ||
                    nodal_data_arr(nodeid, MASS_INDEX) <= shunya)
                    return;

                // --- Compute outward wall normal from ∇φ ---
                amrex::Real normaldir[AMREX_SPACEDIM] = {
                    AMREX_D_DECL(1.0, 0.0, 0.0)};

                // lsref=1 since lsphi_coarse is at coarse resolution
                get_levelset_grad(lsarr, plo, dx, xp, 1, normaldir);

                amrex::Real gradmag = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; d++)
                    gradmag += normaldir[d] * normaldir[d];
                gradmag = std::sqrt(gradmag);

                for (int d = 0; d < AMREX_SPACEDIM; d++)
                    normaldir[d] /= (gradmag + TINYVAL);

                // --- Form relative velocity (node vel minus wall vel) ---
                amrex::Real relvel_in[AMREX_SPACEDIM];
                amrex::Real relvel_out[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; d++)
                {
                    relvel_in[d] =
                        nodal_data_arr(nodeid, VELX_INDEX + d) - wall_vel[d];
                    relvel_out[d] = relvel_in[d];
                }

                // --- Apply BC ---
                // lsetbc: 0=none, 1=no-slip, 2=free-slip, 3=Coulomb friction
                applybc(relvel_in, relvel_out, lset_wall_mu, normaldir, lsetbc);

                // --- Restore wall velocity and write back ---
                for (int d = 0; d < AMREX_SPACEDIM; d++)
                    nodal_data_arr(nodeid, VELX_INDEX + d) =
                        relvel_out[d] + wall_vel[d];
            });
    }
}
#endif

/**
 * @brief Computes nodal Δv = v_new − v_old for PIC/FLIP blending.
 *
 * Requires that backup_current_velocity() has already stored v_old in
 * DELTA_VELX_INDEX. This routine overwrites DELTA_VELX_INDEX with:
 *
 *      Δv[d] = v[d] − v_old[d]
 *
 * Only nodes with positive mass are updated.
 *
 * @param[in,out] nodaldata  Nodal MultiFab containing velocity fields.
 *
 * @return None.
 */

void store_delta_velocity(MultiFab &nodaldata)
{
    // amrex::Print() << "Storing delta velocity\n";
    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        // const Box &bx = mfi.validbox();
        //  Box nodalbox = convert(bx, {AMREX_D_DECL(1, 1, 1)});
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));

        Array4<Real> nodal_data_arr = nodaldata.array(mfi);

        amrex::ParallelFor(
            nodalbox,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                if (nodal_data_arr(i, j, k, MASS_INDEX) > shunya)
                {
                    for (int d = 0; d < AMREX_SPACEDIM; d++)
                    {
                        nodal_data_arr(i, j, k, DELTA_VELX_INDEX + d) =
                            nodal_data_arr(i, j, k, VELX_INDEX + d) -
                            nodal_data_arr(i, j, k, DELTA_VELX_INDEX + d);
                    }
                }
            });
    }
}

#if USE_TEMP
/**
 * @brief Computes nodal ΔT = T_new − T_old for thermal PIC/FLIP updates.
 *
 * Requires that backup_current_temperature() has already stored T_old in
 * DELTA_TEMPERATURE. This routine overwrites DELTA_TEMPERATURE with:
 *
 *      ΔT = T − T_old
 *
 * Only nodes with positive MASS_SPHEAT are updated.
 *
 * @param[in,out] nodaldata  Nodal MultiFab containing thermal fields.
 *
 * @return None.
 */
void store_delta_temperature(MultiFab &nodaldata)
{

    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));

        Array4<Real> nodal_data_arr = nodaldata.array(mfi);

        amrex::ParallelFor(
            nodalbox,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                if (nodal_data_arr(i, j, k, MASS_SPHEAT) > shunya)
                {
                    nodal_data_arr(i, j, k, DELTA_TEMPERATURE) =
                        nodal_data_arr(i, j, k, TEMPERATURE) -
                        nodal_data_arr(i, j, k, DELTA_TEMPERATURE);
                }
            });
    }
}
#endif

/**
 * @brief Advances nodal velocities using nodal forces (explicit time
 * integration).
 *
 * For each node with mass ≥ mass_tolerance:
 *
 *      v[d] ← v[d] + (F[d] / m) * dt
 *
 * Otherwise, nodal velocity is zeroed.
 *
 * @param[in,out] nodaldata      Nodal MultiFab containing mass, force,
 * velocity.
 * @param[in]     dt             Time step.
 * @param[in]     mass_tolerance Minimum mass required to update velocity.
 *
 * @return None.
 */

void Nodal_Time_Update_Momentum(MultiFab &nodaldata,
                                const amrex::Real &dt,
                                const amrex::Real &mass_tolerance)
{
    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        // const Box &bx = mfi.validbox();
        //  Box nodalbox = convert(bx, {AMREX_D_DECL(1, 1, 1)});
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));

        Array4<Real> nodal_data_arr = nodaldata.array(mfi);

        amrex::ParallelFor(
            nodalbox,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                if (nodal_data_arr(i, j, k, MASS_INDEX) >= mass_tolerance)
                {
                    for (int d = 0; d < AMREX_SPACEDIM; d++)
                    {
                        nodal_data_arr(i, j, k, VELX_INDEX + d) +=
                            nodal_data_arr(i, j, k, FRCX_INDEX + d) /
                            nodal_data_arr(i, j, k, MASS_INDEX) * dt;
                    }
                }
                else
                {
                    nodal_data_arr(i, j, k, VELX_INDEX) = 0.0;
                    nodal_data_arr(i, j, k, VELY_INDEX) = 0.0;
                    nodal_data_arr(i, j, k, VELZ_INDEX) = 0.0;
                }
            });
    }
}

#if USE_TEMP
/**
 * @brief Advances nodal temperature using nodal heat sources.
 *
 * For each node with MASS_SPHEAT ≥ mass_tolerance:
 *
 *      T ← T + (source / MASS_SPHEAT) * dt
 *
 * Otherwise, temperature is set to zero.
 *
 * @param[in,out] nodaldata      Nodal MultiFab containing thermal fields.
 * @param[in]     dt             Time step.
 * @param[in]     mass_tolerance Minimum thermal mass required to update T.
 *
 * @return None.
 */
void Nodal_Time_Update_Temperature(MultiFab &nodaldata,
                                   const amrex::Real &dt,
                                   const amrex::Real &mass_tolerance)
{

    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));

        Array4<Real> nodal_data_arr = nodaldata.array(mfi);

        amrex::ParallelFor(
            nodalbox,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                if (nodal_data_arr(i, j, k, MASS_SPHEAT) >= mass_tolerance)
                {
                    nodal_data_arr(i, j, k, TEMPERATURE) +=
                        nodal_data_arr(i, j, k, SOURCE_TEMP_INDEX) /
                        nodal_data_arr(i, j, k, MASS_SPHEAT) * dt;
                }
                else
                {
                    nodal_data_arr(i, j, k, TEMPERATURE) = 0.0;
                }
            });
    }
}
#endif

/**
 * @brief Detects and resolves contact between material nodes and rigid bodies.
 *
 * A node is in contact if:
 *   - MASS_INDEX > contact_tolerance
 *   - MASS_RIGID_INDEX > contact_tolerance
 *   - RIGID_BODY_ID != −1
 *
 * For such nodes:
 *   1. Computes relative velocity between node and rigid body.
 *   2. Projects out the normal component if the node is approaching the rigid
 * body:
 *
 *        v_node ← v_node − (v_rel ⋅ n) n
 *
 * This enforces non‑penetration at the nodal level.
 *
 * @param[in,out] nodaldata  Nodal MultiFab containing velocity and normals.
 * @param[in]     geom       Geometry (unused).
 * @param[in]     contact_tolerance  Minimum mass to consider contact.
 * @param[in]     velocity   Rigid‑body velocities indexed by rigid body ID.
 *
 * @return None.
 */

void nodal_detect_contact(
    MultiFab &nodaldata,
    const Geometry /*geom*/,
    amrex::Real &contact_tolerance,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>,
                    numrigidbodies> velocity)
{
    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        // const Box &bx = mfi.validbox();
#if (AMREX_SPACEDIM == 1)
        // Box nodalbox = convert(bx, {1});
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));
#elif (AMREX_SPACEDIM == 2)
        // Box nodalbox = convert(bx, {1, 1});
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));
#else
        // Box nodalbox = convert(bx, {AMREX_D_DECL(1, 1, 1)});
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));
#endif

        Array4<Real> nodal_data_arr = nodaldata.array(mfi);

        amrex::ParallelFor(
            nodalbox,
            [=] AMREX_GPU_DEVICE(AMREX_D_DECL(int i, int j, int k)) noexcept
            {
                if (nodal_data_arr(IntVect(AMREX_D_DECL(i, j, k)), MASS_INDEX) >
                        contact_tolerance &&
                    nodal_data_arr(IntVect(AMREX_D_DECL(i, j, k)),
                                   MASS_RIGID_INDEX) > contact_tolerance &&
                    int(nodal_data_arr(IntVect(AMREX_D_DECL(i, j, k)),
                                       RIGID_BODY_ID)) != -1)
                {
                    const int rb_id = int(nodal_data_arr(
                        IntVect(AMREX_D_DECL(i, j, k)), RIGID_BODY_ID));

                    // Compute contact_alpha = (v_node - v_rigid) � normal
                    amrex::Real contact_alpha = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        contact_alpha +=
                            (nodal_data_arr(IntVect(AMREX_D_DECL(i, j, k)),
                                            VELX_INDEX + d) -
                             velocity[rb_id][d]) *
                            nodal_data_arr(IntVect(AMREX_D_DECL(i, j, k)),
                                           NORMALX + d);
                    }

                    if (contact_alpha >= 0.0)
                    {
                        // Relative velocity along normal
                        amrex::Real V_relative = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            V_relative +=
                                (nodal_data_arr(IntVect(AMREX_D_DECL(i, j, k)),
                                                VELX_INDEX + d) -
                                 velocity[rb_id][d]) *
                                nodal_data_arr(IntVect(AMREX_D_DECL(i, j, k)),
                                               NORMALX + d);
                        }

                        // Project out normal component
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            nodal_data_arr(IntVect(AMREX_D_DECL(i, j, k)),
                                           VELX_INDEX + d) -=
                                V_relative *
                                nodal_data_arr(IntVect(AMREX_D_DECL(i, j, k)),
                                               NORMALX + d);
                        }
                    }
                }
            });
    }
}

/**
 * @brief Precomputes per‑node shape‑function index categories for boundary
 * handling.
 *
 * For each node, assigns an integer code (0–4) in each dimension indicating:
 *   - 0: domain lower boundary
 *   - 1: near lower boundary
 *   - 2: interior
 *   - 3: near upper boundary
 *   - 4: domain upper boundary
 *
 * These indices are used to select one‑sided or symmetric interpolation
 * stencils.
 *
 * @param[out] shapefunctionindex  iMultiFab storing index codes.
 * @param[in]  geom                Geometry describing the domain.
 *
 * @return None.
 */

void initialise_shape_function_indices(iMultiFab &shapefunctionindex,
                                       const amrex::Geometry geom)
{
    const int *domloarr = geom.Domain().loVect();
    const int *domhiarr = geom.Domain().hiVect();

    /*int periodic[AMREX_SPACEDIM] = {
        geom.isPeriodic(XDIR), geom.isPeriodic(YDIR), geom.isPeriodic(ZDIR)};*/

    GpuArray<int, AMREX_SPACEDIM> lo = {
        AMREX_D_DECL(domloarr[0], domloarr[1], domloarr[2])};
    GpuArray<int, AMREX_SPACEDIM> hi = {
        AMREX_D_DECL(domhiarr[0], domhiarr[1], domhiarr[2])};

    for (MFIter mfi(shapefunctionindex); mfi.isValid(); ++mfi)
    {
        // const Box &bx = mfi.validbox();
        //  Box nodalbox = convert(bx, {AMREX_D_DECL(1, 1, 1)});
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));

        Array4<int> shapefunctionindex_arr = shapefunctionindex.array(mfi);

        amrex::ParallelFor(nodalbox,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           {
                               if (i == lo[0])
                               {
                                   shapefunctionindex_arr(i, j, k, 0) = 0;
                               }
                               else if (i == lo[0] + 1)
                               {
                                   shapefunctionindex_arr(i, j, k, 0) = 1;
                               }
                               else if (i == hi[0])
                               {
                                   shapefunctionindex_arr(i, j, k, 0) = 4;
                               }
                               else if (i == hi[0] - 1)
                               {
                                   shapefunctionindex_arr(i, j, k, 0) = 3;
                               }
                               else
                               {
                                   shapefunctionindex_arr(i, j, k, 0) = 2;
                               }

                               if (j == lo[1])
                               {
                                   shapefunctionindex_arr(i, j, k, 1) = 0;
                               }
                               else if (j == lo[1] + 1)
                               {
                                   shapefunctionindex_arr(i, j, k, 1) = 1;
                               }
                               else if (j == hi[1])
                               {
                                   shapefunctionindex_arr(i, j, k, 1) = 4;
                               }
                               else if (j == hi[1] - 1)
                               {
                                   shapefunctionindex_arr(i, j, k, 1) = 3;
                               }
                               else
                               {
                                   shapefunctionindex_arr(i, j, k, 1) = 2;
                               }

                               if (k == lo[2])
                               {
                                   shapefunctionindex_arr(i, j, k, 2) = 0;
                               }
                               else if (k == lo[2] + 1)
                               {
                                   shapefunctionindex_arr(i, j, k, 2) = 1;
                               }
                               else if (k == hi[2])
                               {
                                   shapefunctionindex_arr(i, j, k, 2) = 4;
                               }
                               else if (k == hi[2] - 1)
                               {
                                   shapefunctionindex_arr(i, j, k, 2) = 3;
                               }
                               else
                               {
                                   shapefunctionindex_arr(i, j, k, 2) = 2;
                               }
                           });
    }
}

/**
 * @brief Applies velocity boundary conditions at nodal locations.
 *
 * For each node lying on a domain boundary:
 *   1. Subtracts wall velocity to obtain relative velocity.
 *   2. Applies applybc() using the appropriate wall friction μ and BC type.
 *   3. Restores wall velocity to obtain the final nodal velocity.
 *
 * Supports:
 *   - No‑slip
 *   - Slip
 *   - Partial slip
 *   - Periodic
 *
 * @param[in]     geom             Geometry describing the domain.
 * @param[in,out] nodaldata        Nodal MultiFab containing velocity fields.
 * @param[in]     bcloarr          BC types at lower faces.
 * @param[in]     bchiarr          BC types at upper faces.
 * @param[in]     wall_mu_loarr    Friction coefficients at lower faces.
 * @param[in]     wall_mu_hiarr    Friction coefficients at upper faces.
 * @param[in]     wall_vel_loarr   Wall velocities at lower faces.
 * @param[in]     wall_vel_hiarr   Wall velocities at upper faces.
 * @param[in]     dt               Time step (unused).
 *
 * @return None.
 */

void nodal_bcs(const amrex::Geometry geom,
               MultiFab &nodaldata,
               int bcloarr[AMREX_SPACEDIM],
               int bchiarr[AMREX_SPACEDIM],
               amrex::Real wall_mu_loarr[AMREX_SPACEDIM],
               amrex::Real wall_mu_hiarr[AMREX_SPACEDIM],
               amrex::Real wall_vel_loarr[AMREX_SPACEDIM * AMREX_SPACEDIM],
               amrex::Real wall_vel_hiarr[AMREX_SPACEDIM * AMREX_SPACEDIM],
               const amrex::Real & /*dt*/)
{
    const int *domloarr = geom.Domain().loVect();
    const int *domhiarr = geom.Domain().hiVect();

    GpuArray<int, AMREX_SPACEDIM> domlo;
    GpuArray<int, AMREX_SPACEDIM> domhi;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        domlo[d] = domloarr[d];
        domhi[d] = domhiarr[d];
    }

    GpuArray<int, AMREX_SPACEDIM> bclo;
    GpuArray<int, AMREX_SPACEDIM> bchi;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        bclo[d] = bcloarr[d];
        bchi[d] = bchiarr[d];
    }

    GpuArray<amrex::Real, AMREX_SPACEDIM> wall_mu_lo;
    GpuArray<amrex::Real, AMREX_SPACEDIM> wall_mu_hi;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        wall_mu_lo[d] = wall_mu_loarr[d];
        wall_mu_hi[d] = wall_mu_hiarr[d];
    }

    GpuArray<amrex::Real, AMREX_SPACEDIM * AMREX_SPACEDIM> wall_vel_lo;
    GpuArray<amrex::Real, AMREX_SPACEDIM * AMREX_SPACEDIM> wall_vel_hi;
    for (int d = 0; d < AMREX_SPACEDIM * AMREX_SPACEDIM; ++d)
    {
        wall_vel_lo[d] = wall_vel_loarr[d];
        wall_vel_hi[d] = wall_vel_hiarr[d];
    }

    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));

        Array4<amrex::Real> nodal_data_arr = nodaldata.array(mfi);

        amrex::ParallelFor(
            nodalbox,
            [=] AMREX_GPU_DEVICE(AMREX_D_DECL(int i, int j, int k)) noexcept
            {
                IntVect nodeid(AMREX_D_DECL(i, j, k));

                amrex::Real relvel_in[AMREX_SPACEDIM],
                    relvel_out[AMREX_SPACEDIM];
                amrex::Real wallvel[AMREX_SPACEDIM] = {0.0};

                // Initialize relvel arrays
                for (int d = 0; d < AMREX_SPACEDIM; d++)
                {
                    relvel_in[d] = nodal_data_arr(nodeid, VELX_INDEX + d);
                    relvel_out[d] = relvel_in[d];
                }

                // Loop over each dimension for boundary conditions
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    if (nodeid[dir] == domlo[dir])
                    {
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            wallvel[d] = wall_vel_lo[dir * AMREX_SPACEDIM + d];
                            relvel_in[d] -= wallvel[d];
                        }
                        amrex::Real normaldir[AMREX_SPACEDIM] = {0.0};
                        normaldir[dir] = 1.0;
                        applybc(relvel_in, relvel_out, wall_mu_lo[dir],
                                normaldir, bclo[dir]);
                    }
                    else if (nodeid[dir] == domhi[dir] + 1)
                    {
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            wallvel[d] = wall_vel_hi[dir * AMREX_SPACEDIM + d];
                            relvel_in[d] -= wallvel[d];
                        }
                        amrex::Real normaldir[AMREX_SPACEDIM] = {0.0};
                        normaldir[dir] = -1.0;
                        applybc(relvel_in, relvel_out, wall_mu_hi[dir],
                                normaldir, bchi[dir]);
                    }
                    nodal_data_arr(nodeid, VELX_INDEX + dir) =
                        relvel_out[dir] + wallvel[dir];
                }

                // Update nodal velocities
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                }
            });
    }
}

#if USE_TEMP
/**
 * @brief Applies Dirichlet temperature boundary conditions at nodal locations.
 *
 * For each dimension d:
 *   - At nodeid[d] = domlo[d], set T = wall_temp_lo[d]
 *   - At nodeid[d] = domhi[d] + 1, set T = wall_temp_hi[d]
 *
 * @param[in]     geom                   Geometry describing the domain.
 * @param[in,out] nodaldata              Nodal MultiFab containing temperature.
 * @param[in]     bcloarr                BC types at lower faces (unused).
 * @param[in]     bchiarr                BC types at upper faces (unused).
 * @param[in]     dirichlet_temperature_lo  Temperature at lower faces.
 * @param[in]     dirichlet_temperature_hi  Temperature at upper faces.
 *
 * @return None.
 */

void nodal_bcs_temperature(const amrex::Geometry geom,
                           MultiFab &nodaldata,
                           int bcloarr[AMREX_SPACEDIM],
                           int bchiarr[AMREX_SPACEDIM],
                           amrex::Real dirichlet_temperature_lo[AMREX_SPACEDIM],
                           amrex::Real dirichlet_temperature_hi[AMREX_SPACEDIM])
{

    const int *domloarr = geom.Domain().loVect();
    const int *domhiarr = geom.Domain().hiVect();

    GpuArray<int, AMREX_SPACEDIM> domlo;
    GpuArray<int, AMREX_SPACEDIM> domhi;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        domlo[d] = domloarr[d];
        domhi[d] = domhiarr[d];
    }

    GpuArray<int, AMREX_SPACEDIM> bclo;
    GpuArray<int, AMREX_SPACEDIM> bchi;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        bclo[d] = bcloarr[d];
        bchi[d] = bchiarr[d];
    }

    GpuArray<amrex::Real, AMREX_SPACEDIM> wall_temp_lo;
    GpuArray<amrex::Real, AMREX_SPACEDIM> wall_temp_hi;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        wall_temp_lo[d] = dirichlet_temperature_lo[d];
        wall_temp_hi[d] = dirichlet_temperature_hi[d];
    }

    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));

        Array4<amrex::Real> nodal_data_arr = nodaldata.array(mfi);

        amrex::ParallelFor(nodalbox,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           {
                               (void)j;
                               (void)k;
                               IntVect nodeid(AMREX_D_DECL(i, j, k));

                               // Loop over each dimension
                               for (int d = 0; d < AMREX_SPACEDIM; ++d)
                               {
                                   // At lower boundary in dimension d
                                   if (nodeid[d] == domlo[d])
                                   {
                                       nodal_data_arr(nodeid, TEMPERATURE) =
                                           wall_temp_lo[d];
                                   }
                                   // At upper boundary in dimension d
                                   else if (nodeid[d] == domhi[d] + 1)
                                   {
                                       nodal_data_arr(nodeid, TEMPERATURE) =
                                           wall_temp_hi[d];
                                   }
                               }
                           });
    }
}
#endif

/**
 * @brief Computes interpolation error for a manufactured cosine velocity field.
 *
 * Overwrites nodaldata(nodaldataindex) with:
 *
 *      error = v_x − cos(2π x)
 *
 * where x is computed from domain lower bound and grid spacing.
 *
 * @param[in]     geom            Geometry describing the domain.
 * @param[in,out] nodaldata       Nodal MultiFab containing velocity/error.
 * @param[in]     nodaldataindex  Component index to store the error.
 *
 * @return None.
 */

void CalculateInterpolationError(const amrex::Geometry geom,
                                 amrex::MultiFab &nodaldata,
                                 int nodaldataindex)
{
    const int *domloarr = geom.Domain().loVect();

    const auto dx = geom.CellSizeArray();
    const auto Pi = 4.0 * atan(1.0);

    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));

        Array4<Real> nodal_data_arr = nodaldata.array(mfi);

        amrex::ParallelFor(nodalbox,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           {
                               nodal_data_arr(i, j, k, nodaldataindex) =
                                   nodal_data_arr(i, j, k, VELX_INDEX) -
                                   cos(2.0 * Pi * (domloarr[0] + i * dx[0]) /
                                       1.0);
                           });
    }
}

/**
 * @brief Sets all nodal data components to zero (shunya).
 *
 * Resets the MultiFab including ghost cells.
 *
 * @param[in,out] nodaldata            Nodal MultiFab to reset.
 * @param[in]     ng_cells_nodaldata   Number of ghost cells to include.
 *
 * @return None.
 */

void Reset_Nodaldata_to_Zero(amrex::MultiFab &nodaldata, int ng_cells_nodaldata)
{
    if (testing == 1)
        amrex::Print() << "\n Reseting Nodal Data to shunya \n";
    nodaldata.setVal(shunya, ng_cells_nodaldata);
}
