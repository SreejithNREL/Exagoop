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
 * @brief Applies level-set-based velocity boundary conditions to nodal data.
 *
 * For each grid node where phi < 0 (inside the obstacle) and nodal mass > 0:
 *   1. Evaluates phi at the physical node location via multilinear
 * interpolation on the coarsened lsphi (average_down_nodal from refined grid).
 *   2. Computes the level-set gradient at that node to obtain the wall normal.
 *   3. Guards against degenerate (zero-gradient) nodes — Bug 3 fix.
 *   4. Applies applybc() to enforce the requested BC type (no-slip, free-slip,
 *      Coulomb friction) relative to the wall velocity.
 *
 * Bug fixes applied (relative to 235793b):
 *
 *   Bug 1 — average_down_nodal + FillBoundary:
 *     lsphi is stored at ls_refinement * coarse resolution.  Reading it
 *     directly with a coarse MFIter (Bug 4) or without FillBoundary (Bug 1)
 *     produces garbage in ghost cells => phantom obstacles at tile boundaries.
 *     Fix: average_down_nodal onto a coarse nodal MultiFab, then FillBoundary,
 *     then pass lsref=1 to get_levelset_value / get_levelset_grad.
 *
 *   Bug 3 — degenerate gradient guard:
 *     Nodes deep inside the obstacle have zero gradient.  Dividing by
 *     (gradmag + TINYVAL) with TINYVAL~1e-20 produces ~1e20 normals =>
 *     velocity blowup => NaN particles => locateParticle crash.
 *     Fix: if gradmag < 1e-10, skip the node entirely.
 *
 *   Bug 4 — coarse MFIter with refined lsphi:
 *     mpm_ebtools::lsphi is refined; indexing it with a coarse MFIter tile
 *     gives the wrong array region.
 *     Fix: use lsphi_coarse.array(mfi) after average_down_nodal.
 *
 * @param[in,out] nodaldata     Nodal MultiFab containing velocity fields.
 * @param[in]     geom          Coarse-level geometry.
 * @param[in]     dt            Time step (reserved for future moving-wall use).
 * @param[in]     lsetbc        BC type: 0=none, 1=no-slip, 2=free-slip,
 *                              3=Coulomb friction.
 * @param[in]     lset_wall_mu  Friction coefficient (used when lsetbc==3).
 * @param[in]     wall_vel      Wall velocity vector (default: zero = static).
 *
 * @return None.
 */
void nodal_levelset_bcs(MultiFab &nodaldata,
                        const Geometry geom,
                        amrex::Real & /*dt*/,
                        int lsetbc,
                        amrex::Real lset_wall_mu,
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> wall_vel)
{
    int lsref = mpm_ebtools::ls_refinement;

    // ------------------------------------------------------------------
    // Bug 1 + Bug 4 fix:
    // average_down_nodal coarsens lsphi from the refined grid onto a new
    // nodal MultiFab that matches nodaldata's BoxArray exactly.
    // nghost >= 1 is required: get_levelset_grad reads the ghost layer.
    // FillBoundary must follow to populate those ghost cells — without it
    // any tile boundary where a ghost cell is uninitialised becomes a
    // phantom obstacle (Bug 1).
    // ------------------------------------------------------------------
    MultiFab lsphi_coarse(nodaldata.boxArray(), nodaldata.DistributionMap(),
                          1,  // ncomp
                          1); // nghost — MUST be >= 1
    amrex::average_down_nodal(*mpm_ebtools::lsphi, lsphi_coarse,
                              amrex::IntVect(lsref));
    // Use coarse geometry periodicity — lsphi_coarse lives on the coarse grid
    lsphi_coarse.FillBoundary(geom.periodicity());

    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();

    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));

        Array4<Real> nodal_data_arr = nodaldata.array(mfi);
        // Bug 4 fix: use lsphi_coarse, NOT mpm_ebtools::lsphi->array(mfi)
        Array4<Real> lsarr = lsphi_coarse.array(mfi);

        amrex::ParallelFor(
            nodalbox,
            [=] AMREX_GPU_DEVICE(AMREX_D_DECL(int i, int j, int k)) noexcept
            {
                IntVect nodeid(AMREX_D_DECL(i, j, k));

                // Physical coordinates of this node
                amrex::Real xp[AMREX_SPACEDIM] = {AMREX_D_DECL(
                    plo[XDIR] + i * dx[XDIR], plo[YDIR] + j * dx[YDIR],
                    plo[ZDIR] + k * dx[ZDIR])};

                // lsphi_coarse is already at coarse resolution, so lsref=1
                amrex::Real lsval =
                    get_levelset_value(lsarr, plo, dx, xp, /*lsref=*/1);

                // Only act on nodes inside the obstacle with nonzero mass
                if (lsval >= 0.0 ||
                    nodal_data_arr(nodeid, MASS_INDEX) <= shunya)
                    return;

                // Compute wall-normal from level-set gradient
                amrex::Real normaldir[AMREX_SPACEDIM] = {
                    AMREX_D_DECL(1.0, 0.0, 0.0)};
                get_levelset_grad(lsarr, plo, dx, xp, /*lsref=*/1, normaldir);

                amrex::Real gradmag = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; d++)
                    gradmag += normaldir[d] * normaldir[d];
                gradmag = std::sqrt(gradmag);

                // Bug 3 fix: skip degenerate nodes (zero gradient deep inside
                // obstacle). Dividing by TINYVAL~1e-20 gives ~1e20 normals =>
                // velocity blowup => NaN particles.
                if (gradmag < 1.0e-10)
                    return;

                for (int d = 0; d < AMREX_SPACEDIM; d++)
                    normaldir[d] /= gradmag;

                // Velocity relative to wall
                amrex::Real relvel_in[AMREX_SPACEDIM];
                amrex::Real relvel_out[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; d++)
                {
                    relvel_in[d] =
                        nodal_data_arr(nodeid, VELX_INDEX + d) - wall_vel[d];
                    relvel_out[d] = relvel_in[d]; // default: no change
                }

                // Normal component of velocity relative to wall.
                // The level-set gradient points OUTWARD (away from obstacle),
                // so veln > 0 means the node is moving away — no BC needed.
                // Only apply BC when veln <= 0 (node moving into the wall).
                amrex::Real veln = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; d++)
                    veln += relvel_in[d] * normaldir[d];

                if (veln <= 0.0)
                {
                    // Apply BC (no-slip / free-slip / Coulomb friction)
                    applybc(relvel_in, relvel_out, lset_wall_mu, normaldir,
                            lsetbc);

                    // Write back: add wall velocity to restore absolute frame
                    for (int d = 0; d < AMREX_SPACEDIM; d++)
                        nodal_data_arr(nodeid, VELX_INDEX + d) =
                            relvel_out[d] + wall_vel[d];
                }
                // veln > 0: node already moving away from wall — leave velocity
                // unchanged
            });
    }
}
#endif

#if USE_TEMP
/**
 * @brief Applies thermal BCs on nodes inside the level-set obstacle (phi < 0).
 *
 * Supports BC types:
 *   1  DIRICHLET  — T = T_wall (direct override)
 *   3  HEAT FLUX  — ghost-point normal: T_node = T_surf + (q/k) * d
 *                   where d = |lsval| = distance from node to EB surface
 *   4  CONVECTIVE — ghost-point Robin: T_node = (T_surf + Bi*T_inf)/(1+Bi)
 *                   where Bi = h*d/k, d = |lsval|
 *
 * Ghost-point approach for types 3 and 4:
 *   The level-set value at a node inside the obstacle is negative.
 *   Its magnitude |lsval| approximates the normal distance to the
 *   EB surface. The ghost-point extrapolates the surface temperature
 *   from the node temperature using this distance:
 *
 *   Type 3: -k * dT/dn|_surf = q
 *     → (T_surf - T_node) / d = -q/k   [normal points outward]
 *     → T_node = T_surf + (q/k) * d
 *     We set T_node to enforce this — particles inside pick up the
 *     correct gradient toward the surface.
 *
 *   Type 4: -k * dT/dn|_surf = h*(T_surf - T_inf)
 *     → T_surf = T_node / (1 + h*d/k) + Bi*T_inf/(1+Bi)  [T_inf=0 simplifies]
 *     → T_node = (T_surf_old + Bi*T_inf) / (1 + Bi)  [iterative, one pass]
 *     where Bi = h*d/k
 *
 * Only acts on nodes with MASS_SPHEAT > 0 (particles present) and lsval < 0.
 *
 * Uses the same average_down_nodal + FillBoundary pattern as nodal_levelset_bcs
 * (Bug 1 + Bug 4 fix).
 */
void nodal_levelset_bcs_temperature(MultiFab &nodaldata,
                                    const Geometry geom,
                                    amrex::Real /*dt*/,
                                    int bc_type,
                                    amrex::Real T_wall,
                                    amrex::Real flux,
                                    amrex::Real h_conv,
                                    amrex::Real T_inf,
                                    bool /*pre_update*/)
{
#if USE_EB
    if (bc_type == 0 || bc_type == 2)
        return;

    int lsref = mpm_ebtools::ls_refinement;

    // Bug 1 + Bug 4 fix: coarsen lsphi to nodaldata BoxArray with nghost>=1
    MultiFab lsphi_coarse(nodaldata.boxArray(), nodaldata.DistributionMap(),
                          1,  // ncomp
                          1); // nghost — must be >= 1 for get_levelset_grad
    amrex::average_down_nodal(*mpm_ebtools::lsphi, lsphi_coarse,
                              amrex::IntVect(lsref));
    lsphi_coarse.FillBoundary(geom.periodicity());

    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();

    // Capture by value for GPU kernel
    amrex::Real T_wall_ = T_wall;
    amrex::Real flux_ = flux;
    amrex::Real h_conv_ = h_conv;
    amrex::Real T_inf_ = T_inf;
    int bc_type_ = bc_type;

    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));
        Array4<Real> arr = nodaldata.array(mfi);
        Array4<Real> lsarr = lsphi_coarse.array(mfi);

        amrex::ParallelFor(
            nodalbox,
            [=] AMREX_GPU_DEVICE(AMREX_D_DECL(int i, int j, int k)) noexcept
            {
                IntVect nodeid(AMREX_D_DECL(i, j, k));

                amrex::Real xp[AMREX_SPACEDIM] = {AMREX_D_DECL(
                    plo[XDIR] + i * dx[XDIR], plo[YDIR] + j * dx[YDIR],
                    plo[ZDIR] + k * dx[ZDIR])};

                // lsphi_coarse already at coarse resolution → lsref=1
                amrex::Real lsval =
                    get_levelset_value(lsarr, plo, dx, xp, /*lsref=*/1);

                // Only act on nodes inside obstacle with particles
                if (lsval >= 0.0 || arr(nodeid, MASS_SPHEAT) <= shunya)
                    return;

                // Distance from this node to the EB surface = |lsval|
                // (lsval < 0 inside obstacle, magnitude = normal distance)
                amrex::Real d = -lsval; // positive distance to surface

                if (bc_type_ == 1)
                {
                    // Dirichlet: direct override
                    arr(nodeid, TEMPERATURE) = T_wall_;
                }
                else if (bc_type_ == 3)
                {
                    // Prescribed flux — ghost-point Neumann:
                    // -k * dT/dn|_surf = q  (n = outward normal from obstacle)
                    // Node is inside obstacle at distance d from surface.
                    // T_node = T_surf + (q/k) * d
                    // We don't know T_surf directly, but we know the gradient
                    // must be q/k. Set T to reflect this from the current
                    // value: Use current T as T_surf estimate → T_node = T +
                    // (q/k)*d
                    amrex::Real T_curr = arr(nodeid, TEMPERATURE);
                    arr(nodeid, TEMPERATURE) = T_curr + (flux_ / 1.0) * d;
                    // Note: k=1 assumed; for variable k, use
                    // arr(nodeid, THERMAL_CONDUCTIVITY) if stored nodally
                }
                else if (bc_type_ == 4)
                {
                    // Convective — ghost-point Robin:
                    // -k * dT/dn|_surf = h*(T_surf - T_inf)
                    // Bi = h*d/k
                    // T_node = (T_curr + Bi*T_inf) / (1 + Bi)
                    amrex::Real Bi = h_conv_ * d; // k=1 assumed
                    amrex::Real T_curr = arr(nodeid, TEMPERATURE);
                    arr(nodeid, TEMPERATURE) =
                        (T_curr + Bi * T_inf_) / (1.0 + Bi);
                }
            });
    }
#endif
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
                           int bclo_type[AMREX_SPACEDIM],
                           int bchi_type[AMREX_SPACEDIM],
                           amrex::Real bclo_val[AMREX_SPACEDIM],
                           amrex::Real bchi_val[AMREX_SPACEDIM],
                           amrex::Real bclo_Tinf[AMREX_SPACEDIM],
                           amrex::Real bchi_Tinf[AMREX_SPACEDIM],
                           bool /*pre_update*/)
{
    const int *domloarr = geom.Domain().loVect();
    const int *domhiarr = geom.Domain().hiVect();
    const auto dx = geom.CellSizeArray();

    GpuArray<int, AMREX_SPACEDIM> domlo, domhi;
    GpuArray<int, AMREX_SPACEDIM> bc_lo_type, bc_hi_type;
    GpuArray<amrex::Real, AMREX_SPACEDIM> lo_val, hi_val, lo_Tinf, hi_Tinf;

    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        domlo[d] = domloarr[d];
        domhi[d] = domhiarr[d];
        bc_lo_type[d] = bclo_type[d];
        bc_hi_type[d] = bchi_type[d];
        lo_val[d] = bclo_val[d];
        hi_val[d] = bchi_val[d];
        lo_Tinf[d] = bclo_Tinf[d];
        hi_Tinf[d] = bchi_Tinf[d];
    }

    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        Box nodalbox = convert(mfi.tilebox(), IntVect(AMREX_D_DECL(1, 1, 1)));
        Array4<amrex::Real> arr = nodaldata.array(mfi);

        amrex::ParallelFor(nodalbox,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           {
                               IntVect nodeid(AMREX_D_DECL(i, j, k));

                               // Track whether a strong BC has already set
                               // TEMPERATURE on this node. Prevents weaker BCs
                               // (adiabatic/none) from overwriting at corners.
                               bool bc_applied = false;

                               for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                               {
                                   int bc_type = -1;
                                   amrex::Real val = 0.0;
                                   amrex::Real Tinf = 0.0;
                                   int sign = 0;

                                   if (nodeid[dir] == domlo[dir])
                                   {
                                       bc_type = bc_lo_type[dir];
                                       val = lo_val[dir];
                                       Tinf = lo_Tinf[dir];
                                       sign = +1;
                                   }
                                   else if (nodeid[dir] == domhi[dir] + 1)
                                   {
                                       bc_type = bc_hi_type[dir];
                                       val = hi_val[dir];
                                       Tinf = hi_Tinf[dir];
                                       sign = -1;
                                   }

                                   if (bc_type == 1)
                                   {
                                       // Dirichlet: hard override
                                       arr(nodeid, TEMPERATURE) = val;
                                       bc_applied = true;
                                   }
                                   else if (sign != 0 && bc_type != -1)
                                   {
                                       IntVect nb = nodeid;
                                       nb[dir] += sign;
                                       amrex::Real T_int = arr(nb, TEMPERATURE);

                                       if (bc_type == 2 || bc_type == 0)
                                       {
                                           // Adiabatic/none: T_b = T_interior
                                           if (!bc_applied)
                                           {
                                               arr(nodeid, TEMPERATURE) = T_int;
                                               bc_applied = true;
                                           }
                                       }
                                       else if (bc_type == 3)
                                       {
                                           // Prescribed flux: T_b = T_int + (q/k)*dx
                                           arr(nodeid, TEMPERATURE) =
                                               T_int + val * dx[dir];
                                           bc_applied = true;
                                       }
                                       else if (bc_type == 4)
                                       {
                                           // Convective: T_b = (T_int + Bi*T_inf)/(1+Bi)
                                           amrex::Real Bi = val * dx[dir];
                                           arr(nodeid, TEMPERATURE) =
                                               (T_int + Bi * Tinf) / (1.0 + Bi);
                                           bc_applied = true;
                                       }
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
