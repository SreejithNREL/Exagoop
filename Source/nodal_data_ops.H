#ifndef NODAL_DATA_OPS_H_
#define NODAL_DATA_OPS_H_
// clang-format off
#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <constants.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <string>
#include <vector>
// clang-format on

class MPMspecs;

void write_grid_file(std::string fname,
                     amrex::MultiFab &nodaldata,
                     amrex::Vector<std::string> fieldnames,
                     amrex::Geometry geom,
                     amrex::BoxArray ba,
                     amrex::DistributionMapping dm,
                     amrex::Real time);

void backup_current_velocity(amrex::MultiFab &nodaldata);
void backup_current_temperature(amrex::MultiFab &nodaldata);
void store_delta_velocity(amrex::MultiFab &nodaldata);
void store_delta_temperature(amrex::MultiFab &nodaldata);
void Nodal_Time_Update_Momentum(amrex::MultiFab &nodaldata,
                                const amrex::Real &dt,
                                const amrex::Real &mass_tolerance);
void Nodal_Time_Update_Temperature(amrex::MultiFab &nodaldata,
                                   const amrex::Real &dt,
                                   const amrex::Real &mass_tolerance);
void nodal_detect_contact(
    amrex::MultiFab &nodaldata,
    const amrex::Geometry geom,
    amrex::Real &contact_tolerance,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>,
                    numrigidbodies>);
void initialise_shape_function_indices(amrex::iMultiFab &shapefunctionindex,
                                       const amrex::Geometry geom);

void nodal_bcs(const amrex::Geometry geom,
               amrex::MultiFab &nodaldata,
               int bclo[AMREX_SPACEDIM],
               int bchi[AMREX_SPACEDIM],
               amrex::Real wall_mu_lo[AMREX_SPACEDIM],
               amrex::Real wall_mu_hi[AMREX_SPACEDIM],
               amrex::Real wall_vel_lo[AMREX_SPACEDIM * AMREX_SPACEDIM],
               amrex::Real wall_vel_hi[AMREX_SPACEDIM * AMREX_SPACEDIM],
               const amrex::Real &dt);
void nodal_bcs(const amrex::Geometry geom,
               amrex::MultiFab &nodaldata,
               const amrex::Real &dt);
void nodal_bcs_temperature(
    const amrex::Geometry geom,
    amrex::MultiFab &nodaldata,
    int bcloarr[AMREX_SPACEDIM],
    int bchiarr[AMREX_SPACEDIM],
    amrex::Real dirichlet_temperature_lo[AMREX_SPACEDIM],
    amrex::Real dirichlet_temperature_hi[AMREX_SPACEDIM]);

#if USE_TEMP
/**
 * @brief Applies extended nodal temperature boundary conditions.
 *
 * Handles all five BC types per face independently:
 *   0 = no BC (pass-through)
 *   1 = Dirichlet      — T = T_wall
 *   2 = Adiabatic      — zero flux (no-op)
 *   3 = Heat flux      — Q += q * A_node
 *   4 = Convective     — Q += h * (T_inf - T_node) * A_node
 *   5 = Convective UDF — h(x,y,z) and T_inf(x,y,z) from shared library
 *
 * Types 3, 4, 5 add to SOURCE_TEMP_INDEX (before Nodal_Time_Update_Temperature).
 * Type 1 directly sets TEMPERATURE (after Nodal_Time_Update_Temperature).
 * Types 3/4/5 should be called BEFORE Nodal_Time_Update_Temperature.
 * Type 1 should be called AFTER  Nodal_Time_Update_Temperature.
 *
 * @param geom          AMReX geometry
 * @param nodaldata     Nodal MultiFab (TEMPERATURE, SOURCE_TEMP_INDEX, MASS_SPHEAT)
 * @param bclo          BC type at lower face per dimension
 * @param bchi          BC type at upper face per dimension
 * @param val_lo        Scalar value per lower face (T_wall / q / h)
 * @param val_hi        Scalar value per upper face
 * @param Tinf_lo       Ambient temperature per lower face (types 4, 5)
 * @param Tinf_hi       Ambient temperature per upper face
 * @param udf_so_lo     Path to .so for lower faces with type=5 (empty = unused)
 * @param udf_so_hi     Path to .so for upper faces with type=5
 * @param dt            Time step (used for flux accumulation)
 */
void nodal_bcs_temperature_extended(
    const amrex::Geometry &geom,
    amrex::MultiFab &nodaldata,
    int bclo[AMREX_SPACEDIM],
    int bchi[AMREX_SPACEDIM],
    amrex::Real val_lo[AMREX_SPACEDIM],
    amrex::Real val_hi[AMREX_SPACEDIM],
    amrex::Real Tinf_lo[AMREX_SPACEDIM],
    amrex::Real Tinf_hi[AMREX_SPACEDIM],
    const std::vector<std::string> &udf_so_lo,
    const std::vector<std::string> &udf_so_hi,
    amrex::Real dt);
#endif
void CalculateSurfaceIntegralOnBG(const amrex::Geometry geom,
                                  amrex::MultiFab &nodaldata,
                                  int nodaldataindex,
                                  amrex::Real &integral_value);
void CalculateInterpolationError(const amrex::Geometry geom,
                                 amrex::MultiFab &nodaldata,
                                 int nodaldataindex);

void nodal_levelset_bcs(amrex::MultiFab &nodaldata,
                        const amrex::Geometry geom,
                        amrex::Real &dt,
                        int lsetbc,
                        amrex::Real lset_wall_mu);

// ── Multi-body overload: explicit lsphi per body ──────────────────────────────
#if USE_EB
void nodal_levelset_bcs(amrex::MultiFab &nodaldata,
                        const amrex::Geometry geom,
                        amrex::Real &dt,
                        int lsetbc,
                        amrex::Real lset_wall_mu,
                        amrex::MultiFab *body_lsphi);

class RigidBodyManager;   // forward declaration

void nodal_levelset_bcs_all_bodies(amrex::MultiFab &nodaldata,
                                   const amrex::Geometry geom,
                                   amrex::Real dt,
                                   const RigidBodyManager &rb_manager);

#if USE_TEMP
void nodal_levelset_bcs_temperature_all_bodies(
    amrex::MultiFab &nodaldata,
    const amrex::Geometry geom,
    amrex::Real dt,
    const RigidBodyManager &rb_manager,
    bool pre_update);
#endif
#endif

#if USE_EB && USE_TEMP
/**
 * @brief Applies thermal BCs on the level-set embedded boundary.
 *
 * BC types (one per rigid body / level-set surface):
 *   0 = no BC       1 = Dirichlet (T = val)
 *   2 = Adiabatic   3 = Heat flux (Q = q*A)
 *   4 = Convective  5 = Convective UDF
 *
 * Call BEFORE Nodal_Time_Update_Temperature with pre_update=true  (types 3-5).
 * Call AFTER  Nodal_Time_Update_Temperature with pre_update=false (type 1).
 *
 * @param nodaldata       Nodal MultiFab
 * @param geom            AMReX geometry
 * @param dt              Time step
 * @param lset_temp_bc    BC type (0-5)
 * @param lset_temp_val   h, q, or T_wall depending on BC type
 * @param lset_Tinf       Ambient temperature [K] (types 4, 5)
 * @param lset_temp_udf   Path to .so for UDF BC (type 5)
 * @param pre_update      true = pre-update pass, false = post-update pass
 * @param body_id         Rigid body index (default 0)
 */
void nodal_levelset_bcs_temperature(
    amrex::MultiFab   &nodaldata,
    const amrex::Geometry geom,
    amrex::Real        dt,
    int                lset_temp_bc,
    amrex::Real        lset_temp_val,
    amrex::Real        lset_Tinf,
    const std::string &lset_temp_udf,
    bool               pre_update,
    int                body_id = 0);
#endif

void Create_Output_Directories(MPMspecs &specs);
void Reset_Nodaldata_to_Zero(amrex::MultiFab &nodaldata,
                             int ng_cells_nodaldata);

#endif
