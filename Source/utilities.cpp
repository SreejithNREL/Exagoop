#include <AMReX.H> // for amrex::Print and amrex::Real
// #include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <aesthetics.H>
#include <iomanip>  // for std::setprecision
#include <iostream> // optional, if you use std::cout
#include <mpm_check_pair.H>
#include <mpm_eb.H>
#include <nodal_data_ops.H>
#include <sstream> // optional, if you later use string streams
#include <string>  // for std::string

/**
 * @brief Writes all particle, grid, and level‑set outputs for the current step.
 *
 * This routine performs the full output pipeline:
 *
 *   1. Redistributes particles and updates neighbor lists.
 *   2. Writes particle plotfile via writeParticles().
 *   3. Writes nodal grid plotfile via write_grid_file().
 *   4. Writes level‑set plotfile (if enabled).
 *   5. Writes checkpoint file (if rewrite_checkpoint = true).
 *   6. Writes ASCII particle output (if enabled).
 *
 * Output filenames are constructed using the user‑specified prefixes and
 * number‑of‑digits formatting.
 *
 * @param[in]     specs               Simulation specification structure.
 * @param[in,out] mpm_pc              Particle container.
 * @param[in]     nodaldata           Nodal MultiFab for grid output.
 * @param[in]     levset_data         Level‑set MultiFab (if enabled).
 * @param[in]     nodaldata_names     Names of nodal data components.
 * @param[in]     geom                Geometry for grid output.
 * @param[in]     geom_levset         Geometry for level‑set output.
 * @param[in]     ba                  BoxArray for grid output.
 * @param[in]     dm                  DistributionMapping for grid output.
 * @param[in]     time                Current simulation time.
 * @param[in]     steps               Current step index.
 * @param[in]     output_it           Output iteration counter.
 * @param[in]     rewrite_checkpoint  Whether to write a checkpoint file.
 *
 * @return None.
 */

void Write_Particle_Grid_Levset_Output(
    MPMspecs &specs,
    MPMParticleContainer &mpm_pc,
    MultiFab &nodaldata,
    MultiFab &levset_data,
    amrex::Vector<std::string> &nodaldata_names,
    Geometry &geom,
    Geometry &geom_levset,
    BoxArray &ba,
    DistributionMapping &dm,
    Real time,
    int steps,
    int output_it,
    bool rewrite_checkpoint)
{
    mpm_pc.Redistribute();
    mpm_pc.fillNeighbors();
    BL_PROFILE_VAR("OUTPUT_TIME", outputs);
    Print() << "\nWriting outputs at step, time:" << steps << ", " << time
            << "\n";

    std::string msg;
    std::string pltfile = amrex::Concatenate(
        specs.prefix_gridfilename, output_it, specs.num_of_digits_in_filenames);

    mpm_pc.writeParticles(specs.particle_output_folder +
                              specs.prefix_particlefilename,
                          specs.num_of_digits_in_filenames, steps);

    pltfile = amrex::Concatenate(specs.prefix_gridfilename, steps,
                                 specs.num_of_digits_in_filenames);
    write_grid_file(specs.grid_output_folder + pltfile, nodaldata,
                    nodaldata_names, geom, ba, dm, time);

    if (specs.levset_output)
    {
        pltfile = amrex::Concatenate(specs.levset_output_folder +
                                         specs.prefix_densityfilename,
                                     steps, specs.num_of_digits_in_filenames);
        WriteSingleLevelPlotfile(pltfile, levset_data, {"levset"}, geom_levset,
                                 time, 0);
    }

    if (rewrite_checkpoint)
    {
        mpm_pc.writeCheckpointFile(
            specs.checkpoint_output_folder + specs.prefix_checkpointfilename,
            specs.num_of_digits_in_filenames, time, steps, output_it);
    }

    if (specs.write_ascii)
    {
        mpm_pc.writeAsciiFiles(specs.ascii_output_folder + "/" +
                                   specs.prefix_asciifilename,
                               6, time);
    }

    BL_PROFILE_VAR_STOP(outputs);
}

/**
 * @brief Performs particle‑to‑grid (P2G) transfer of mass, momentum, and
 * forces.
 *
 * A thin wrapper around deposit_onto_grid_momentum(), forwarding all relevant
 * simulation parameters from specs.
 *
 * @param[in] specs         Simulation specification structure.
 * @param[in,out] mpm_pc    Particle container.
 * @param[in,out] nodaldata Nodal MultiFab to receive P2G contributions.
 * @param[in] update_mass   Whether to deposit mass.
 * @param[in] update_vel    Whether to deposit momentum.
 * @param[in] update_forces Whether to deposit internal/external forces.
 *
 * @return None.
 */

void P2G_Momentum(MPMspecs &specs,
                  MPMParticleContainer &mpm_pc,
                  amrex::MultiFab &nodaldata,
                  int update_mass,
                  int update_vel,
                  int update_forces)
{
    if (testing == 1)
        amrex::Print() << "\n Doing P2G \n";
    mpm_pc.deposit_onto_grid_momentum(
        nodaldata, specs.gravity, specs.external_loads_present,
        specs.force_slab_lo, specs.force_slab_hi, specs.extforce, update_mass,
        update_vel, update_forces, specs.mass_tolerance,
        specs.order_scheme_directional, specs.periodic);
}

#if USE_TEMP
/**
 * @brief Performs particle‑to‑grid (P2G) transfer of thermal quantities.
 *
 * Wraps deposit_onto_grid_temperature(), forwarding all thermal parameters
 * from specs.
 *
 * @param[in] specs                   Simulation specification structure.
 * @param[in,out] mpm_pc              Particle container.
 * @param[in,out] nodaldata           Nodal thermal MultiFab.
 * @param[in] reset_nodaldata_to_zero Whether to zero nodal fields first.
 * @param[in] update_temp             Whether to deposit temperature.
 * @param[in] update_source           Whether to deposit heat sources.
 *
 * @return None.
 */

void P2G_Temperature(MPMspecs &specs,
                     MPMParticleContainer &mpm_pc,
                     amrex::MultiFab &nodaldata,
                     int reset_nodaldata_to_zero,
                     int update_temp,
                     int update_source)
{
    if (testing == 1)
        amrex::Print() << "\n Doing P2G for temperature\n";
    mpm_pc.deposit_onto_grid_temperature(
        nodaldata, reset_nodaldata_to_zero, update_temp, update_source,
        specs.mass_tolerance, specs.order_scheme_directional, specs.periodic);
}
#endif

/**
 * @brief Applies velocity boundary conditions at nodal locations.
 *
 * This routine:
 *   1. Applies domain boundary conditions via nodal_bcs().
 *   2. Applies embedded‑boundary (level‑set) BCs if enabled.
 *   3. Computes Δv for PIC/FLIP blending via store_delta_velocity().
 *
 * @param[in]     geom       Geometry describing the domain.
 * @param[in,out] nodaldata  Nodal MultiFab containing velocity fields.
 * @param[in]     specs      Simulation specification structure.
 * @param[in]     dt         Time step.
 *
 * @return None.
 */

void Apply_Nodal_BCs(amrex::Geometry &geom,
                     amrex::MultiFab &nodaldata,
                     MPMspecs &specs,
                     amrex::Real dt)
{

    nodal_bcs(geom, nodaldata, specs.bclo.data(), specs.bchi.data(),
              specs.wall_mu_lo.data(), specs.wall_mu_hi.data(),
              specs.wall_vel_lo.data(), specs.wall_vel_hi.data(), dt);

#if USE_EB
    if (mpm_ebtools::using_levelset_geometry)
    {
        nodal_levelset_bcs(nodaldata, geom, dt, specs.levelset_bc,
                           specs.levelset_wall_mu);
    }
#endif

    // Calculate velocity diff
    store_delta_velocity(nodaldata);
}

#if USE_TEMP
/**
 * @brief Applies temperature boundary conditions at nodal locations.
 *
 * Applies Dirichlet temperature BCs using nodal_bcs_temperature(), then
 * computes ΔT for PIC/FLIP thermal updates via store_delta_temperature().
 *
 * @param[in]     geom       Geometry describing the domain.
 * @param[in,out] nodaldata  Nodal MultiFab containing temperature fields.
 * @param[in]     specs      Simulation specification structure.
 * @param[in]     dt         Time step.
 *
 * @return None.
 */

void Apply_Nodal_BCs_Temperature(amrex::Geometry &geom,
                                 amrex::MultiFab &nodaldata,
                                 MPMspecs &specs,
                                 amrex::Real dt)
{
    amrex::Array<amrex::Real, AMREX_SPACEDIM> temp_lo;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> temp_hi;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        temp_lo[d] = specs.bclo_tempval[d];
        temp_hi[d] = specs.bchi_tempval[d];
    }
    nodal_bcs_temperature(geom, nodaldata, specs.bclo.data(), specs.bchi.data(),
                          temp_lo.data(), temp_hi.data());
    store_delta_temperature(nodaldata);
}
#endif

/**
 * @brief Performs grid‑to‑particle (G2P) transfer of velocity and strain‑rate.
 *
 * Wraps interpolate_from_grid(), forwarding interpolation order, periodicity,
 * PIC/FLIP blending parameter, and time step.
 *
 * @param[in] specs             Simulation specification structure.
 * @param[in,out] mpm_pc        Particle container.
 * @param[in] nodaldata         Nodal MultiFab containing grid values.
 * @param[in] update_vel        Whether to interpolate velocity.
 * @param[in] update_strainrate Whether to compute strain‑rate and F.
 * @param[in] dt                Time step.
 *
 * @return None.
 */

void G2P_Momentum(MPMspecs &specs,
                  MPMParticleContainer &mpm_pc,
                  amrex::MultiFab &nodaldata,
                  int update_vel,
                  int update_strainrate,
                  amrex::Real dt)
{
    if (testing == 1)
        amrex::Print() << "\n Doing G2P \n";
    mpm_pc.interpolate_from_grid(nodaldata, update_vel, update_strainrate,
                                 specs.order_scheme_directional, specs.periodic,
                                 specs.alpha_pic_flip, dt);
}
#if USE_TEMP
/**
 * @brief Performs grid‑to‑particle (G2P) transfer of temperature and heat flux.
 *
 * Wraps interpolate_from_grid_temperature(), forwarding interpolation order,
 * periodicity, and PIC/FLIP blending parameter.
 *
 * @param[in] specs               Simulation specification structure.
 * @param[in,out] mpm_pc          Particle container.
 * @param[in] nodaldata           Nodal thermal MultiFab.
 * @param[in] update_temperature  Whether to interpolate ΔT.
 * @param[in] update_heatflux     Whether to compute heat flux.
 * @param[in] dt                  Time step (unused).
 *
 * @return None.
 */

void G2P_Temperature(MPMspecs &specs,
                     MPMParticleContainer &mpm_pc,
                     amrex::MultiFab &nodaldata,
                     int update_temperature,
                     int update_heatflux,
                     amrex::Real dt)
{
    if (testing == 1)
        amrex::Print() << "\n Doing G2P \n";
    mpm_pc.interpolate_from_grid_temperature(
        nodaldata, update_temperature, update_heatflux,
        specs.order_scheme_directional, specs.periodic, specs.alpha_pic_flip);
}
#endif

/**
 * @brief Advances particle positions and applies particle‑level boundary
 * conditions.
 *
 * Wraps moveParticles(), forwarding all wall BCs, friction coefficients,
 * wall velocities, and level‑set BC parameters from specs.
 *
 * @param[in] specs   Simulation specification structure.
 * @param[in,out] mpm_pc  Particle container.
 * @param[in] dt      Time step.
 *
 * @return None.
 */

void Update_MP_Positions(MPMspecs &specs,
                         MPMParticleContainer &mpm_pc,
                         amrex::Real dt)
{
    mpm_pc.moveParticles(dt, specs.bclo.data(), specs.bchi.data(),
                         specs.levelset_bc, specs.wall_mu_lo.data(),
                         specs.wall_mu_hi.data(), specs.wall_vel_lo.data(),
                         specs.wall_vel_hi.data(), specs.levelset_wall_mu);
}

/**
 * @brief Updates particle volume, density, and Jacobian.
 *
 * Thin wrapper around updateVolume().
 *
 * @param[in,out] mpm_pc  Particle container.
 *
 * @return None.
 */

void Update_MP_Volume(MPMParticleContainer &mpm_pc)
{
    mpm_pc.updateVolume();
}

/**
 * @brief Updates particle stress and strain using the chosen constitutive
 * model.
 *
 * If current time < applied_strainrate_time:
 *   - Applies externally imposed strain‑rate.
 * Otherwise:
 *   - Applies zero external strain‑rate.
 *
 * Depending on specs.calculate_strain_based_on_delta:
 *   - Uses apply_constitutive_model_delta()  (incremental)
 *   - Uses apply_constitutive_model()        (total strain)
 *
 * @param[in] specs   Simulation specification structure.
 * @param[in,out] mpm_pc  Particle container.
 * @param[in] time    Current simulation time.
 * @param[in] dt      Time step.
 *
 * @return None.
 */

void Calculate_MP_Stress_Strain(MPMspecs &specs,
                                MPMParticleContainer &mpm_pc,
                                amrex::Real time,
                                amrex::Real dt)
{
    if (time < specs.applied_strainrate_time)
    {
        if (specs.calculate_strain_based_on_delta == 1)
        {
            mpm_pc.apply_constitutive_model_delta(dt, specs.applied_strainrate);
        }
        else
        {
            mpm_pc.apply_constitutive_model(dt, specs.applied_strainrate);
        }
    }
    else
    {
        if (specs.calculate_strain_based_on_delta == 1)
        {
            mpm_pc.apply_constitutive_model_delta(dt, 0.0);
        }
        else
        {
            mpm_pc.apply_constitutive_model(dt, 0.0);
        }
    }
}

/**
 * @brief Updates particle neighbor lists and redistributes particles
 * periodically.
 *
 * Every specs.num_redist steps:
 *   - RedistributeLocal()
 *   - fillNeighbors()
 *   - buildNeighborList()
 *
 * Otherwise:
 *   - updateNeighbors()
 *
 * @param[in] specs   Simulation specification structure.
 * @param[in,out] mpm_pc  Particle container.
 * @param[in] steps   Current step index.
 *
 * @return None.
 */

void Redistribute_Fill_Update(MPMspecs &specs,
                              MPMParticleContainer &mpm_pc,
                              int steps)
{
    if (steps % specs.num_redist == 0)
    {
        mpm_pc.RedistributeLocal();
        mpm_pc.fillNeighbors();
        mpm_pc.buildNeighborList(CheckPair());
    }
    else
    {
        mpm_pc.updateNeighbors();
    }
}

/**
 * @brief Opens diagnostic output files and writes header lines.
 *
 * Initializes output streams for:
 *   - TKE/TSE
 *   - Mass‑weighted average velocity components
 *   - Mass‑weighted average velocity magnitude
 *   - Min/max particle positions
 *
 * Only executed if specs.print_diagnostics = 1.
 *
 * @param[in,out] specs  Simulation specification structure.
 *
 * @return None.
 */

void Initialise_Diagnostic_Streams(MPMspecs &specs)
{
    if (specs.print_diagnostics == 0)
        return;
    if (specs.do_calculate_tke_tse)
    {
        std::string fullfilename =
            specs.diagnostic_output_folder + "/" + specs.file_tke_tse;
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            specs.tmp_tke_tse.open(fullfilename.c_str(),
                                   std::ios::out | std::ios::app |
                                       std::ios_base::binary);

            specs.tmp_tke_tse.precision(12);
            specs.tmp_tke_tse << "iter,time,TKE,TSE,TE\n";
        }
        specs.tmp_tke_tse.flush();
    }

    if (specs.do_calculate_mwa_velcomp)
    {
        amrex::Print() << "\n Diag vel comp";
        std::string fullfilename =
            specs.diagnostic_output_folder + "/" + specs.file_mwa_velcomp;
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            specs.tmp_mwa_velcomp.open(fullfilename.c_str(),
                                       std::ios::out | std::ios::app |
                                           std::ios_base::binary);
            specs.tmp_mwa_velcomp.precision(12);
            specs.tmp_mwa_velcomp << "iter,time,xvel";
#if (AMREX_SPACEDIM >= 2)
            specs.tmp_mwa_velcomp << ",yvel";
#endif
#if (AMREX_SPACEDIM == 3)
            specs.tmp_mwa_velcomp << ",zvel";
#endif
            specs.tmp_mwa_velcomp << "\n";
        }
        specs.tmp_mwa_velcomp.flush();
    }

    if (specs.do_calculate_mwa_velmag)
    {
        amrex::Print() << "\n Diag vel mag";
        std::string fullfilename =
            specs.diagnostic_output_folder + "/" + specs.file_mwa_velmag;
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            specs.tmp_mwa_velmag.open(fullfilename.c_str(),
                                      std::ios::out | std::ios::app |
                                          std::ios_base::binary);
            specs.tmp_mwa_velmag.precision(12);
            specs.tmp_mwa_velmag << "iter,time,velmag\n";
        }
        specs.tmp_mwa_velmag.flush();
    }

    if (specs.do_calculate_minmaxpos)
    {
        std::string fullfilename =
            specs.diagnostic_output_folder + "/" + specs.file_minmaxpos;
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            specs.tmp_minmaxpos.open(fullfilename.c_str(),
                                     std::ios::out | std::ios::app |
                                         std::ios_base::binary);
            specs.tmp_minmaxpos.precision(12);
#if (AMREX_SPACEDIM == 1)
            specs.tmp_minmaxpos << "iter,time,xmin,xmax\n";
#elif (AMREX_SPACEDIM == 2)
            specs.tmp_minmaxpos << "iter,time,xmin,ymin,xmax,ymax\n";
#else
            specs.tmp_minmaxpos << "iter,time,xmin,ymin,xmax,ymax,zmin,zmax\n";
#endif
        }

        specs.tmp_minmaxpos.flush();
    }
}

/**
 * @brief Computes and writes all enabled diagnostics for the current step.
 *
 * Diagnostics include:
 *   - Total kinetic and strain energy
 *   - Mass‑weighted average velocity components
 *   - Mass‑weighted average velocity magnitude
 *   - Min/max particle positions
 *
 * Each diagnostic is written to its corresponding output stream.
 *
 * @param[in] specs         Simulation specification structure.
 * @param[in] mpm_pc        Particle container.
 * @param[in] steps         Current step index.
 * @param[in] current_time  Current simulation time.
 *
 * @return None.
 */

void Do_All_Diagnostics(MPMspecs &specs,
                        MPMParticleContainer &mpm_pc,
                        int steps,
                        amrex::Real current_time)
{
    if (specs.do_calculate_tke_tse)
    {

        amrex::Real tke = 0.0, tse = 0.0;
        mpm_pc.Calculate_Total_Energies(tke, tse);
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            specs.tmp_tke_tse << steps << " " << current_time << " " << tke
                              << " " << tse << " " << tke + tse << "\n";
        }
        specs.tmp_tke_tse.flush();
    }

    if (specs.do_calculate_mwa_velcomp)
    {

        amrex::GpuArray<Real, AMREX_SPACEDIM> Vcm;
        mpm_pc.Calculate_MWA_VelocityComponents(Vcm);
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            specs.tmp_mwa_velcomp << steps << " " << current_time;
            for (int dim = 0; dim < AMREX_SPACEDIM; dim++)
            {
                specs.tmp_mwa_velcomp << " " << Vcm[dim];
            }
            specs.tmp_mwa_velcomp << "\n";
        }
        specs.tmp_mwa_velcomp.flush();
    }
    if (specs.do_calculate_mwa_velmag)
    {

        amrex::Real Vmag;
        mpm_pc.Calculate_MWA_VelocityMagnitude(Vmag);
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            specs.tmp_mwa_velcomp << steps << " " << current_time << " " << Vmag
                                  << "\n";
        }
        specs.tmp_mwa_velcomp.flush();
    }

    if (specs.do_calculate_minmaxpos)
    {

        amrex::GpuArray<Real, AMREX_SPACEDIM> minpos;
        amrex::GpuArray<Real, AMREX_SPACEDIM> maxpos;
        mpm_pc.Calculate_MinMaxPos(minpos, maxpos);
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            specs.tmp_minmaxpos << steps << " " << current_time;
            for (int dim = 0; dim < AMREX_SPACEDIM; dim++)
            {
                specs.tmp_minmaxpos << " " << minpos[dim] << " " << maxpos[dim];
            }
            specs.tmp_minmaxpos << "\n";
        }
        specs.tmp_minmaxpos.flush();
    }
}

/**
 * @brief Flushes and closes all diagnostic output streams.
 *
 * Called at the end of the simulation or when diagnostics are no longer needed.
 *
 * @param[in,out] specs  Simulation specification structure.
 *
 * @return None.
 */

void Close_Diagnostic_Streams(MPMspecs &specs)
{
    if (specs.do_calculate_tke_tse)
    {
        specs.tmp_tke_tse.flush();
        specs.tmp_tke_tse.close();
    }

    if (specs.do_calculate_mwa_velcomp)
    {
        specs.tmp_mwa_velcomp.flush();
        specs.tmp_mwa_velcomp.close();
    }

    if (specs.do_calculate_mwa_velmag)
    {
        specs.tmp_mwa_velmag.flush();
        specs.tmp_mwa_velmag.close();
    }
    if (specs.do_calculate_minmaxpos)
    {
        specs.tmp_minmaxpos.flush();
        specs.tmp_minmaxpos.close();
    }
}
