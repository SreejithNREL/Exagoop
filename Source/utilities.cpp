#include <AMReX.H> // for amrex::Print and amrex::Real
// #include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <aesthetics.H>
#include <iomanip>  // for std::setprecision
#include <iostream> // optional, if you use std::cout
#include <mpm_eb.H>
#include <nodal_data_ops.H>
#include <sstream> // optional, if you later use string streams
#include <string>  // for std::string

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

    BL_PROFILE_VAR_STOP(outputs);
}

void P2G_Momentum(MPMspecs &specs,
                  MPMParticleContainer &mpm_pc,
                  amrex::MultiFab &nodaldata,
                  int update_massvel,
                  int update_forces)
{
    mpm_pc.deposit_onto_grid_momentum(
        nodaldata, specs.gravity, specs.external_loads_present,
        specs.force_slab_lo, specs.force_slab_hi, specs.extforce,
        update_massvel, update_forces, specs.mass_tolerance,
        specs.order_scheme_directional, specs.periodic);
}

void Apply_Nodal_BCs(amrex::Geometry &geom,
                     amrex::MultiFab &nodaldata,
                     MPMspecs &specs,
                     amrex::Real dt)
{

    nodal_bcs(geom, nodaldata, specs.bclo.data(), specs.bchi.data(),
              specs.wall_mu_lo.data(), specs.wall_mu_hi.data(),
              specs.wall_vel_lo.data(), specs.wall_vel_hi.data(), dt);

    if (mpm_ebtools::using_levelset_geometry)
    {
        nodal_levelset_bcs(nodaldata, geom, dt, specs.levelset_bc,
                           specs.levelset_wall_mu);
    }

    // Calculate velocity diff
    store_delta_velocity(nodaldata);
}

void G2P_Momentum(MPMspecs &specs,
                  MPMParticleContainer &mpm_pc,
                  amrex::MultiFab &nodaldata,
                  int update_vel,
                  int update_strainrate,
                  amrex::Real dt)
{
    mpm_pc.interpolate_from_grid(nodaldata, update_vel, update_strainrate,
                                 specs.order_scheme_directional, specs.periodic,
                                 specs.alpha_pic_flip, dt);
}

void Update_MP_Positions(MPMspecs &specs,
                         MPMParticleContainer &mpm_pc,
                         amrex::Real dt)
{
    mpm_pc.moveParticles(dt, specs.bclo.data(), specs.bchi.data(),
                         specs.levelset_bc, specs.wall_mu_lo.data(),
                         specs.wall_mu_hi.data(), specs.wall_vel_lo.data(),
                         specs.wall_vel_hi.data(), specs.levelset_wall_mu);
}

void Update_MP_Volume(MPMParticleContainer &mpm_pc)
{
    mpm_pc.updateVolume();
}

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
