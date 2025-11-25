#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <aesthetics.H>
#include <constants.H>
#include <mpm_eb.H>
#include <mpm_init.H>
#include <mpm_particle_container.H>
#include <nodal_data_ops.H>

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
    Print() << "\nWriting outputs at step, time:" << steps << ", " << time;

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

#endif
