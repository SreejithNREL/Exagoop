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
    bool rewrite_checkpoint);
#endif
