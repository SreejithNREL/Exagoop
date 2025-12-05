#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <aesthetics.H>
#include <constants.H>
#include <mpm_eb.H>
#include <mpm_init.H>
#include <mpm_particle_container.H>
#include <nodal_data_ops.H>
#include <mpm_check_pair.H>

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

void P2G_Momentum(MPMspecs &specs, MPMParticleContainer &mpm_pc, amrex::MultiFab &nodaldata,int update_mass,int update_vel, int update_forces);
void Apply_Nodal_BCs(amrex::Geometry &geom,amrex::MultiFab &nodaldata,MPMspecs &specs,amrex::Real dt);
void G2P_Momentum(MPMspecs &specs, MPMParticleContainer &mpm_pc, amrex::MultiFab &nodaldata, int update_vel, int update_strainrate, amrex::Real dt);
void Update_MP_Positions(MPMspecs &specs, MPMParticleContainer &mpm_pc, amrex::Real dt);
void Update_MP_Volume(MPMParticleContainer &mpm_pc);
void Calculate_MP_Stress_Strain(MPMspecs &specs,MPMParticleContainer &mpm_pc,amrex::Real time, amrex::Real dt);
void Redistribute_Fill_Update(MPMspecs &specs,MPMParticleContainer &mpm_pc, int steps);
void Initialise_Diagnostic_Streams(MPMspecs &specs);
void Do_All_Diagnostics(MPMspecs &specs, MPMParticleContainer &mpm_pc,int steps, amrex::Real current_time);
void Close_Diagnostic_Streams(MPMspecs &specs);
