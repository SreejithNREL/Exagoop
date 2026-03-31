#ifndef MPM_PARTICLE_CONTAINER_H_
#define MPM_PARTICLE_CONTAINER_H_
// clang-format off
#include <AMReX_Particles.H>
#include <AMReX_NeighborParticles.H>
#include <mpm_specs.H>
#include <constants.H>
// clang-format on

class MPMParticleContainer
    : public amrex::NeighborParticleContainer<realData::count, intData::count>
{

  public:
    /**
     * @brief Constructs the MPM particle container and initialises the
     *        underlying AMReX NeighborParticleContainer.
     *
     * @param[in] geom      Computational geometry (domain extent, periodicity).
     * @param[in] dmap      Distribution mapping across MPI ranks.
     * @param[in] ba        BoxArray defining the grid decomposition.
     * @param[in] numcells  Neighbour stencil half‑width in grid cells.
     */
    MPMParticleContainer(const amrex::Geometry &geom,
                         const amrex::DistributionMapping &dmap,
                         const amrex::BoxArray &ba,
                         int numcells)

        : NeighborParticleContainer<realData::count, intData::count>(
              geom, dmap, ba, numcells)
    {
    }

    /**
     * @brief Initializes particles by reading from an ASCII particle file
     *        using a chunked, streaming reader.
     *
     * @param[in]  filename                    Path to the ASCII particle file.
     * @param[out] total_mass                  Total mass of MPM particles.
     * @param[out] total_vol                   Total volume of MPM particles.
     * @param[out] total_rigid_mass            Total mass of rigid body 0.
     * @param[out] index_rigid_body_read_so_far Number of rigid bodies found.
     * @param[out] ifrigidnodespresent         1 if any rigid particles exist.
     *
     * @return None.
     */
    void InitParticles(const std::string &filename,
                       Real &total_mass,
                       Real &total_vol,
                       Real &total_rigid_mass,
                       int &index_rigid_body_read_so_far,
                       int &ifrigidnodespresent);

    /**
     * @brief Initializes particles by reading from an HDF5 particle file.
     *
     * Requires compilation with @c AMREX_USE_HDF5. Reads per‑particle datasets
     * from the HDF5 file and inserts particles in chunks using Redistribute().
     *
     * @param[in]  filename                    Path to the HDF5 particle file.
     * @param[out] total_mass                  Total mass of MPM particles.
     * @param[out] total_vol                   Total volume of MPM particles.
     * @param[out] total_rigid_mass            Total mass of rigid body 0.
     * @param[out] index_rigid_body_read_so_far Number of rigid bodies found.
     * @param[out] ifrigidnodespresent         1 if any rigid particles exist.
     *
     * @return None.
     */
    void InitParticlesFromHDF5(const std::string &filename,
                               Real &total_mass,
                               Real &total_vol,
                               Real &total_rigid_mass,
                               int &index_rigid_body_read_so_far,
                               int &ifrigidnodespresent);

    /**
     * @brief Legacy HDF5 particle reader (deprecated, kept for reference).
     *
     * An older implementation of HDF5‑based particle initialisation.
     * Superseded by InitParticlesFromHDF5(); retained for comparison only.
     * Requires @c AMREX_USE_HDF5.
     *
     * @param[in]  filename                    Path to the HDF5 particle file.
     * @param[out] total_mass                  Total mass of MPM particles.
     * @param[out] total_vol                   Total volume of MPM particles.
     * @param[out] total_rigid_mass            Total mass of rigid body 0.
     * @param[out] index_rigid_body_read_so_far Number of rigid bodies found.
     * @param[out] ifrigidnodespresent         1 if any rigid particles exist.
     *
     * @return None.
     */
    void InitParticlesFromHDF5old(const std::string &filename,
                                  Real &total_mass,
                                  Real &total_vol,
                                  Real &total_rigid_mass,
                                  int &index_rigid_body_read_so_far,
                                  int &ifrigidnodespresent);

    /**
     * @brief Autogenerates a uniform lattice of particles inside a bounding
     * box.
     *
     * Places particles at evenly spaced sub‑cell positions within
     * [mincoords, maxcoords], assigning uniform material properties. When
     * @c USE_TEMP is enabled, thermal fields (temperature, specific heat,
     * thermal conductivity, heat source) are also initialised.
     *
     * @param[in]  mincoords           Lower corner of the particle region.
     * @param[in]  maxcoords           Upper corner of the particle region.
     * @param[in]  vel                 Initial velocity for all particles.
     * @param[in]  ppc                 Particles per cell in each dimension.
     * @param[in]  dens                Density.
     * @param[in]  constmodel          Constitutive model index (0 = elastic,
     *                                 1 = Newtonian fluid).
     * @param[in]  E                   Young's modulus.
     * @param[in]  nu                  Poisson's ratio.
     * @param[in]  bulkMod             Bulk modulus (fluid model).
     * @param[in]  Gama_pres           Gamma exponent for EOS pressure.
     * @param[in]  visc                Dynamic viscosity.
     * @param[in]  T                   Initial temperature (USE_TEMP only).
     * @param[in]  thermcond           Thermal conductivity (USE_TEMP only).
     * @param[in]  cp                  Specific heat (USE_TEMP only).
     * @param[in]  heatsrc             Volumetric heat source (USE_TEMP only).
     * @param[in]  do_multi_part_per_cell  Whether to place multiple ppc>1.
     * @param[out] total_mass          Total mass of generated particles.
     * @param[out] total_vol           Total volume of generated particles.
     *
     * @return None.
     */
    void InitParticles(Real mincoords[AMREX_SPACEDIM],
                       Real maxcoords[AMREX_SPACEDIM],
                       Real vel[AMREX_SPACEDIM],
                       int ppc[AMREX_SPACEDIM],
                       Real dens,
                       int constmodel,
                       Real E,
                       Real nu,
                       Real bulkMod,
                       Real Gama_pres,
                       Real visc,
#if USE_TEMP
                       amrex::Real T,
                       amrex::Real thermcond,
                       amrex::Real cp,
                       amrex::Real heatsrc,
#endif
                       int do_multi_part_per_cell,
                       Real &total_mass,
                       Real &total_vol);

    /**
     * @brief Advances particle positions and applies boundary conditions.
     *
     * Loops over all real particles, updates positions using the stored
     * particle velocity, and enforces Cartesian wall and level‑set boundary
     * conditions at domain edges. Rigid particles are moved using the
     * prescribed rigid‑body velocity.
     *
     * @param[in] dt            Time step size.
     * @param[in] bclo          Lower boundary condition type per dimension.
     * @param[in] bchi          Upper boundary condition type per dimension.
     * @param[in] lsetbc        Level‑set BC flag (unused when USE_EB=0).
     * @param[in] wall_mu_lo    Wall friction coefficient, lower boundary.
     * @param[in] wall_mu_hi    Wall friction coefficient, upper boundary.
     * @param[in] wall_vel_lo   Wall velocity, lower boundary (row‑major).
     * @param[in] wall_vel_hi   Wall velocity, upper boundary (row‑major).
     * @param[in] lset_wall_mu  Level‑set wall friction (unused when USE_EB=0).
     *
     * @return None.
     */
    void moveParticles(const amrex::Real &dt,
                       int bclo[AMREX_SPACEDIM],
                       int bchi[AMREX_SPACEDIM],
                       int lsetbc,
                       amrex::Real wall_mu_lo[AMREX_SPACEDIM],
                       amrex::Real wall_mu_hi[AMREX_SPACEDIM],
                       amrex::Real wall_vel_lo[AMREX_SPACEDIM * AMREX_SPACEDIM],
                       amrex::Real wall_vel_hi[AMREX_SPACEDIM * AMREX_SPACEDIM],
                       amrex::Real lset_wall_mu);

    /**
     * @brief Updates particle volume, density, and Jacobian from the
     *        deformation gradient.
     *
     * Computes J = det(F) from the stored deformation gradient tensor,
     * then sets volume = vol_init * J and density = mass / volume.
     *
     * @return None.
     */
    void updateVolume();

    /**
     * @brief Prints particle data to stdout for debugging (currently a no‑op).
     *
     * @return None.
     */
    void PrintParticleData();

    /**
     * @brief Computes the exact velocity for a 1D axial‑bar vibration mode.
     *
     * Evaluates the analytical solution v(t) = v0 * sin(n π x / L) * cos(ω t)
     * where ω = (n π / 2L) √(E / ρ).
     *
     * @param[in] modenumber  Vibration mode number n.
     * @param[in] E           Young's modulus.
     * @param[in] rho         Density.
     * @param[in] v0          Amplitude of the initial velocity perturbation.
     * @param[in] L           Length of the bar.
     * @param[in] time        Current simulation time.
     *
     * @return amrex::Real  Exact velocity at the given time.
     */
    amrex::Real CalculateExactVelocity(int modenumber,
                                       amrex::Real E,
                                       amrex::Real rho,
                                       amrex::Real v0,
                                       amrex::Real L,
                                       amrex::Real time);

    /**
     * @brief Writes particle data to an ASCII file with a time‑encoded
     * filename.
     *
     * @param[in] prefix_particlefilename  Directory/prefix for output files.
     * @param[in] num_of_digits_in_filenames Number of digits in time suffix.
     * @param[in] n                        Current output time (used in name).
     *
     * @return None.
     */
    void writeAsciiFiles(std::string prefix_particlefilename,
                         int num_of_digits_in_filenames,
                         amrex::Real n);

    /**
     * @brief Writes particle data to an AMReX binary plotfile.
     *
     * @param[in] prefix_particlefilename  Directory/prefix for output files.
     * @param[in] num_of_digits_in_filenames Number of digits in step suffix.
     * @param[in] n                        Current output step index.
     *
     * @return None.
     */
    void writeParticles(std::string prefix_particlefilename,
                        int num_of_digits_in_filenames,
                        const int n);

    /**
     * @brief Writes a complete particle checkpoint to disk.
     *
     * Outputs binary particle data, a header file, and all metadata needed
     * to restart the simulation from this point.
     *
     * @param[in] prefix_particlefilename  Directory/prefix for checkpoint
     * files.
     * @param[in] num_of_digits_in_filenames Number of digits in step suffix.
     * @param[in] cur_time                 Current simulation time.
     * @param[in] nstep                    Current step number.
     * @param[in] output_it                Output iteration index.
     *
     * @return None.
     */
    void writeCheckpointFile(std::string prefix_particlefilename,
                             int num_of_digits_in_filenames,
                             amrex::Real cur_time,
                             int nstep,
                             int output_it);

    /**
     * @brief Writes the header file for a checkpoint or plotfile directory.
     *
     * @param[in] name                  Path to the output directory.
     * @param[in] is_checkpoint         True if writing a checkpoint.
     * @param[in] cur_time              Current simulation time.
     * @param[in] nstep                 Current step number.
     * @param[in] EB_generate_max_level EB refinement depth (unused when
     *                                  USE_EB=0).
     * @param[in] output_it             Output iteration index.
     *
     * @return None.
     */
    void WriteHeader(const std::string &name,
                     bool is_checkpoint,
                     amrex::Real cur_time,
                     int nstep,
                     int EB_generate_max_level,
                     int output_it) const;

    /**
     * @brief Reads particle data and metadata from a checkpoint directory.
     *
     * @param[in,out] restart_chkfile  Path to the checkpoint directory.
     * @param[out]    nstep            Step number read from the checkpoint.
     * @param[out]    cur_time         Simulation time read from the checkpoint.
     * @param[out]    output_it        Output iteration index from checkpoint.
     *
     * @return None.
     */
    void readCheckpointFile(std::string &restart_chkfile,
                            int &nstep,
                            double &cur_time,
                            int &output_it);

    /**
     * @brief Returns 1 if any rigid‑body (phase=1) particles are present.
     *
     * Performs a GPU‑parallel reduction checking the phase flag of every
     * particle; returns 1 if at least one rigid particle is found, 0 otherwise.
     *
     * @return int  1 if rigid particles are present, 0 otherwise.
     */
    int checkifrigidnodespresent();

    /**
     * @brief Computes the total mass of rigid‑body particles for a given
     *        body ID.
     *
     * @param[in]  body_id     Rigid body identifier to sum over.
     * @param[out] total_mass  Accumulated mass of matching particles.
     *
     * @return None.
     */
    void Calculate_Total_Mass_RigidParticles(int body_id,
                                             amrex::Real &total_mass);

    /**
     * @brief Computes the total volume of rigid‑body particles for a given
     *        body ID.
     *
     * @param[in] body_id  Rigid body identifier to sum over.
     *
     * @return amrex::Real  Total volume of matching rigid particles.
     */
    amrex::Real Calculate_Total_Vol_RigidParticles(int body_id);

    /**
     * @brief Returns the maximum Y‑coordinate among all particles.
     *
     * Typically used to track the top surface of a deforming body for spring
     * compression diagnostics.
     *
     * @return amrex::Real  Maximum particle Y‑position.
     */
    amrex::Real GetPosPiston();

    /**
     * @brief Returns the minimum Y‑coordinate among all rigid‑body‑0 particles.
     *
     * Used to track the bottom face of the piston rigid body.
     *
     * @return amrex::Real  Minimum Y‑position of rigid body 0.
     */
    amrex::Real GetPosSpring();

    /**
     * @brief Estimates the normal contact force at the top and bottom surfaces.
     *
     * @param[in]  gravity     Gravitational acceleration vector.
     * @param[out] Fy_top      Estimated upward reaction force at the top.
     * @param[out] Fy_bottom   Estimated downward reaction force at the bottom.
     *
     * @return None.
     */
    void CalculateSurfaceIntegralTop(Array<Real, AMREX_SPACEDIM> gravity,
                                     Real &Fy_top,
                                     Real &Fy_bottom);

    /**
     * @brief Performs particle‑to‑grid (P2G) transfer of mass, momentum, and
     *        forces using basis‑function interpolation (MPM P2G step).
     *
     * @param[in,out] nodaldata                Nodal MultiFab to accumulate
     * into.
     * @param[in]     gravity                  Gravitational acceleration.
     * @param[in]     external_loads_present   Flag enabling body forces.
     * @param[in]     force_slab_lo            Lower corner of body‑force
     * region.
     * @param[in]     force_slab_hi            Upper corner of body‑force
     * region.
     * @param[in]     extforce                 External force vector.
     * @param[in]     update_mass              Flag to accumulate nodal mass.
     * @param[in]     update_vel               Flag to accumulate nodal
     * momentum.
     * @param[in]     update_forces            Flag to accumulate nodal forces.
     * @param[in]     mass_tolerance           Minimum nodal mass threshold.
     * @param[in]     order_scheme_directional Per‑dimension interpolation
     * order.
     * @param[in]     periodic                 Per‑dimension periodicity flags.
     *
     * @return None.
     */
    void deposit_onto_grid_momentum(
        MultiFab &nodaldata,
        Array<Real, AMREX_SPACEDIM> gravity,
        int external_loads_present,
        Array<Real, AMREX_SPACEDIM> force_slab_lo,
        Array<Real, AMREX_SPACEDIM> force_slab_hi,
        Array<Real, AMREX_SPACEDIM> extforce,
        int update_mass,
        int update_vel,
        int update_forces,
        amrex::Real mass_tolerance,
        GpuArray<int, AMREX_SPACEDIM> order_scheme_directional,
        GpuArray<int, AMREX_SPACEDIM> periodic);

    /**
     * @brief Performs particle‑to‑grid (P2G) transfer of thermal quantities.
     *
     * Deposits mass×specific_heat, temperature×mass×cp, and heat source terms
     * onto nodal data. Requires @c USE_TEMP.
     *
     * @param[in,out] nodaldata                Nodal MultiFab to accumulate
     * into.
     * @param[in]     resetnodaldata_to_zero   Zero thermal fields before P2G.
     * @param[in]     update_mass_temp         Accumulate mass×cp field.
     * @param[in]     update_source            Accumulate heat source field.
     * @param[in]     mass_tolerance           Minimum nodal mass threshold.
     * @param[in]     order_scheme_directional Per‑dimension interpolation
     * order.
     * @param[in]     periodic                 Per‑dimension periodicity flags.
     *
     * @return None.
     */
    void deposit_onto_grid_temperature(
        MultiFab &nodaldata,
        bool resetnodaldata_to_zero,
        bool update_mass_temp,
        bool update_source,
        amrex::Real mass_tolerance,
        GpuArray<int, AMREX_SPACEDIM> order_scheme_directional,
        GpuArray<int, AMREX_SPACEDIM> periodic);

    /**
     * @brief Computes nodal surface normals from particle mass‑weighted basis
     *        gradients (P2G normal deposition).
     *
     * @param[in,out] nodaldata                Nodal MultiFab; normal components
     *                                         are accumulated here.
     * @param[in]     mass_tolerance           Minimum nodal mass threshold.
     * @param[in]     order_scheme_directional Per‑dimension interpolation
     * order.
     * @param[in]     periodic                 Per‑dimension periodicity flags.
     *
     * @return None.
     */
    void calculate_nodal_normal(
        MultiFab &nodaldata,
        amrex::Real mass_tolerance,
        GpuArray<int, AMREX_SPACEDIM> order_scheme_directional,
        GpuArray<int, AMREX_SPACEDIM> periodic);

    /**
     * @brief Deposits rigid‑body particle quantities onto the grid
     *        (stub — body currently disabled).
     *
     * Intended to perform a P2G transfer using only phase=1 particles,
     * segregating rigid‑body contributions on the nodal MultiFab.
     * The implementation is currently a no‑op placeholder.
     *
     * @param[in,out] nodaldata                Nodal MultiFab (unused in stub).
     * @param[in]     gravity                  Gravitational acceleration.
     * @param[in]     external_loads_present   Flag enabling body forces.
     * @param[in]     force_slab_lo            Lower corner of body‑force
     * region.
     * @param[in]     force_slab_hi            Upper corner of body‑force
     * region.
     * @param[in]     extforce                 External force vector.
     * @param[in]     update_massvel           Flag to update mass/velocity.
     * @param[in]     update_forces            Flag to update nodal forces.
     * @param[in]     mass_tolerance           Minimum nodal mass threshold.
     * @param[in]     order_scheme_directional Per‑dimension interpolation
     * order.
     * @param[in]     periodic                 Per‑dimension periodicity flags.
     *
     * @return None.
     */
    void deposit_onto_grid_rigidnodesonly(
        MultiFab &nodaldata,
        Array<Real, AMREX_SPACEDIM> gravity,
        int external_loads_present,
        Array<Real, AMREX_SPACEDIM> force_slab_lo,
        Array<Real, AMREX_SPACEDIM> force_slab_hi,
        Array<Real, AMREX_SPACEDIM> extforce,
        int update_massvel,
        int update_forces,
        amrex::Real mass_tolerance,
        GpuArray<int, AMREX_SPACEDIM> order_scheme_directional,
        GpuArray<int, AMREX_SPACEDIM> periodic);

    /**
     * @brief Checks that all particles remain inside the problem domain and
     *        aborts if any have escaped.
     *
     * @return None.
     */
    void checkifallparticlesinsidedomain();

    /**
     * @brief Assigns a uniform velocity to all particles belonging to a rigid
     *        body.
     *
     * Sets both the primary velocity (xvel) and the FLIP velocity (xvel_prime)
     * for every particle with phase=1 and the given rigid_body_id.
     *
     * @param[in] rigid_body_id  ID of the rigid body to update.
     * @param[in] velocity       Prescribed velocity vector.
     *
     * @return None.
     */
    void
    UpdateRigidParticleVelocities(int rigid_body_id,
                                  Array<amrex::Real, AMREX_SPACEDIM> velocity);

    /**
     * @brief Interpolates nodal velocity and strain‑rate back to particles
     *        (G2P transfer, momentum).
     *
     * Applies PIC/FLIP blending: v_pic = N_I v_I,
     * v_flip = v_p + Σ N_I Δv_I, blended as (1−α) v_pic + α v_flip.
     * Also computes the velocity gradient and updates the deformation gradient
     * F and strain‑rate tensor.
     *
     * @param[in,out] nodaldata                Nodal MultiFab containing
     *                                         velocity and Δv fields.
     * @param[in]     update_vel               Flag to interpolate velocity.
     * @param[in]     update_strainrate        Flag to compute strain‑rate and
     * F.
     * @param[in]     order_scheme_directional Per‑dimension interpolation
     * order.
     * @param[in]     periodic                 Per‑dimension periodicity flags.
     * @param[in]     alpha_pic_flip           Blending factor α (0=PIC,
     * 1=FLIP).
     * @param[in]     dt                       Time step (for F update).
     *
     * @return None.
     */
    void interpolate_from_grid(
        MultiFab &nodaldata,
        int update_vel,
        int update_strainrate,
        GpuArray<int, AMREX_SPACEDIM> order_scheme_directional,
        GpuArray<int, AMREX_SPACEDIM> periodic,
        amrex::Real alpha_pic_flip,
        amrex::Real dt);

    /**
     * @brief Interpolates temperature and heat‑flux from the nodal grid to
     *        particles (G2P transfer, thermal). Requires @c USE_TEMP.
     *
     * @param[in,out] nodaldata                Nodal thermal MultiFab.
     * @param[in]     update_temperature       Flag to interpolate ΔT.
     * @param[in]     update_heatflux          Flag to interpolate heat flux.
     * @param[in]     order_scheme_directional Per‑dimension interpolation
     * order.
     * @param[in]     periodic                 Per‑dimension periodicity flags.
     * @param[in]     alpha_pic_flip           PIC/FLIP blending factor.
     *
     * @return None.
     */
    void interpolate_from_grid_temperature(
        MultiFab &nodaldata,
        bool update_temperature,
        bool update_heatflux,
        GpuArray<int, AMREX_SPACEDIM> order_scheme_directional,
        GpuArray<int, AMREX_SPACEDIM> periodic,
        amrex::Real alpha_pic_flip);

    /**
     * @brief Advances particle positions using nodal (grid) velocities
     *        interpolated with a linear basis function.
     *
     * @param[in,out] nodaldata   Nodal MultiFab containing velocity field.
     * @param[in]     dt          Time step size.
     * @param[in]     bclo        Lower boundary condition types.
     * @param[in]     bchi        Upper boundary condition types.
     * @param[in]     order_scheme Interpolation order (1=linear).
     *
     * @return None.
     */
    void move_particles_from_nodevel(MultiFab &nodaldata,
                                     const amrex::Real &dt,
                                     GpuArray<int, AMREX_SPACEDIM> bclo,
                                     GpuArray<int, AMREX_SPACEDIM> bchi,
                                     int order_scheme);

    /**
     * @brief Computes the total kinetic energy (TKE) and total strain energy
     *        (TSE) of all material particles.
     *
     * @param[out] TKE  Total kinetic energy Σ ½ m |v|².
     * @param[out] TSE  Total strain energy Σ ½ V σ:ε.
     *
     * @return None.
     */
    void Calculate_Total_Energies(Real &TKE, Real &TSE);

    /**
     * @brief Computes the mass‑weighted average velocity components of all
     *        material particles.
     *
     * @param[out] Vcm  Mass‑weighted mean velocity vector.
     *
     * @return None.
     */
    void Calculate_MWA_VelocityComponents(
        amrex::GpuArray<Real, AMREX_SPACEDIM> &Vcm);

    /**
     * @brief Computes the minimum and maximum particle positions in each
     *        spatial dimension.
     *
     * @param[out] minpos  Minimum position per dimension.
     * @param[out] maxpos  Maximum position per dimension.
     *
     * @return None.
     */
    void Calculate_MinMaxPos(amrex::GpuArray<Real, AMREX_SPACEDIM> &minpos,
                             amrex::GpuArray<Real, AMREX_SPACEDIM> &maxpos);

    /**
     * @brief Computes the mass‑weighted average speed (velocity magnitude) of
     *        all material particles.
     *
     * @param[out] Vcm  Mass‑weighted mean speed.
     *
     * @return None.
     */
    void Calculate_MWA_VelocityMagnitude(amrex::Real &Vcm);

    /**
     * @brief Computes the effective spring constant from particle strain
     * energy.
     *
     * Uses k = 2U / δ² where U is the total elastic strain energy and
     * δ = L0 − V_total / Area is the axial compression. Returns the
     * restoring force F = 2U / δ.
     *
     * @param[in] Area  Cross‑sectional area of the specimen.
     * @param[in] L0    Reference (undeformed) length.
     *
     * @return amrex::Real  Restoring force (zero if deflection is negligible).
     */
    amrex::Real CalculateEffectiveSpringConstant(amrex::Real Area,
                                                 amrex::Real L0);

    /**
     * @brief Computes and writes the centre‑of‑mass velocity of all particles.
     *
     * @param[out] Vcm  Centre‑of‑mass speed.
     *
     * @return None.
     */
    void CalculateVelocity(Real &Vcm);

    /**
     * @brief Computes and writes the tip velocity for a cantilever beam test.
     *
     * @param[out] Vcm  Tip velocity of the cantilever.
     *
     * @return None.
     */
    void CalculateVelocityCantilever(Real &Vcm);

    /**
     * @brief Computes the total number of rigid‑body particles for a given
     *        body ID.
     *
     * @param[in]  body_id    Rigid body identifier.
     * @param[out] total_num  Count of matching rigid particles.
     *
     * @return None.
     */
    void Calculate_Total_Number_of_rigid_particles(int body_id, int &total_num);

    /**
     * @brief Computes the total number of material (phase=0) particles.
     *
     * @param[out] total_num  Count of material particles across all ranks.
     *
     * @return None.
     */
    void Calculate_Total_Number_of_MaterialParticles(int &total_num);

    /**
     * @brief Computes the total mass of all material (phase=0) particles.
     *
     * @param[out] total_mass  Accumulated mass across all MPI ranks.
     *
     * @return None.
     */
    void Calculate_Total_Mass_MaterialPoints(Real &total_mass);

    /**
     * @brief Computes the total volume of all material (phase=0) particles.
     *
     * @param[out] total_vol  Accumulated volume across all MPI ranks.
     *
     * @return None.
     */
    void Calculate_Total_Vol_MaterialPoints(Real &total_vol);

    /**
     * @brief Writes the tip deflection of a cantilever beam to a diagnostic
     *        file.
     *
     * @return None.
     */
    void WriteDeflectionCantilever();

    /**
     * @brief Locates the water‑front position (leading edge of the fluid) along
     *        the X direction.
     *
     * @param[out] Xwf  X‑coordinate of the water front.
     *
     * @return None.
     */
    void FindWaterFront(Real &Xwf);

    /**
     * @brief Applies the constitutive model to all material particles using
     *        total‑strain formulation.
     *
     * Updates stress from the accumulated strain tensor. Supports linear
     * elastic (model=0) and Newtonian fluid with weakly‑compressible EOS
     * (model=1).
     *
     * @param[in] dt                  Time step for strain accumulation.
     * @param[in] applied_strainrate  Optional external axial strain‑rate.
     *
     * @return None.
     */
    void apply_constitutive_model(const amrex::Real &dt,
                                  amrex::Real applied_strainrate);

    /**
     * @brief Applies the constitutive model using an incremental (delta‑strain)
     *        formulation.
     *
     * Computes Δσ = C : Δε and adds the increment to the current stress,
     * suitable for implicit or sub‑stepped formulations.
     *
     * @param[in] dt                  Time step for incremental strain.
     * @param[in] applied_strainrate  Optional external axial strain‑rate.
     *
     * @return None.
     */
    void apply_constitutive_model_delta(const amrex::Real &dt,
                                        amrex::Real applied_strainrate);

    /**
     * @brief Interpolates nodal mass back to particles (G2P mass transfer).
     *
     * @param[in,out] nodaldata   Nodal MultiFab containing mass field.
     * @param[in]     order_scheme Interpolation order.
     *
     * @return None.
     */
    void interpolate_mass_from_grid(MultiFab &nodaldata, int order_scheme);

    /**
     * @brief Updates the Eulerian phase‑field / level‑set representation from
     *        particle positions.
     *
     * Deposits a smooth indicator function onto @p phasedata using a Gaussian
     * kernel centred at each particle, then normalises by the kernel volume.
     *
     * @param[in,out] phasedata    Eulerian MultiFab to update.
     * @param[in]     refratio    Refinement ratio for the phase‑field grid.
     * @param[in]     smoothfactor Kernel bandwidth scale factor.
     *
     * @return None.
     */
    void update_phase_field(MultiFab &nodaldata,
                            int refratio,
                            amrex::Real smoothfactor);

    /**
     * @brief Computes the stable time step for the MPM update using a CFL
     *        condition.
     *
     * Inspects all particles to find the maximum wave speed and returns
     * dt = CFL * dx / c_max, clamped to [dt_min, dt_max] from the spec.
     *
     * @param[in] specs  Simulation specification structure containing CFL,
     *                   dt_min_limit, dt_max_limit, and fixed_timestep.
     *
     * @return amrex::Real  Stable time step size.
     */
    amrex::Real
    // Calculate_time_step(MPMspecs &specs, amrex::Real CFL, amrex::Real dtmax,
    // amrex::Real dtmin);
    Calculate_time_step(MPMspecs &specs);

    /**
     * @brief Removes particles located inside the embedded boundary (EB)
     * region.
     *
     * Requires @c USE_EB. Queries the level‑set field for each particle;
     * particles with a negative level‑set value (inside the EB) are marked
     * invalid and removed on the next Redistribute() call.
     *
     * @return None.
     */
    void removeParticlesInsideEB();

  private:
    ParticleType generate_particle(amrex::Real coords[AMREX_SPACEDIM],
                                   amrex::Real vel[AMREX_SPACEDIM],
                                   amrex::Real dens,
                                   amrex::Real vol,
                                   int constmodel,
                                   amrex::Real E,
                                   amrex::Real nu,
                                   amrex::Real bulkmod,
                                   amrex::Real Gama_pres,
                                   amrex::Real visc
#if USE_TEMP
                                   ,
                                   amrex::Real T,
                                   amrex::Real cp,
                                   amrex::Real thermcond,
                                   amrex::Real heatsrc
#endif
    );
};
#endif
