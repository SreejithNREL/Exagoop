// clang-format off
#include <mpm_particle_container.H>
#include <constants.H>
#include <mpm_eb.H>
#include <aesthetics.H>
#include <nodal_data_ops.H>

#ifdef AMREX_USE_HDF5
#include <hdf5.h>
#endif

// clang-format on

/**
 * @brief Populates the list of nodal data variable names.
 *
 * Appends human‑readable names for each nodal data component in the
 * nodal MultiFab. These names are used for diagnostics, plotfiles,
 * and debugging output.
 *
 * @param[out] nodaldata_names  Vector of strings to be filled with names.
 *
 * @return None.
 */
void Name_Nodaldata_Variables(amrex::Vector<std::string> &nodaldata_names)
{
    nodaldata_names.push_back("mass");
    nodaldata_names.push_back("vel_x");
    nodaldata_names.push_back("vel_y");
    nodaldata_names.push_back("vel_z");
    nodaldata_names.push_back("force_x");
    nodaldata_names.push_back("force_y");
    nodaldata_names.push_back("force_z");
    nodaldata_names.push_back("delta_velx");
    nodaldata_names.push_back("delta_vely");
    nodaldata_names.push_back("delta_velz");
    nodaldata_names.push_back("mass_old");
    nodaldata_names.push_back("VELX_RIGID_INDEX");
    nodaldata_names.push_back("VELY_RIGID_INDEX");
    nodaldata_names.push_back("VELZ_RIGID_INDEX");
    nodaldata_names.push_back("MASS_RIGID_INDEX");
    nodaldata_names.push_back("STRESS_INDEX");
    nodaldata_names.push_back("RIGID_BODY_ID");
    nodaldata_names.push_back("NX");
    nodaldata_names.push_back("NY");
    nodaldata_names.push_back("NZ");
#if USE_TEMP
    nodaldata_names.push_back("MASS_SPHEAT");
    nodaldata_names.push_back("MASS_SPHEAT_TEMP");
    nodaldata_names.push_back("TEMPERATURE");
    nodaldata_names.push_back("SOURCE_TEMP_INDEX");
    nodaldata_names.push_back("DELTA_TEMPERATURE");
#endif
}

/**
 * @brief Initializes the computational domain, geometry, nodal data, and
 * level‑set structures.
 *
 * Performs the following:
 *  - Defines the physical RealBox from user specs.
 *  - Constructs the index‑space domain.
 *  - Builds AMReX Geometry, BoxArray, and DistributionMapping.
 *  - Determines ghost‑cell requirements based on interpolation order.
 *  - Initializes nodal MultiFab and (optionally) level‑set MultiFab.
 *  - Sets directional spline order based on periodicity and grid size.
 *
 * @param[in,out] specs                 User‑defined MPM simulation
 * specifications.
 * @param[out]    geom                  AMReX Geometry for the primary grid.
 * @param[out]    geom_levset           Geometry for the refined level‑set grid.
 * @param[out]    ba                    BoxArray defining the domain
 * decomposition.
 * @param[out]    dm                    DistributionMapping for parallel layout.
 * @param[out]    ng_cells              Number of ghost cells for particle‑grid
 * ops.
 * @param[out]    ng_cells_nodaldata    Number of ghost cells for nodal data.
 * @param[out]    nodaldata             MultiFab storing nodal quantities.
 * @param[out]    levset_data           MultiFab storing level‑set field (if
 * enabled).
 * @param[out]    nodaldata_names       Vector of nodal variable names.
 *
 * @return None.
 */

void Initialise_Domain(MPMspecs &specs,
                       Geometry &geom,
                       Geometry &geom_levset,
                       BoxArray &ba,
                       DistributionMapping &dm,
                       int &ng_cells,
                       int &ng_cells_nodaldata,
                       MultiFab &nodaldata,
                       MultiFab &levset_data,
                       amrex::Vector<std::string> &nodaldata_names)
{
    PrintMessage("\n Setting up problem variables", print_length, true);

    // RealBox from specs (dimension-aware)
    int coord = 0;
    RealBox real_box;
    for (int n = 0; n < AMREX_SPACEDIM; ++n)
    {
        real_box.setLo(n, specs.plo[n]);
        real_box.setHi(n, specs.phi[n]);
    }

    // Index-space domain (dimension-aware)
    IntVect domain_lo{AMREX_D_DECL(0, 0, 0)};
    IntVect domain_hi;
    {
        int hi[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            hi[d] = specs.ncells[d] - 1;
        }
#if (AMREX_SPACEDIM == 1)
        domain_hi = IntVect(hi[0]);
#elif (AMREX_SPACEDIM == 2)
        domain_hi = IntVect(hi[0], hi[1]);
#else
        domain_hi = IntVect(hi[0], hi[1], hi[2]);
#endif
    }

    const Box domain(domain_lo, domain_hi);

    // Geometry
    geom.define(domain, &real_box, coord, specs.periodic.data());

    // BoxArray
    ba.define(domain);
    ba.maxSize(specs.max_grid_size);

    // Distribution mapping
    dm.define(ba);

    // Variable names
    Name_Nodaldata_Variables(nodaldata_names);

    // Ghost cells for particle data
    ng_cells = 1; // Defining number of ghost cells for particle data
    if (specs.order_scheme == 3)
    {
        ng_cells = 2;
    }
    if (specs.order_scheme == 2)
    {
        ng_cells = 3;
    }

    // Ghost cells for nodal data
    if (specs.order_scheme == 1)
    {
        ng_cells_nodaldata = 1;
    }
    else if (specs.order_scheme == 2)
    {
        amrex::Print() << "\n Yes the order is 2";
        ng_cells_nodaldata = 3;

        // Set directional order-scheme based on periodicity and grid size
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            const int ncd = specs.ncells[d];
            const int periodic_d = specs.periodic[d];
            // Non-periodic: need >=5 to allow cubic; periodic: need >=3
            specs.order_scheme_directional[d] =
                ((periodic_d == 0) ? ((ncd < 5) ? 1 : 2) : ((ncd < 3) ? 1 : 2));
        }

        // Warn if all directions fell back to linear
        bool all_linear = true;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            all_linear &= (specs.order_scheme_directional[d] == 1);
        }
        if (all_linear)
        {
            amrex::Print() << "\nWarning! Number of cells in all directions do "
                              "not qualify for cubic-spline shape functions\n"
                           << "Reverting to linear hat shape functions in all "
                              "directions\n";
        }

        // Ensure no spline box has size==1 in any dimension
        for (int box_index = 0; box_index < ba.size(); ++box_index)
        {
            const auto sz = ba[box_index].size();
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                if (sz[d] == 1 && specs.order_scheme_directional[d] == 3)
                {
                    amrex::Abort("Error: Box cannot be of size = 1 when using "
                                 "spline shape functions. "
                                 "Please adjust max_grid_size so all boxes "
                                 "have size > 1.");
                }
            }
        }
    }
    else if (specs.order_scheme == 3)
    {
        ng_cells_nodaldata = 3;

        // Set directional order-scheme based on periodicity and grid size
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            const int ncd = specs.ncells[d];
            const int periodic_d = specs.periodic[d];
            // Non-periodic: need >=5 to allow cubic; periodic: need >=3
            specs.order_scheme_directional[d] =
                ((periodic_d == 0) ? ((ncd < 5) ? 1 : 3) : ((ncd < 3) ? 1 : 3));
        }

        // Warn if all directions fell back to linear
        bool all_linear = true;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            all_linear &= (specs.order_scheme_directional[d] == 1);
        }
        if (all_linear)
        {
            amrex::Print() << "\nWarning! Number of cells in all directions do "
                              "not qualify for cubic-spline shape functions\n"
                           << "Reverting to linear hat shape functions in all "
                              "directions\n";
        }

        // Ensure no spline box has size==1 in any dimension
        for (int box_index = 0; box_index < ba.size(); ++box_index)
        {
            const auto sz = ba[box_index].size();
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                if (sz[d] == 1 && specs.order_scheme_directional[d] == 3)
                {
                    amrex::Abort("Error: Box cannot be of size = 1 when using "
                                 "spline shape functions. "
                                 "Please adjust max_grid_size so all boxes "
                                 "have size > 1.");
                }
            }
        }
    }
    else
    {
        amrex::Abort("Order scheme not implemented yet (use 1 or 3).");
    }

    // Nodal layout convert (dimension-aware IntVect of ones)
#if (AMREX_SPACEDIM == 1)
    const IntVect nodal_iv(1);
#elif (AMREX_SPACEDIM == 2)
    const IntVect nodal_iv(1, 1);
#else
    const IntVect nodal_iv(1, 1, 1);
#endif
    const BoxArray nodeba = amrex::convert(ba, nodal_iv);

    // Nodal data (NUM_STATES components, ng_cells_nodaldata ghost)
    nodaldata.define(nodeba, dm, NUM_STATES, ng_cells_nodaldata);
    nodaldata.setVal(0.0, ng_cells_nodaldata);

    // Level-set geometry and data (refined domain)
    Box dom_levset = geom.Domain();
    dom_levset.refine(specs.levset_gridratio);
    // Note: real_box/periodicity for levset geom is inherited from parent
    // domain in many setups; if you need distinct real_box or periodic flags,
    // pass them explicitly via another define overload.
    geom_levset.define(dom_levset);

    if (specs.levset_output)
    {
        BoxArray phase_ba = ba;
        phase_ba.refine(specs.levset_gridratio);

        const int ng_phase = 3;
        levset_data.define(phase_ba, dm, /*ncomp=*/1, ng_phase);
        levset_data.setVal(0.0, ng_phase);
    }

    PrintMessage("", print_length, false);
}

/**
 * @brief Creates all required output directories for the simulation.
 *
 * Ensures that folders for particle output, grid output, checkpoints,
 * ASCII diagnostics, and level‑set output exist before the simulation begins.
 *
 * @param[in] specs  Simulation specification structure containing folder paths.
 *
 * @return None.
 */

void Create_Output_Directories(MPMspecs &specs)
{

    amrex::UtilCreateDirectory(specs.particle_output_folder, 0755);
    amrex::UtilCreateDirectory(specs.grid_output_folder, 0755);
    amrex::UtilCreateDirectory(specs.checkpoint_output_folder, 0755);
    amrex::Print() << "\nCreating folder " << specs.checkpoint_output_folder;
    amrex::UtilCreateDirectory(specs.ascii_output_folder, 0755);
    amrex::Print() << "\nCreating ascii folder " << specs.ascii_output_folder;
    if (specs.levset_output)
    {
        amrex::UtilCreateDirectory(specs.levset_output_folder, 0755);
    }
    if (specs.print_diagnostics)
    {
        amrex::UtilCreateDirectory(specs.diagnostic_output_folder, 0755);
    }
}

/**
 * @brief Computes initial internal forces, strain rates, stresses, and
 * phase‑field values.
 *
 * Performs:
 *  - Momentum deposition from particles to grid.
 *  - Backup of nodal velocities.
 *  - Grid‑to‑particle interpolation of velocity.
 *  - Constitutive model update at particles.
 *  - Optional level‑set / phase‑field update.
 *  - Optional thermal initialization (if USE_TEMP).
 *
 * @param[in]     specs        Simulation specifications.
 * @param[in,out] mpm_pc       Particle container.
 * @param[in,out] nodaldata    Nodal MultiFab storing grid quantities.
 * @param[in,out] levset_data  Level‑set MultiFab (if enabled).
 *
 * @return None.
 */

void Initialise_Internal_Forces(MPMspecs &specs,
                                MPMParticleContainer &mpm_pc,
                                amrex::MultiFab &nodaldata,
                                amrex::MultiFab &levset_data)
{

    // Momentum deposition and initial stress/strainrate
    {
        std::string msg = "\n Calculating initial strainrates and stresses";
        PrintMessage(msg, print_length, true);
        amrex::Real dt = 0.0;

        mpm_pc.deposit_onto_grid_momentum(
            nodaldata, specs.gravity, specs.external_loads_present,
            specs.force_slab_lo, specs.force_slab_hi, specs.extforce,
            /*update mass*/ 1,
            /*do_reset=*/1,
            /*do_average=*/1, specs.mass_tolerance,
            specs.order_scheme_directional, specs.periodic);

        backup_current_velocity(nodaldata);

        // Interpolate grid -> particles
        mpm_pc.interpolate_from_grid(nodaldata,
                                     /*momentum_comp=*/0,
                                     /*mass_comp=*/1,
                                     specs.order_scheme_directional,
                                     specs.periodic, specs.alpha_pic_flip, dt);

        // Constitutive update
        mpm_pc.apply_constitutive_model(dt, specs.applied_strainrate);

        PrintMessage(msg, print_length, false);
    }

    // Phase-field / levelset update
    {
        std::string msg = "\n Updating phase field";
        PrintMessage(msg, print_length, true);

        if (specs.levset_output)
        {
            mpm_pc.update_phase_field(levset_data, specs.levset_gridratio,
                                      specs.levset_smoothfactor);
        }

        PrintMessage(msg, print_length, false);
    }

#if USE_TEMP
    {
        std::string msg = "\n Calculating initial heat flux";
        PrintMessage(msg, print_length, true);

        // Dimension-aware thermal BC ranges
        amrex::Array<amrex::Real, AMREX_SPACEDIM> temp_lo;
        amrex::Array<amrex::Real, AMREX_SPACEDIM> temp_hi;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            temp_lo[d] = specs.bclo_tempval[d];
            temp_hi[d] = specs.bchi_tempval[d];
        }

        mpm_pc.deposit_onto_grid_temperature(
            nodaldata, 1, 1, 0, specs.mass_tolerance,
            specs.order_scheme_directional, specs.periodic);

        backup_current_temperature(nodaldata);

        // Apply nodal boundary conditions (ensure correct Geometry is passed)
        const Geometry &geom = mpm_pc.Geom(0);
        nodal_bcs_temperature(geom, nodaldata, specs.bclo.data(),
                              specs.bchi.data(), temp_lo.data(),
                              temp_hi.data());
        store_delta_temperature(nodaldata);

        // Interpolate temperature grid -> particles
        mpm_pc.interpolate_from_grid_temperature(nodaldata, 1, 1,
                                                 specs.order_scheme_directional,
                                                 specs.periodic, 1.0);

        PrintMessage(msg, print_length, false);
    }
#endif
}

/**
 * @brief Initializes material points either from a checkpoint, a particle file,
 * or autogeneration.
 *
 * Three modes:
 *  - Restart from checkpoint file.
 *  - Read particle data from an external file.
 *  - Autogenerate particles inside a bounding box.
 *
 * Also:
 *  - Redistributes particles across tiles.
 *  - Fills neighbor lists.
 *  - Removes particles inside embedded boundaries (if EB active).
 *
 * @param[in]     specs        Simulation specifications.
 * @param[in,out] mpm_pc       Particle container.
 * @param[out]    steps        Initial step index.
 * @param[out]    time         Initial simulation time.
 * @param[out]    output_it    Output iteration counter.
 *
 * @return None.
 */

void Initialise_Material_Points(MPMspecs &specs,
                                MPMParticleContainer &mpm_pc,
                                int &steps,
                                amrex::Real &time,
                                int &output_it)
{
    if (!specs.restart_checkfile.empty())
    {
        std::string msg =
            "\n Acquiring particle data (restarting from checkpoint file)";
        PrintMessage(msg, print_length, true);
        mpm_pc.readCheckpointFile(specs.restart_checkfile, steps, time,
                                  output_it);
        PrintMessage(msg, print_length, true);
    }
    else if (!specs.use_autogen)
    {
        std::string msg =
            "\n Acquiring particle data (Reading from particle file)";
        PrintMessage(msg, print_length, true);

        // dimension‑aware InitParticles (file‑based)
        if (specs.particlefilename.size() >= 3 &&
            specs.particlefilename.compare(specs.particlefilename.size() - 3, 3,
                                           ".h5") == 0)
        {
#ifdef AMREX_USE_HDF5
            mpm_pc.InitParticlesFromHDF5(
                specs.particlefilename, specs.total_mass, specs.total_vol,
                specs.total_rigid_mass, specs.no_of_rigidbodies_present,
                specs.ifrigidnodespresent);
#else
            amrex::Abort("ExaGOOP was built without HDF5 support.");
#endif
        }
        else
        {
            mpm_pc.InitParticles(specs.particlefilename, specs.total_mass,
                                 specs.total_vol, specs.total_rigid_mass,
                                 specs.no_of_rigidbodies_present,
                                 specs.ifrigidnodespresent);
        }

        PrintMessage(msg, print_length, false);
        mpm_pc.RedistributeLocal();
        mpm_pc.fillNeighbors();

        if (specs.no_of_rigidbodies_present != numrigidbodies)
        {
            amrex::Print() << "\n specs.no_of_rigidbodies_present= "
                           << specs.no_of_rigidbodies_present << " "
                           << numrigidbodies;
            // amrex::Abort("Mismatch in rigid body count between file and
            // constants.H");
        }
    }
    else
    {
        std::string msg = "\n Acquiring particle data (using autogen)";
        PrintMessage(msg, print_length, true);

        // dimension‑aware InitParticles (autogen)
        mpm_pc.InitParticles(
            specs.autogen_mincoords.data(), specs.autogen_maxcoords.data(),
            specs.autogen_vel.data(), specs.autogen_dens,
            specs.autogen_constmodel, specs.autogen_E, specs.autogen_nu,
            specs.autogen_bulkmod, specs.autogen_Gama_pres, specs.autogen_visc,
            specs.autogen_multi_part_per_cell, specs.total_mass,
            specs.total_vol);

        PrintMessage(msg, print_length, false);
    }

    // remove particles inside EB if levelset geometry is active
#if USE_EB
    if (mpm_ebtools::using_levelset_geometry)
    {
        mpm_pc.removeParticlesInsideEB();
    }
#endif

    mpm_pc.RedistributeLocal();
    mpm_pc.fillNeighbors();
}

/**
 * @brief Initializes particles by reading from an ASCII particle file.
 *
 * Reads:
 *  - Phase (MPM or rigid)
 *  - Rigid body ID
 *  - Position
 *  - Radius and density
 *  - Velocity components
 *  - Constitutive model and material parameters
 *  - Optional thermal fields
 *
 * Computes:
 *  - Volume (sphere assumption)
 *  - Mass
 *  - Initial deformation gradient (identity)
 *  - Zero strain, stress, strain‑rate
 *
 * @param[in]  filename              Path to particle file.
 * @param[out] total_mass            Total mass of MPM particles.
 * @param[out] total_vol             Total volume of MPM particles.
 * @param[out] total_rigid_mass      Total mass of rigid particles (ID=0).
 * @param[out] num_of_rigid_bodies   Number of rigid bodies encountered.
 * @param[out] ifrigidnodespresent   Flag indicating presence of rigid nodes.
 *
 * @return None.
 */

/*
void MPMParticleContainer::InitParticles(const std::string &filename,
                                         amrex::Real &total_mass,
                                         amrex::Real &total_vol,
                                         amrex::Real &total_rigid_mass,
                                         int &num_of_rigid_bodies,
                                         int &ifrigidnodespresent)
{
    // only read the file on the IO proc
    if (ParallelDescriptor::IOProcessor())
    {
        std::ifstream ifs(filename);
        if (!ifs.good())
        {
            amrex::FileOpenFailed(filename);
        }

        int np = -1;

        // ------------------------------------------------------------
        // 1. Read "dim: <value>"
        // ------------------------------------------------------------
        std::string label;
        int file_dim = -1;

        ifs >> label >> file_dim;   // label = "dim:", file_dim = 1/2/3

        if (label != "dim:") {
            amrex::Abort("mpm_particles.dat: Expected 'dim:' at line 1");
        }

        if (file_dim != AMREX_SPACEDIM) {
            amrex::Print() << "ERROR: Particle file dimension = " << file_dim <<
"\n"
                           << "       AMREX_SPACEDIM        = " <<
AMREX_SPACEDIM << "\n"; amrex::Abort("Dimension mismatch between particle file
and ExaGOOP build");
        }

        // ------------------------------------------------------------
        // 2. Read "number_of_material_points: <value>"
        // ------------------------------------------------------------
        std::string label2;


        ifs >> label2 >> np;  // label2 = "number_of_material_points:"

        if (label2 != "number_of_material_points:") {
            amrex::Abort("mpm_particles.dat: Expected
'number_of_material_points:' at line 2");
        }

        if (np <= 0) {
            amrex::Abort("mpm_particles.dat: Invalid
number_of_material_points");
        }

        // ------------------------------------------------------------
        // 3. Skip the header line beginning with '#'
        // ------------------------------------------------------------
        std::string header_line;
        std::getline(ifs, header_line); // finish line 2
        std::getline(ifs, header_line); // read line 3 (column names)

        // header_line should start with '#'
        if (header_line.empty() || header_line[0] != '#') {
            amrex::Abort("mpm_particles.dat: Expected header line beginning with
'#'");
        }


        const int lev = 0, grid = 0, tile = 0;
        const int tot_rig_body_tmp = 10;
        int rigid_bodies_read_so_far[tot_rig_body_tmp] = {-1};
        int index_rigid_body_read_so_far = 0;

        total_mass = 0.0;
        total_vol = 0.0;
        total_rigid_mass = 0.0;

        auto &particle_tile = DefineAndReturnParticleTile(lev, grid, tile);
        Gpu::HostVector<ParticleType> host_particles;

        for (int i = 0; i < np; ++i)
        {
            ParticleType p;

            // id/cpu
            p.id() = ParticleType::NextID();
            p.cpu() = ParallelDescriptor::MyProc();

            // phase: 0 = mpm, 1 = rigid
            ifs >> p.idata(intData::phase);

            if (p.idata(intData::phase) == 1)
            {
                ifrigidnodespresent = 1;
                ifs >> p.idata(intData::rigid_body_id);

                bool body_present = false;
                for (int j = 0; j < index_rigid_body_read_so_far; ++j)
                {
                    body_present |= (rigid_bodies_read_so_far[j] ==
                                     p.idata(intData::rigid_body_id));
                }
                if (!body_present &&
                    index_rigid_body_read_so_far < tot_rig_body_tmp)
                {
                    rigid_bodies_read_so_far[index_rigid_body_read_so_far++] =
                        p.idata(intData::rigid_body_id);
                }
            }
            else
            {
                p.idata(intData::rigid_body_id) = -1;
            }

            // positions (dimension‑aware)
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                p.pos(d) = 0.0;
            }
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                amrex::Real coord;
                ifs >> coord;
                p.pos(d) = coord;
            }
            // radius & density
            ifs >> p.rdata(realData::radius);
            ifs >> p.rdata(realData::density);

            // velocities (dimension‑aware)
            for (int d = 0; d < 3; ++d)
            {

                p.rdata(realData::xvel + d) = 0.0;
                p.rdata(realData::xvel_prime + d) = 0.0;
            }

            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                amrex::Real v;
                ifs >> v;
                p.rdata(realData::xvel + d) = v;
            }

            // constitutive model
            ifs >> p.idata(intData::constitutive_model);
            if (p.idata(intData::constitutive_model) == 0)
            {
                // Elastic solid
                ifs >> p.rdata(realData::E);
                ifs >> p.rdata(realData::nu);
                p.rdata(realData::Bulk_modulus) = 0.0;
                p.rdata(realData::Gama_pressure) = 0.0;
                p.rdata(realData::Dynamic_viscosity) = 0.0;
            }
            else if (p.idata(intData::constitutive_model) == 1)
            {
                // Fluid‑like (bulk/Gamma/viscosity provided)
                p.rdata(realData::E) = 0.0;
                p.rdata(realData::nu) = 0.0;
                ifs >> p.rdata(realData::Bulk_modulus);
                ifs >> p.rdata(realData::Gama_pressure);
                ifs >> p.rdata(realData::Dynamic_viscosity);
            }
            else
            {
                amrex::Abort("\n\tIncorrect constitutive model. Please check "
                             "your particle file");
            }

#if USE_TEMP
            // thermal fields
            ifs >> p.rdata(realData::temperature);
            ifs >> p.rdata(realData::specific_heat);
            ifs >> p.rdata(realData::thermal_conductivity);
            ifs >> p.rdata(realData::heat_source);
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                p.rdata(realData::heat_flux + d) = 0.0;
            }
#endif

            // volume (sphere assumption) and mass
            p.rdata(realData::volume) =
                fourbythree * PI * std::pow(p.rdata(realData::radius), three);
            p.rdata(realData::mass) =
                p.rdata(realData::density) * p.rdata(realData::volume);

            if (p.idata(intData::phase) == 0)
            {
                total_mass += p.rdata(realData::mass);
                total_vol += p.rdata(realData::volume);
            }
            else if (p.idata(intData::phase) == 1 &&
                     p.idata(intData::rigid_body_id) == 0)
            {
                total_rigid_mass += p.rdata(realData::mass);
            }

            // state init
            p.rdata(realData::jacobian) = 1.0;
            p.rdata(realData::vol_init) = p.rdata(realData::volume);
            p.rdata(realData::pressure) = 0.0;

            // deformation gradient (identity in active dims)
            for (int comp = 0; comp < NCOMP_FULLTENSOR; ++comp)
            {
                p.rdata(realData::deformation_gradient + comp) = 0.0;
            }
            // Map (d,d) to linear index; assumes row‑major 3x3 storage in
            // NCOMP_FULLTENSOR
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                // indices for 3x3: (0,0)=0, (1,1)=4, (2,2)=8
                const int diag_idx = d * AMREX_SPACEDIM + d;
                p.rdata(realData::deformation_gradient + diag_idx) = 1.0;
            }

            for (int comp = 0; comp < NCOMP_TENSOR; ++comp)
            {
                p.rdata(realData::strainrate + comp) = shunya;
                p.rdata(realData::strain + comp) = shunya;
                p.rdata(realData::stress + comp) = shunya;
            }

            host_particles.push_back(p);

            if (!ifs.good())
            {
                amrex::Abort("Error initializing particles from Ascii file.\n");
            }
        }

        num_of_rigid_bodies = index_rigid_body_read_so_far;

        auto old_size = particle_tile.GetArrayOfStructs().size();
        particle_tile.resize(old_size + host_particles.size());
        Gpu::copy(Gpu::hostToDevice, host_particles.begin(),
                  host_particles.end(),
                  particle_tile.GetArrayOfStructs().begin() + old_size);
    }
    Redistribute();
}
*/

#ifdef AMREX_USE_HDF5

void MPMParticleContainer::InitParticlesFromHDF5(const std::string &filename,
                                                 amrex::Real &total_mass,
                                                 amrex::Real &total_vol,
                                                 amrex::Real &total_rigid_mass,
                                                 int &num_of_rigid_bodies,
                                                 int &ifrigidnodespresent)
{
    BL_PROFILE("ReadHDF5Particles");

    // ------------------------------------------------------------
    // Open file (serial HDF5 on macOS)
    // ------------------------------------------------------------
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    // No H5Pset_fapl_mpio here: Homebrew HDF5 is serial-only

    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist_id);
    H5Pclose(plist_id);

    if (file_id < 0)
        amrex::Abort("ERROR: Could not open HDF5 particle file");

    // ------------------------------------------------------------
    // Read dimension + number of particles
    // ------------------------------------------------------------
    int dim;
    {
        hid_t dset = H5Dopen(file_id, "dim", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dim);
        H5Dclose(dset);
    }

    long npart;
    {
        hid_t dset = H5Dopen(file_id, "number_of_material_points", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &npart);
        H5Dclose(dset);
    }

    if (ParallelDescriptor::IOProcessor())
        amrex::Print() << "Reading " << npart << " particles (dim=" << dim
                       << ")\n";

    // ------------------------------------------------------------
    // Helper to read a dataset into a Vector<Real>
    // ------------------------------------------------------------
    auto read_dset =
        [&](const std::string &name, amrex::Vector<amrex::Real> &vec)
    {
        hid_t dset = H5Dopen(file_id, name.c_str(), H5P_DEFAULT);
        if (dset < 0)
            amrex::Abort("Missing dataset: " + name);

        hid_t space = H5Dget_space(dset);
        hsize_t dims[1];
        H5Sget_simple_extent_dims(space, dims, nullptr);
        vec.resize(dims[0]);

        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                vec.data());

        H5Sclose(space);
        H5Dclose(dset);
    };

    // ------------------------------------------------------------
    // Read mandatory datasets
    // ------------------------------------------------------------
    amrex::Vector<amrex::Real> x, y, z, vx, vy, vz, radius, density, cm_id;

    read_dset("x", x);
    if (dim >= 2)
        read_dset("y", y);
    if (dim == 3)
        read_dset("z", z);

    read_dset("vx", vx);
    if (dim >= 2)
        read_dset("vy", vy);
    if (dim == 3)
        read_dset("vz", vz);

    read_dset("radius", radius);
    read_dset("density", density);
    read_dset("cm_id", cm_id);

    // ------------------------------------------------------------
    // Read any extra constitutive model fields dynamically
    // ------------------------------------------------------------
    std::vector<std::string> extra_fields;

    {
        hid_t root = H5Gopen(file_id, "/", H5P_DEFAULT);
        hsize_t nobj;
        H5Gget_num_objs(root, &nobj);

        for (hsize_t i = 0; i < nobj; ++i)
        {
            char name[256];
            H5Gget_objname_by_idx(root, i, name, sizeof(name));

            std::string s(name);
            if (s != "x" && s != "y" && s != "z" && s != "vx" && s != "vy" &&
                s != "vz" && s != "radius" && s != "density" && s != "cm_id" &&
                s != "dim" && s != "number_of_material_points")
            {
                extra_fields.push_back(s);
            }
        }
        H5Gclose(root);
    }

    std::map<std::string, amrex::Vector<amrex::Real>> extra_data;
    for (auto &f : extra_fields)
    {
        read_dset(f, extra_data[f]);
    }

    H5Fclose(file_id);

    // ------------------------------------------------------------
    // Fill AMReX ParticleContainer
    // ------------------------------------------------------------
    clearParticles();

    // Level 0 particle tiles: map< (grid, tile), ParticleTile >
    auto &levelmap = this->GetParticles(0);

    // Create (0,0) tile if it doesn't exist yet
    std::pair<int, int> index{0, 0};
    auto &tile =
        levelmap[index]; // operator[] default-constructs a ParticleTile

    auto &aos = tile.GetArrayOfStructs();
    auto &soa = tile.GetStructOfArrays();

    aos.resize(npart);
    soa.resize(npart);

    using PType = MPMParticleContainer::ParticleType;

    for (long i = 0; i < npart; ++i)
    {
        PType &p = aos[i];

        p.id() = PType::NextID();
        p.cpu() = ParallelDescriptor::MyProc();

        p.pos(0) = x[i];
        if (dim >= 2)
            p.pos(1) = y[i];
        if (dim == 3)
            p.pos(2) = z[i];

        p.rdata(realData::radius) = radius[i];
        p.rdata(realData::density) = density[i];

        for (int d = 0; d < 3; ++d)
        {
            p.rdata(realData::xvel + d) = 0.0;
            p.rdata(realData::xvel_prime + d) = 0.0;
        }

        p.rdata(realData::xvel + 0) = vx[i];
        if (dim >= 2)
            p.rdata(realData::xvel + 1) = vy[i];
        if (dim == 3)
            p.rdata(realData::xvel + 2) = vz[i];

        p.idata(intData::constitutive_model) = cm_id[i];

        if (cm_id[i] == 0)
        {
            p.rdata(realData::E) = extra_data.at("E")[i];
            p.rdata(realData::nu) = extra_data.at("nu")[i];
        }
        else if (cm_id[i] == 1)
        {
            p.rdata(realData::Bulk_modulus) = extra_data.at("Bulk_modulus")[i];
            p.rdata(realData::Gama_pressure) =
                extra_data.at("Gama_pressure")[i];
            p.rdata(realData::Dynamic_viscosity) =
                extra_data.at("Dynamic_viscosity")[i];
        }
#if USE_TEMP
        p.rdata(realData::temperature) = extra_data.at("T")[i];
        p.rdata(realData::specific_heat) = extra_data.at("spheat")[i];
        p.rdata(realData::thermal_conductivity) = extra_data.at("thermcond")[i];
        p.rdata(realData::heat_source) = extra_data.at("heatsrc")[i];
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
            p.rdata(realData::heat_flux + d) = 0.0;
#endif
        // volume & mass
        p.rdata(realData::volume) =
            fourbythree * PI * std::pow(p.rdata(realData::radius), three);
        p.rdata(realData::mass) =
            p.rdata(realData::density) * p.rdata(realData::volume);

        if (p.idata(intData::phase) == 0)
        {
            total_mass += p.rdata(realData::mass);
            total_vol += p.rdata(realData::volume);
        }
        else if (p.idata(intData::phase) == 1 &&
                 p.idata(intData::rigid_body_id) == 0)
        {
            total_rigid_mass += p.rdata(realData::mass);
        }

        p.rdata(realData::jacobian) = 1.0;
        p.rdata(realData::vol_init) = p.rdata(realData::volume);
        p.rdata(realData::pressure) = 0.0;

        // deformation gradient init
        for (int comp = 0; comp < NCOMP_FULLTENSOR; ++comp)
            p.rdata(realData::deformation_gradient + comp) = 0.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            int diag = d * AMREX_SPACEDIM + d;
            p.rdata(realData::deformation_gradient + diag) = 1.0;
        }

        // strain/stress init
        for (int comp = 0; comp < NCOMP_TENSOR; ++comp)
        {
            p.rdata(realData::strainrate + comp) = 0.0;
            p.rdata(realData::strain + comp) = 0.0;
            p.rdata(realData::stress + comp) = 0.0;
        }
    }

    Redistribute();
}
#endif

void MPMParticleContainer::InitParticles(const std::string &filename,
                                         amrex::Real &total_mass,
                                         amrex::Real &total_vol,
                                         amrex::Real &total_rigid_mass,
                                         int &num_of_rigid_bodies,
                                         int &ifrigidnodespresent)
{
    const int CHUNK_SIZE = 100000; // tune as needed

    if (ParallelDescriptor::IOProcessor())
    {
        std::ifstream ifs(filename);
        if (!ifs.good())
        {
            amrex::FileOpenFailed(filename);
        }

        long np = -1;

        // ------------------------------------------------------------
        // 1. Read "dim: <value>"
        // ------------------------------------------------------------
        std::string label;
        int file_dim = -1;

        ifs >> label >> file_dim; // label = "dim:", file_dim = 1/2/3

        if (label != "dim:")
        {
            amrex::Abort("mpm_particles.dat: Expected 'dim:' at line 1");
        }

        if (file_dim != AMREX_SPACEDIM)
        {
            amrex::Print() << "ERROR: Particle file dimension = " << file_dim
                           << "\n"
                           << "       AMREX_SPACEDIM        = "
                           << AMREX_SPACEDIM << "\n";
            amrex::Abort(
                "Dimension mismatch between particle file and ExaGOOP build");
        }

        // ------------------------------------------------------------
        // 2. Read "number_of_material_points: <value>"
        // ------------------------------------------------------------
        std::string label2;

        ifs >> label2 >> np; // label2 = "number_of_material_points:"

        if (label2 != "number_of_material_points:")
        {
            amrex::Abort("mpm_particles.dat: Expected "
                         "'number_of_material_points:' at line 2");
        }

        if (np <= 0)
        {
            amrex::Abort(
                "mpm_particles.dat: Invalid number_of_material_points");
        }

        // ------------------------------------------------------------
        // 3. Skip the header line beginning with '#'
        // ------------------------------------------------------------
        std::string header_line;
        std::getline(ifs, header_line); // finish line 2
        std::getline(ifs, header_line); // read line 3 (column names)

        // header_line should start with '#'
        if (header_line.empty() || header_line[0] != '#')
        {
            amrex::Abort(
                "mpm_particles.dat: Expected header line beginning with '#'");
        }

        total_mass = 0.0;
        total_vol = 0.0;
        total_rigid_mass = 0.0;
        ifrigidnodespresent = 0;
        num_of_rigid_bodies = 0;

        const int lev = 0, grid = 0, tile = 0;
        auto &ptile = DefineAndReturnParticleTile(lev, grid, tile);
        auto &aos = ptile.GetArrayOfStructs();

        Gpu::HostVector<ParticleType> host_particles;
        host_particles.reserve(CHUNK_SIZE);

        int rigid_bodies_seen[32] = {-1};
        int rigid_count = 0;

        auto &particle_tile = DefineAndReturnParticleTile(lev, grid, tile);

        for (int i = 0; i < np; ++i)
        {
            ParticleType p;

            // id/cpu
            p.id() = ParticleType::NextID();
            p.cpu() = ParallelDescriptor::MyProc();

            // phase
            ifs >> p.idata(intData::phase);

            if (p.idata(intData::phase) == 1)
            {
                ifrigidnodespresent = 1;
                ifs >> p.idata(intData::rigid_body_id);

                bool found = false;
                for (int j = 0; j < rigid_count; ++j)
                    found |= (rigid_bodies_seen[j] ==
                              p.idata(intData::rigid_body_id));

                if (!found && rigid_count < 32)
                    rigid_bodies_seen[rigid_count++] =
                        p.idata(intData::rigid_body_id);
            }
            else
            {
                p.idata(intData::rigid_body_id) = -1;
            }

            // positions
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                amrex::Real coord;
                ifs >> coord;
                p.pos(d) = coord;
            }

            // radius & density
            ifs >> p.rdata(realData::radius);
            ifs >> p.rdata(realData::density);

            // velocities
            for (int d = 0; d < 3; ++d)
            {
                p.rdata(realData::xvel + d) = 0.0;
                p.rdata(realData::xvel_prime + d) = 0.0;
            }
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                amrex::Real v;
                ifs >> v;
                p.rdata(realData::xvel + d) = v;
            }

            // constitutive model
            ifs >> p.idata(intData::constitutive_model);
            if (p.idata(intData::constitutive_model) == 0)
            {
                ifs >> p.rdata(realData::E);
                ifs >> p.rdata(realData::nu);
                p.rdata(realData::Bulk_modulus) = 0.0;
                p.rdata(realData::Gama_pressure) = 0.0;
                p.rdata(realData::Dynamic_viscosity) = 0.0;
            }
            else if (p.idata(intData::constitutive_model) == 1)
            {
                p.rdata(realData::E) = 0.0;
                p.rdata(realData::nu) = 0.0;
                ifs >> p.rdata(realData::Bulk_modulus);
                ifs >> p.rdata(realData::Gama_pressure);
                ifs >> p.rdata(realData::Dynamic_viscosity);
            }
            else
            {
                amrex::Abort("Incorrect constitutive model");
            }

#if USE_TEMP
            ifs >> p.rdata(realData::temperature);
            ifs >> p.rdata(realData::specific_heat);
            ifs >> p.rdata(realData::thermal_conductivity);
            ifs >> p.rdata(realData::heat_source);
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                p.rdata(realData::heat_flux + d) = 0.0;
#endif

            // volume & mass
            p.rdata(realData::volume) =
                fourbythree * PI * std::pow(p.rdata(realData::radius), three);
            p.rdata(realData::mass) =
                p.rdata(realData::density) * p.rdata(realData::volume);

            if (p.idata(intData::phase) == 0)
            {
                total_mass += p.rdata(realData::mass);
                total_vol += p.rdata(realData::volume);
            }
            else if (p.idata(intData::phase) == 1 &&
                     p.idata(intData::rigid_body_id) == 0)
            {
                total_rigid_mass += p.rdata(realData::mass);
            }

            p.rdata(realData::jacobian) = 1.0;
            p.rdata(realData::vol_init) = p.rdata(realData::volume);
            p.rdata(realData::pressure) = 0.0;

            // deformation gradient init
            for (int comp = 0; comp < NCOMP_FULLTENSOR; ++comp)
                p.rdata(realData::deformation_gradient + comp) = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                int diag = d * AMREX_SPACEDIM + d;
                p.rdata(realData::deformation_gradient + diag) = 1.0;
            }

            // strain/stress init
            for (int comp = 0; comp < NCOMP_TENSOR; ++comp)
            {
                p.rdata(realData::strainrate + comp) = 0.0;
                p.rdata(realData::strain + comp) = 0.0;
                p.rdata(realData::stress + comp) = 0.0;
            }

            host_particles.push_back(p);

            // If chunk is full → insert + redistribute
            if ((int)host_particles.size() == CHUNK_SIZE)
            {
                auto old_size = aos.size();
                aos.resize(old_size + host_particles.size());

                // host-to-host copy
                std::copy(host_particles.begin(), host_particles.end(),
                          aos.begin() + old_size);

                host_particles.clear();

                // redistribute immediately
                Redistribute();
            }
        }

        // Final partial chunk

        if (!host_particles.empty())
        {
            auto old_size = aos.size();
            aos.resize(old_size + host_particles.size());

            std::copy(host_particles.begin(), host_particles.end(),
                      aos.begin() + old_size);

            host_particles.clear();

            Redistribute();
        }

        num_of_rigid_bodies = rigid_count;
        /*auto old_size = particle_tile.GetArrayOfStructs().size();
        particle_tile.resize(old_size + host_particles.size());
        Gpu::copy(Gpu::hostToDevice, host_particles.begin(),
                  host_particles.end(),
                  particle_tile.GetArrayOfStructs().begin() + old_size);*/
    }
    else
    {
        // Non-IO ranks still need to participate in Redistribute()
        Redistribute();
    }
}

/**
 * @brief Autogenerates particles inside a user‑specified bounding box.
 *
 * For each cell:
 *  - Places either one particle at the cell center, or
 *  - 2^D particles at sub‑cell centers (if do_multi_part_per_cell enabled)
 *
 * Assigns:
 *  - Position
 *  - Velocity
 *  - Density
 *  - Material properties
 *  - Volume and mass
 *  - Zero stress/strain fields
 *
 * @param[in]  mincoords   Minimum coordinates of autogen region.
 * @param[in]  maxcoords   Maximum coordinates of autogen region.
 * @param[in]  vel         Initial velocity vector.
 * @param[in]  dens        Density.
 * @param[in]  constmodel  Constitutive model ID.
 * @param[in]  E           Young’s modulus.
 * @param[in]  nu          Poisson ratio.
 * @param[in]  bulkmod     Bulk modulus.
 * @param[in]  Gama_pres   Gamma pressure parameter.
 * @param[in]  visc        Dynamic viscosity.
 * @param[in]  do_multi_part_per_cell  Flag for multi‑particle seeding.
 * @param[out] total_mass  Total mass of generated particles.
 * @param[out] total_vol   Total volume of generated particles.
 *
 * @return None.
 */
void MPMParticleContainer::InitParticles(amrex::Real mincoords[AMREX_SPACEDIM],
                                         amrex::Real maxcoords[AMREX_SPACEDIM],
                                         amrex::Real vel[AMREX_SPACEDIM],
                                         amrex::Real dens,
                                         int constmodel,
                                         amrex::Real E,
                                         amrex::Real nu,
                                         amrex::Real bulkmod,
                                         amrex::Real Gama_pres,
                                         amrex::Real visc,
                                         int do_multi_part_per_cell,
                                         amrex::Real &total_mass,
                                         amrex::Real &total_vol)
{
    const int lev = 0;
    const auto dxA = Geom(lev).CellSizeArray(); // dimension-aware dx
    const auto ploA = Geom(lev).ProbLoArray();  // dimension-aware prob lo

    total_mass = 0.0;
    total_vol = 0.0;

    // Precompute cell volume = product(dx[d])
    amrex::Real cell_vol = 1.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        cell_vol *= dxA[d];
    }

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        const Box &tile_box = mfi.tilebox();
        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();
        auto &particle_tile =
            GetParticles(lev)[std::make_pair(grid_id, tile_id)];

        Gpu::HostVector<ParticleType> host_particles;

        for (IntVect iv = tile_box.smallEnd(); iv <= tile_box.bigEnd();
             tile_box.next(iv))
        {
            if (do_multi_part_per_cell == 0)
            {
                // Cell center position
                amrex::Real coords[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    coords[d] = ploA[d] + (iv[d] + HALF_CONST) * dxA[d];
                }

                // Inside user box?
                bool inside = true;
                for (int d = 0; d < AMREX_SPACEDIM && inside; ++d)
                {
                    inside = (coords[d] >= mincoords[d] &&
                              coords[d] <= maxcoords[d]);
                }

                if (inside)
                {
                    ParticleType p = generate_particle(
                        coords, vel, dens, cell_vol, constmodel, E, nu, bulkmod,
                        Gama_pres, visc);

                    total_mass += p.rdata(realData::mass);
                    total_vol += p.rdata(realData::volume);
                    host_particles.push_back(p);
                }
            }
            else
            {
                // Lower corner of the cell
                amrex::Real base[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    base[d] = ploA[d] + iv[d] * dxA[d];
                }

                // Place 2^dim particles per cell at sub‑cell centers
                const int corners = 1 << AMREX_SPACEDIM;
                for (int c = 0; c < corners; ++c)
                {
                    amrex::Real coords[AMREX_SPACEDIM];
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        int bit = (c >> d) & 1; // 0 or 1 per dimension
                        amrex::Real offset =
                            (bit + HALF_CONST) * HALF_CONST * dxA[d];
                        coords[d] = base[d] + offset;
                    }

                    bool inside = true;
                    for (int d = 0; d < AMREX_SPACEDIM && inside; ++d)
                    {
                        inside = (coords[d] >= mincoords[d] &&
                                  coords[d] <= maxcoords[d]);
                    }

                    if (inside)
                    {
                        ParticleType p = generate_particle(
                            coords, vel, dens, cell_vol / corners, constmodel,
                            E, nu, bulkmod, Gama_pres, visc);

                        total_mass += p.rdata(realData::mass);
                        total_vol += p.rdata(realData::volume);
                        host_particles.push_back(p);
                    }
                }
            }
        }

        auto old_size = particle_tile.GetArrayOfStructs().size();
        particle_tile.resize(old_size + host_particles.size());
        Gpu::copy(Gpu::hostToDevice, host_particles.begin(),
                  host_particles.end(),
                  particle_tile.GetArrayOfStructs().begin() + old_size);
    }

    // Move particles to correct tiles if necessary
    Redistribute();
}

/**
 * @brief Generates a single particle with given position, velocity, and
 * material properties.
 *
 * Initializes:
 *  - Position
 *  - Density, velocity
 *  - Constitutive model parameters
 *  - Volume and mass
 *  - Jacobian, pressure, initial volume
 *  - Zero stress, strain, and strain‑rate tensors
 *
 * @param[in] coords      Particle position.
 * @param[in] vel         Velocity vector.
 * @param[in] dens        Density.
 * @param[in] vol         Particle volume.
 * @param[in] constmodel  Constitutive model ID.
 * @param[in] E           Young’s modulus.
 * @param[in] nu          Poisson ratio.
 * @param[in] bulkmod     Bulk modulus.
 * @param[in] Gama_pres   Gamma pressure parameter.
 * @param[in] visc        Dynamic viscosity.
 *
 * @return A fully initialized ParticleType object.
 */

MPMParticleContainer::ParticleType
MPMParticleContainer::generate_particle(amrex::Real coords[AMREX_SPACEDIM],
                                        amrex::Real vel[AMREX_SPACEDIM],
                                        amrex::Real dens,
                                        amrex::Real vol,
                                        int constmodel,
                                        amrex::Real E,
                                        amrex::Real nu,
                                        amrex::Real bulkmod,
                                        amrex::Real Gama_pres,
                                        amrex::Real visc)
{
    ParticleType p;
    p.id() = ParticleType::NextID();
    p.cpu() = ParallelDescriptor::MyProc();

    // Position assignment dimension‑aware
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        p.pos(d) = coords[d];
    }

    // Phase and radius
    p.idata(intData::phase) = 0; // no rigid body particles
    p.rdata(realData::radius) = std::pow(three * fourth * vol / PI, 1.0 / 3.0);

    // Density and velocity components
    p.rdata(realData::density) = dens;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        p.rdata(realData::xvel + d) = vel[d];
    }

    // Constitutive model and material properties
    p.idata(intData::constitutive_model) = constmodel;
    p.rdata(realData::E) = E;
    p.rdata(realData::nu) = nu;
    p.rdata(realData::Bulk_modulus) = bulkmod;
    p.rdata(realData::Gama_pressure) = Gama_pres;
    p.rdata(realData::Dynamic_viscosity) = visc;

    // Volume, mass, and state variables
    p.rdata(realData::volume) = vol;
    p.rdata(realData::mass) = dens * vol;
    p.rdata(realData::jacobian) = 1.0;
    p.rdata(realData::pressure) = 0.0;
    p.rdata(realData::vol_init) = 0.0;

    // Initialize tensor components
    for (int comp = 0; comp < NCOMP_TENSOR; ++comp)
    {
        p.rdata(realData::strainrate + comp) = shunya;
        p.rdata(realData::strain + comp) = shunya;
        p.rdata(realData::stress + comp) = shunya;
    }

    return p;
}

/**
 * @brief Prints particle data for debugging (currently disabled).
 *
 * This function is intended to loop over all particles on level 0 and
 * print or inspect their properties. The implementation is currently
 * commented out, but the structure shows how to iterate over tiles and
 * access particle AoS data on CPU or GPU.
 *
 * @note No operations are performed because the body is commented out.
 *
 * @return None.
 */
void MPMParticleContainer::PrintParticleData()
{
    /*
      const int lev = 0;
      auto &plev = GetParticles(lev);

      for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
      {
          int gid = mfi.index();
          int tid = mfi.LocalTileIndex();
          auto index = std::make_pair(gid, tid);

          auto &ptile = plev[index];
          auto &aos = ptile.GetArrayOfStructs();

          int np = aos.numRealParticles();
          ParticleType *pstruct = aos().dataPtr();

          amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept
                             { ParticleType &p = pstruct[i]; });
      }
      */
}

#if USE_EB
/**
 * @brief Removes particles located inside the embedded boundary (EB) region.
 *
 * For each particle:
 *   1. Computes its physical position xp.
 *   2. Evaluates the level‑set value φ(xp) using get_levelset_value().
 *   3. If φ(xp) < TINYVAL, the particle is considered inside the EB and is
 *      marked for deletion by setting p.id() = -1.
 *
 * After marking, the function calls Redistribute() to remove invalid particles
 * and rebalance the particle distribution across tiles and MPI ranks.
 *
 * @param None (operates on the particle container at level 0)
 *
 * @note Requires EB and level‑set geometry to be active. Uses the globally
 *       defined lsphi MultiFab and ls_refinement from mpm_ebtools.
 *
 * @return None.
 */

void MPMParticleContainer::removeParticlesInsideEB()
{
    const int lev = 0;
    const Geometry &geom = Geom(lev);
    auto &plev = GetParticles(lev);
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();

#if USE_EB
    int lsref = mpm_ebtools::ls_refinement;
#endif

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        int gid = mfi.index();
        int tid = mfi.LocalTileIndex();
        auto index = std::make_pair(gid, tid);

        auto &ptile = plev[index];
        auto &aos = ptile.GetArrayOfStructs();

        int np = aos.numRealParticles();
        ParticleType *pstruct = aos().dataPtr();

        amrex::Array4<amrex::Real> lsetarr = mpm_ebtools::lsphi->array(mfi);

        amrex::ParallelFor(np,
                           [=] AMREX_GPU_DEVICE(int i) noexcept
                           {
                               ParticleType &p = pstruct[i];

                               // Build position array dimension‑aware
                               amrex::Real xp[AMREX_SPACEDIM];
                               for (int d = 0; d < AMREX_SPACEDIM; ++d)
                               {
                                   xp[d] = p.pos(d);
                               }

                               amrex::Real lsval = get_levelset_value(
                                   lsetarr, plo, dx, xp, lsref);

                               if (lsval < TINYVAL)
                               {
                                   p.id() = -1; // mark particle for removal
                               }
                           });
    }

    Redistribute();
}
#endif
