// clang-format off
#include <mpm_particle_container.H>
#include <constants.H>
#include <mpm_eb.H>
#include <aesthetics.H>
#include <nodal_data_ops.H>
// clang-format on

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
    nodaldata_names.push_back("NZ");
#if USE_TEMP
    nodaldata_names.push_back("MASS_SPHEAT");
    nodaldata_names.push_back("MASS_SPHEAT_TEMP");
    nodaldata_names.push_back("TEMPERATURE");
    nodaldata_names.push_back("SOURCE_TEMP_INDEX");
    nodaldata_names.push_back("DELTA_TEMPERATURE");
#endif
}

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
    int coord = 0;
    RealBox real_box;
    for (int n = 0; n < AMREX_SPACEDIM; n++)
    {
        real_box.setLo(n, specs.plo[n]);
        real_box.setHi(n, specs.phi[n]);
    }

    // Defining index space
    IntVect domain_lo(AMREX_D_DECL(0, 0, 0));
    IntVect domain_hi(AMREX_D_DECL(specs.ncells[XDIR] - 1,
                                   specs.ncells[YDIR] - 1,
                                   specs.ncells[ZDIR] - 1));

    // Defining box
    const Box domain(domain_lo, domain_hi);

    // Defining geometry class
    geom.define(domain, &real_box, coord, specs.periodic.data());

    // Defining box array
    ba.define(domain);

    // Max size for box array chunking
    ba.maxSize(specs.max_grid_size);

    // Defining distribution mapping
    dm.define(ba);

    Name_Nodaldata_Variables(nodaldata_names);

    // Defining number of ghost cells for particle data
    ng_cells = 1;

    if (specs.order_scheme == 3)
    {
        ng_cells = 2;
    }

    ng_cells_nodaldata = 1;
    if (specs.order_scheme == 1)
    {
        ng_cells_nodaldata = 1;
    }
    else if (specs.order_scheme == 3)
    {
        ng_cells_nodaldata = 3;

        specs.order_scheme_directional[XDIR] =
            ((specs.periodic[XDIR] == 0) ? ((specs.ncells[XDIR] < 5) ? 1 : 3)
                                         : ((specs.ncells[XDIR] < 3) ? 1 : 3));
        specs.order_scheme_directional[YDIR] =
            ((specs.periodic[YDIR] == 0) ? ((specs.ncells[YDIR] < 5) ? 1 : 3)
                                         : ((specs.ncells[YDIR] < 3) ? 1 : 3));
        specs.order_scheme_directional[ZDIR] =
            ((specs.periodic[ZDIR] == 0) ? ((specs.ncells[ZDIR] < 5) ? 1 : 3)
                                         : ((specs.ncells[ZDIR] < 3) ? 1 : 3));

        if (specs.order_scheme_directional[XDIR] == 1 &&
            specs.order_scheme_directional[YDIR] == 1 &&
            specs.order_scheme_directional[ZDIR] == 1)
        {
            amrex::Print()
                << "\nWarning! Number of cells in all directions do not "
                   "qualify for cubic-spline shape functions\n";
            amrex::Print() << "Reverting to linear hat shape functions in "
                              "all directions\n";
        }

        // Make sure that none of the boxes that use spline function are of
        // size of 1. For example if ncell=5 and max_grid_size = 2,we get
        // boxes of {2,2,1}. I (Sreejith) noticed that when the box size is
        // one ghost particles are not placed correctly.
        for (int box_index = 0; box_index < ba.size(); box_index++)
        {
            for (int dim = 0; dim < AMREX_SPACEDIM; dim++)
            {
                if (ba[box_index].size()[dim] == 1 and
                    specs.order_scheme_directional[dim] == 3)
                {
                    amrex::Abort("Error: Box cannot be of size =1");
                    // Please change max_grid_size value
                    //              to make sure all boxes have size>1 when
                    //              using splines");
                }
            }
        }
    }
    else
    {
        amrex::Abort("Order scheme not implemented yet");
        // Please use order_scheme=1
        //              or order_scheme=3 in the input file \n");
    }

    const BoxArray &nodeba = amrex::convert(ba, IntVect{1, 1, 1});
    nodaldata.define(nodeba, dm, NUM_STATES, ng_cells_nodaldata);
    nodaldata.setVal(0.0, ng_cells_nodaldata);

    BoxArray phase_ba = ba;
    Box dom_levset = geom.Domain();
    dom_levset.refine(specs.levset_gridratio);
    geom_levset.define(dom_levset);
    int ng_phase = 3;
    if (specs.levset_output)
    {
        phase_ba.refine(specs.levset_gridratio);
        levset_data.define(phase_ba, dm, 1, ng_phase);
        levset_data.setVal(0.0, ng_phase);
    }
    PrintMessage("", print_length, false);
}

void Create_Output_Directories(MPMspecs &specs)
{

    amrex::UtilCreateDirectory(specs.particle_output_folder, 0755);
    amrex::UtilCreateDirectory(specs.grid_output_folder, 0755);
    amrex::UtilCreateDirectory(specs.checkpoint_output_folder, 0755);
    if (specs.levset_output)
    {
        amrex::UtilCreateDirectory(specs.levset_output_folder, 0755);
    }
}

void Initialise_Internal_Forces(MPMspecs &specs,
                                MPMParticleContainer &mpm_pc,
                                amrex::MultiFab &nodaldata,
                                amrex::MultiFab &levset_data,
                                Geometry &geom,
                                Geometry &geom_levset)
{
    amrex::Real dt;
    dt = mpm_pc.Calculate_time_step(specs);

    std::string msg;
    msg = "\n Calculating initial strainrates and stresses";
    PrintMessage(msg, print_length, true);
    mpm_pc.deposit_onto_grid_momentum(
        nodaldata, specs.gravity, specs.external_loads_present,
        specs.force_slab_lo, specs.force_slab_hi, specs.extforce, 1, 0,
        specs.mass_tolerance, specs.order_scheme_directional, specs.periodic);

    // Calculate strainrate at each material point
    mpm_pc.interpolate_from_grid(nodaldata, 0, 1,
                                 specs.order_scheme_directional, specs.periodic,
                                 specs.alpha_pic_flip, dt);

    mpm_pc.apply_constitutive_model(dt, specs.applied_strainrate);
    PrintMessage(msg, print_length, false);

    msg = "\n Updating phase field";
    PrintMessage(msg, print_length, true);
    if (specs.levset_output)
    {
        mpm_pc.update_phase_field(levset_data, specs.levset_gridratio,
                                  specs.levset_smoothfactor);
    }
    PrintMessage(msg, print_length, false);

#if USE_TEMP
    msg = "\n Calculating initial heat flux";
    PrintMessage(msg, print_length, true);

    Array<Real, AMREX_SPACEDIM> temp_lo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    Array<Real, AMREX_SPACEDIM> temp_hi{AMREX_D_DECL(1.0, 0.0, 0.0)};

    mpm_pc.deposit_onto_grid_temperature(
        nodaldata, true, true, specs.mass_tolerance,
        specs.order_scheme_directional, specs.periodic);
    nodal_bcs_temperature(geom, nodaldata, specs.bclo.data(), specs.bchi.data(),
                          temp_lo.data(), temp_hi.data(), dt);
    // backup_current_temperature(nodaldata);

    // store_delta_temperature(nodaldata);
    //  Calculate strainrate at each material point
    mpm_pc.interpolate_from_grid_temperature(nodaldata, true, true,
                                             specs.order_scheme_directional,
                                             specs.periodic, 0.5, dt);

    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        const Box &bx = mfi.validbox();
        Box nodalbox = convert(bx, {1, 1, 1});

        Array4<Real> nodal_data_arr = nodaldata.array(mfi);

        amrex::ParallelFor(nodalbox,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           {
                               IntVect nodeid(i, j, k);
                               // amrex::Print()<<"\n Temperature, i =
                               // "<<i<<" j= "<<j<<" k= "<<k<<"
                               // "<<nodal_data_arr(i,j,k,TEMPERATURE);
                           });
    }

#endif
}

void Initialise_Material_Points(MPMspecs &specs,
                                MPMParticleContainer &mpm_pc,
                                int &steps,
                                Real &time,
                                int &output_it)
{
    if (specs.restart_checkfile != "")
    {
        std::string msg;
        msg = "\n Acquiring particle data (restarting from checkpoint file)";
        PrintMessage(msg, print_length, true);
        mpm_pc.readCheckpointFile(specs.restart_checkfile, steps, time,
                                  output_it);
        PrintMessage(msg, print_length, true);
    }
    else if (!specs.use_autogen)
    {
        std::string msg;
        msg = "\n Acquiring particle data (Reading from particle file)";
        PrintMessage(msg, print_length, true);
        mpm_pc.InitParticles(specs.particlefilename, specs.total_mass,
                             specs.total_vol, specs.total_rigid_mass,
                             specs.no_of_rigidbodies_present,
                             specs.ifrigidnodespresent);
        PrintMessage(msg, print_length, false);
        mpm_pc.RedistributeLocal();
        mpm_pc.fillNeighbors();

        if (specs.no_of_rigidbodies_present != numrigidbodies)
        {
            amrex::Print() << "\n specs.no_of_rigidbodies_present= "
                           << specs.no_of_rigidbodies_present << " "
                           << numrigidbodies;
            // amrex::Abort("\n Sorry! The number of rigid bodies defined in
            // particles file and in constants.H file do not match.
            // Aborting..");
        }
    }
    else
    {
        std::string msg;
        msg = "\n Acquiring particle data (using autogen)";
        PrintMessage(msg, print_length, true);
        mpm_pc.InitParticles(
            specs.autogen_mincoords.data(), specs.autogen_maxcoords.data(),
            specs.autogen_vel.data(), specs.autogen_dens,
            specs.autogen_constmodel, specs.autogen_E, specs.autogen_nu,
            specs.autogen_bulkmod, specs.autogen_Gama_pres, specs.autogen_visc,
            specs.autogen_multi_part_per_cell, specs.total_mass,
            specs.total_vol);
        PrintMessage(msg, print_length, false);
    }

    if (mpm_ebtools::using_levelset_geometry)
    {
        mpm_pc.removeParticlesInsideEB();
    }

    mpm_pc.RedistributeLocal();
    mpm_pc.fillNeighbors();
}

void MPMParticleContainer::InitParticles(const std::string &filename,
                                         Real &total_mass,
                                         Real &total_vol,
                                         Real &total_rigid_mass,
                                         int &num_of_rigid_bodies,
                                         int &ifrigidnodespresent)
{

    // only read the file on the IO proc
    if (ParallelDescriptor::IOProcessor())
    {
        std::ifstream ifs;
        ifs.open(filename.c_str(), std::ios::in);

        if (!ifs.good())
        {
            amrex::FileOpenFailed(filename);
        }

        int np = -1;
        ifs >> np >> std::ws;

        if (np == -1)
        {
            Abort("\nCannot read number of particles from particle file\n");
        }

        const int lev = 0;
        const int grid = 0;
        const int tile = 0;
        const int tot_rig_body_tmp = 10;
        int rigid_bodies_read_so_far[tot_rig_body_tmp] = {-1};
        int index_rigid_body_read_so_far = 0;

        total_mass = 0.0;       // Total mass of phase 0 material points
        total_vol = 0.0;        // Total volume of phase 0 material points
        total_rigid_mass = 0.0; // Total mass of phase 1 material points

        auto &particle_tile = DefineAndReturnParticleTile(lev, grid, tile);
        Gpu::HostVector<ParticleType> host_particles;

        for (int i = 0; i < np; i++)
        {
            ParticleType p;
            int ph;
            amrex::Real junk;

            // Set id and cpu for this particle
            p.id() = ParticleType::NextID();
            p.cpu() = ParallelDescriptor::MyProc();

            // Read from input file
            ifs >>
                p.idata(intData::phase); // phase=0=> use for mpm computation,
                                         // phase=1=> rigid body particles, not
                                         // used in std. mpm operations

            if (p.idata(intData::phase) == 1)
            {
                ifrigidnodespresent = 1;
                ifs >>
                    p.idata(
                        intData::rigid_body_id); // if there are multiple rigid
                                                 // bodies present, then tag
                                                 // them separately using this
                                                 // id. For the HPRO problem,
                                                 // rigid_body_id=0=> top jaw,
                                                 // rigid_body_id=1=>bottom jaw
                // Check if the rigid_body_id is not read before
                bool body_present = false;
                for (int j = 0; j < index_rigid_body_read_so_far; j++)
                {
                    if (rigid_bodies_read_so_far[j] ==
                        p.idata(intData::rigid_body_id))
                    {
                        body_present = true;
                    }
                }
                if (!body_present)
                {
                    rigid_bodies_read_so_far[index_rigid_body_read_so_far] =
                        p.idata(intData::rigid_body_id);
                    index_rigid_body_read_so_far++;
                }
            }
            else
            {
                p.idata(intData::rigid_body_id) =
                    -1; // rigid_body_id is invalid for phase=0 material points.
            }

            ifs >> p.pos(0);
            ifs >> p.pos(1);
            ifs >> p.pos(2);

            ifs >> p.rdata(realData::radius);
            ifs >> p.rdata(realData::density);
            ifs >> p.rdata(realData::xvel);
            ifs >> p.rdata(realData::yvel);
            ifs >> p.rdata(realData::zvel);
            ifs >> p.idata(intData::constitutive_model);

            if (p.idata(intData::constitutive_model) == 0) // Elastic solid
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

                amrex::Abort(
                    "\n\tIncorrect constitutive model. Please check your "
                    "particle file");
            }

#if USE_TEMP
            ifs >> p.rdata(realData::temperature);
            ifs >> p.rdata(realData::specific_heat);
            ifs >> p.rdata(realData::thermal_conductivity);
            ifs >> p.rdata(realData::heat_source);
            p.rdata(realData::heat_flux + 0) = 0.0;
            p.rdata(realData::heat_flux + 1) = 0.0;
            p.rdata(realData::heat_flux + 2) = 0.0;
#endif

            // Set other particle properties
            p.rdata(realData::volume) =
                fourbythree * PI *
                pow(p.rdata(realData::radius),
                    three); // Material point is assumed to be a sphere. The
                            // radius provided in the input particle file is
                            // used to calculate the mp volume
            p.rdata(realData::mass) =
                p.rdata(realData::density) * p.rdata(realData::volume);

            // amrex::Print()<<"\n Mass =  "<<p.rdata(realData::mass)<<"
            // "<<p.rdata(realData::temperature);

            if (p.idata(intData::phase) == 0)
            {
                total_mass += p.rdata(realData::mass);
                total_vol += p.rdata(realData::volume);
            }
            else if (p.idata(intData::phase) == 1 and
                     p.idata(intData::rigid_body_id) == 0)
            {
                total_rigid_mass += p.rdata(realData::mass);
            }

            p.rdata(realData::jacobian) = 1.0;
            p.rdata(realData::vol_init) = p.rdata(realData::volume);
            p.rdata(realData::pressure) = 0.0;

            for (int comp = 0; comp < NCOMP_FULLTENSOR; comp++)
            {
                p.rdata(realData::deformation_gradient + comp) = 0.0;
            }
            p.rdata(realData::deformation_gradient + 0) = 1.0;
            p.rdata(realData::deformation_gradient + 4) = 1.0;
            p.rdata(realData::deformation_gradient + 8) = 1.0;

            for (int comp = 0; comp < NCOMP_TENSOR; comp++)
            {
                p.rdata(realData::strainrate + comp) = zero;
                p.rdata(realData::strain + comp) = zero;
                p.rdata(realData::stress + comp) = zero;
            }

            host_particles.push_back(p);

            if (!ifs.good())
            {
                amrex::Abort(
                    "Error initializing particles from Ascii file. \n");
            }
        }

        num_of_rigid_bodies = index_rigid_body_read_so_far;
        auto old_size = particle_tile.GetArrayOfStructs().size();
        auto new_size = old_size + host_particles.size();
        particle_tile.resize(new_size);

        Gpu::copy(Gpu::hostToDevice, host_particles.begin(),
                  host_particles.end(),
                  particle_tile.GetArrayOfStructs().begin() + old_size);
    }
    Redistribute();
}

void MPMParticleContainer::InitParticles(Real mincoords[AMREX_SPACEDIM],
                                         Real maxcoords[AMREX_SPACEDIM],
                                         Real vel[AMREX_SPACEDIM],
                                         Real dens,
                                         int constmodel,
                                         Real E,
                                         Real nu,
                                         Real bulkmod,
                                         Real Gama_pres,
                                         Real visc,
                                         int do_multi_part_per_cell,
                                         Real &total_mass,
                                         Real &total_vol)
{
    int lev = 0;
    Real x, y, z, x0, y0, z0;

    Real dx = Geom(lev).CellSize(0);
    Real dy = Geom(lev).CellSize(1);
    Real dz = Geom(lev).CellSize(2);
    const Real *plo = Geom(lev).ProbLo();

    total_mass = 0.0;
    total_vol = 0.0;

    // std::mt19937 mt(0451);
    // std::uniform_real_distribution<double> dist(0.4, 0.6);

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
                x = plo[XDIR] + (iv[XDIR] + half) * dx;
                y = plo[YDIR] + (iv[YDIR] + half) * dy;
                z = plo[ZDIR] + (iv[ZDIR] + half) * dz;

                if (x >= mincoords[XDIR] && x <= maxcoords[XDIR] &&
                    y >= mincoords[YDIR] && y <= maxcoords[YDIR] &&
                    z >= mincoords[ZDIR] && z <= maxcoords[ZDIR])
                {
                    ParticleType p = generate_particle(
                        x, y, z, vel, dens, dx * dy * dz, constmodel, E, nu,
                        bulkmod, Gama_pres, visc);

                    total_mass += p.rdata(realData::mass);
                    total_vol += p.rdata(realData::volume);

                    host_particles.push_back(p);
                }
            }
            else
            {
                x0 = plo[XDIR] + iv[XDIR] * dx;
                y0 = plo[YDIR] + iv[YDIR] * dy;
                z0 = plo[ZDIR] + iv[ZDIR] * dz;

                for (int k = 0; k < 2; k++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        for (int i = 0; i < 2; i++)
                        {
                            // x = x0 + (i+dist(mt))*half*dx;
                            // y = y0 + (j+dist(mt))*half*dy;
                            // z = z0 + (k+dist(mt))*half*dz;
                            x = x0 + (i + half) * half * dx;
                            y = y0 + (j + half) * half * dy;
                            z = z0 + (k + half) * half * dz;

                            if (x >= mincoords[XDIR] and
                                x <= maxcoords[XDIR] and
                                y >= mincoords[YDIR] and
                                y <= maxcoords[YDIR] and
                                z >= mincoords[ZDIR] and z <= maxcoords[ZDIR])
                            {
                                ParticleType p = generate_particle(
                                    x, y, z, vel, dens, eighth * dx * dy * dz,
                                    constmodel, E, nu, bulkmod, Gama_pres,
                                    visc);

                                total_mass += p.rdata(realData::mass);
                                total_vol += p.rdata(realData::volume);

                                host_particles.push_back(p);
                            }
                        }
                    }
                }
            }
        }

        auto old_size = particle_tile.GetArrayOfStructs().size();
        auto new_size = old_size + host_particles.size();
        particle_tile.resize(new_size);

        Gpu::copy(Gpu::hostToDevice, host_particles.begin(),
                  host_particles.end(),
                  particle_tile.GetArrayOfStructs().begin() + old_size);
    }

    // We shouldn't need this if the particles are tiled with one tile per grid,
    // but otherwise we do need this to move particles from tile 0 to the
    // correct tile.
    Redistribute();
}

MPMParticleContainer::ParticleType
MPMParticleContainer::generate_particle(Real x,
                                        Real y,
                                        Real z,
                                        Real vel[AMREX_SPACEDIM],
                                        Real dens,
                                        Real vol,
                                        int constmodel,
                                        Real E,
                                        Real nu,
                                        Real bulkmod,
                                        Real Gama_pres,
                                        Real visc)
{
    ParticleType p;
    p.id() = ParticleType::NextID();
    p.cpu() = ParallelDescriptor::MyProc();

    p.pos(XDIR) = x;
    p.pos(YDIR) = y;
    p.pos(ZDIR) = z;

    p.idata(intData::phase) =
        0; // Make sure this simulation does not use rigid body particles
    p.rdata(realData::radius) = std::pow(three * fourth * vol / PI, 0.33333333);

    p.rdata(realData::density) = dens;
    p.rdata(realData::xvel) = vel[XDIR];
    p.rdata(realData::yvel) = vel[YDIR];
    p.rdata(realData::zvel) = vel[ZDIR];

    p.idata(intData::constitutive_model) = constmodel;

    p.rdata(realData::E) = E;
    p.rdata(realData::nu) = nu;
    p.rdata(realData::Bulk_modulus) = bulkmod;
    p.rdata(realData::Gama_pressure) = Gama_pres;
    p.rdata(realData::Dynamic_viscosity) = visc;

    p.rdata(realData::volume) = vol;
    p.rdata(realData::mass) = dens * vol;
    p.rdata(realData::jacobian) = 1.0;
    p.rdata(realData::pressure) = 0.0;
    p.rdata(realData::vol_init) = 0.0;

    for (int comp = 0; comp < NCOMP_TENSOR; comp++)
    {
        p.rdata(realData::strainrate + comp) = zero;
        p.rdata(realData::strain + comp) = zero;
        p.rdata(realData::stress + comp) = zero;
    }

    return (p);
}

void MPMParticleContainer::removeParticlesInsideEB()
{
    const int lev = 0;
    const Geometry &geom = Geom(lev);
    auto &plev = GetParticles(lev);
    const auto dxi = geom.InvCellSizeArray();
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto domain = geom.Domain();

    int lsref = mpm_ebtools::ls_refinement;

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        const amrex::Box &box = mfi.tilebox();
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
                               amrex::Real xp[AMREX_SPACEDIM] = {
                                   p.pos(XDIR), p.pos(YDIR), p.pos(ZDIR)};

                               amrex::Real lsval = get_levelset_value(
                                   lsetarr, plo, dx, xp, lsref);

                               if (lsval < TINYVAL)
                               {
                                   p.id() = -1;
                               }
                           });
    }
    Redistribute();
}
