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
    ng_cells = (specs.order_scheme == 3) ? 2 : 1;

    // Ghost cells for nodal data
    if (specs.order_scheme == 1)
    {
        ng_cells_nodaldata = 1;
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
                                amrex::MultiFab &levset_data)
{
    amrex::Real dt = mpm_pc.Calculate_time_step(specs);

    // Momentum deposition and initial stress/strainrate
    {
        std::string msg = "\n Calculating initial strainrates and stresses";
        PrintMessage(msg, print_length, true);
        amrex::Print()<<"\n Printing..";

        mpm_pc.deposit_onto_grid_momentum(
            nodaldata, specs.gravity, specs.external_loads_present,
            specs.force_slab_lo, specs.force_slab_hi, specs.extforce,
            /*do_reset=*/1,
            /*do_average=*/1, specs.mass_tolerance,
            specs.order_scheme_directional, specs.periodic);

        amrex::Print()<<"\n P2G done";

	    

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
            temp_lo[d] = 0.0;
            // Example: activate BC in first dimension only; adjust as needed
            temp_hi[d] = (d == 0) ? 1.0 : 0.0;
        }

        // Deposit temperature-related nodal quantities
        mpm_pc.deposit_onto_grid_temperature(
            nodaldata,
            /*do_reset=*/true,
            /*do_average=*/true, specs.mass_tolerance,
            specs.order_scheme_directional, specs.periodic);

        // Apply nodal boundary conditions (ensure correct Geometry is passed)
        const Geometry &geom = mpm_pc.Geom(0);
        nodal_bcs_temperature(geom, nodaldata, specs.bclo.data(),
                              specs.bchi.data(), temp_lo.data(), temp_hi.data(),
                              dt);

        // Interpolate temperature grid -> particles
        mpm_pc.interpolate_from_grid_temperature(
            nodaldata,
            /*do_reset=*/true,
            /*do_average=*/true, specs.order_scheme_directional, specs.periodic,
            /*alpha_pic_flip_temp=*/0.5, dt);

        // Dimension-aware nodal box conversion (1 = nodal in each active dim)
        for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
        {
            const Box &bx = mfi.validbox();
            const IntVect nodal_iv{
                AMREX_D_DECL(1, 1, 1)}; // compiled to 1D/2D/3D
            Box nodalbox = convert(bx, nodal_iv);
            auto nodal_data_arr = nodaldata.array(mfi);

            amrex::ParallelFor(nodalbox,
                               [=]
                               AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               {
                                   const IntVect nodeid(i, j, k);
                                   // Example hook for debugging/inspection:
                                   // amrex::Real T = nodal_data_arr(i, j, k,
                                   // TEMPERATURE);
                               });
        }

        PrintMessage(msg, print_length, false);
    }
#endif
}

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
    if (mpm_ebtools::using_levelset_geometry)
    {
        mpm_pc.removeParticlesInsideEB();
    }

    mpm_pc.RedistributeLocal();
    mpm_pc.fillNeighbors();
}

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
        ifs >> np >> std::ws;
        if (np == -1)
        {
            Abort("\nCannot read number of particles from particle file\n");
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
                amrex::Real coord;
                ifs >> coord;
                p.pos(d) = coord;
            }
            // radius & density
                        ifs >> p.rdata(realData::radius);
                        ifs >> p.rdata(realData::density);

            // velocities (dimension‑aware)
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
                p.rdata(realData::strainrate + comp) = zero;
                p.rdata(realData::strain + comp) = zero;
                p.rdata(realData::stress + comp) = zero;
            }

            if(testing==1)
            {
            amrex::Print()<<"\n Particle "<<p.rdata(realData::radius)<<" "
						  <<p.rdata(realData::density)<<" "
						  <<p.rdata(realData::xvel)<<" "
						  <<p.rdata(realData::yvel)<<" "
						  <<p.rdata(realData::zvel)<<" "
						  <<p.idata(intData::constitutive_model)<<" "
						  <<p.rdata(realData::E)<<" "
						  <<p.rdata(realData::nu)<<" "
						  <<p.rdata(realData::volume)<<" "
						  <<p.rdata(realData::mass)<<" "
						  <<p.rdata(realData::deformation_gradient+0)<<" "
						  <<p.rdata(realData::deformation_gradient+1)<<" "
						  <<p.rdata(realData::deformation_gradient+2)<<" "
						  <<p.rdata(realData::deformation_gradient+3)<<" "
						  <<p.rdata(realData::deformation_gradient+4)<<" "
						  <<p.rdata(realData::deformation_gradient+5)<<" "
						  <<p.rdata(realData::deformation_gradient+6)<<" "
						  <<p.rdata(realData::deformation_gradient+7)<<" "
						  <<p.rdata(realData::deformation_gradient+8)<<" ";
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
                    coords[d] = ploA[d] + (iv[d] + half) * dxA[d];
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
                        amrex::Real offset = (bit + half) * half * dxA[d];
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
        p.rdata(realData::strainrate + comp) = zero;
        p.rdata(realData::strain + comp) = zero;
        p.rdata(realData::stress + comp) = zero;
    }

    return p;
}

void MPMParticleContainer::removeParticlesInsideEB()
{
    const int lev = 0;
    const Geometry &geom = Geom(lev);
    auto &plev = GetParticles(lev);
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();

    int lsref = mpm_ebtools::ls_refinement;

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
