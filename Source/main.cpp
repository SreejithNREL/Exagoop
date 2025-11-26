// clang-format off
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <mpm_check_pair.H>
#include <mpm_particle_container.H>
#include <AMReX_PlotFileUtil.H>
#include <nodal_data_ops.H>
#include <mpm_eb.H>
#include <mpm_init.H>
#include <utilities.H>

// clang-format on

#include <aesthetics.H>

using namespace amrex;

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);

    {
        PrintWelcomeMessage();

        // Initializing and reading input file for the simulation
        MPMspecs specs;
        Rigid_Bodies *Rb;
        specs.read_mpm_specs();

        // Declaring solver variables
        int steps = 0;
        Real dt;
        Real time = 0.0;
        int num_of_rigid_bodies = 0;
        int output_it = 0;
        std::string pltfile;
        Real output_time = zero;
        Real output_timePrint = zero;
        GpuArray<int, AMREX_SPACEDIM> order_surface_integral = {AMREX_D_DECL(3, 3, 3)};
        std::string msg;

        int ng_cells;
        Geometry geom;
        Geometry geom_levset;
        BoxArray ba;
        DistributionMapping dm;
        MultiFab nodaldata;
        MultiFab levset_data;
        amrex::Vector<std::string> nodaldata_names;
        int ng_cells_nodaldata;

        Initialise_Domain(specs, geom, geom_levset, ba, dm, ng_cells,
                          ng_cells_nodaldata, nodaldata, levset_data,
                          nodaldata_names);

        mpm_ebtools::init_eb(geom, ba, dm);

        MPMParticleContainer mpm_pc(geom, dm, ba, ng_cells);

        Initialise_Material_Points(specs, mpm_pc, steps, time, output_it);

        Create_Output_Directories(specs);

        Initialise_Internal_Forces(specs, mpm_pc, nodaldata, levset_data, geom,
                                   geom_levset);

        amrex::Print() << "\n\nTimestepping begins\n\n";

        while ((steps < specs.maxsteps) and (time < specs.final_time))
        {
            auto iter_time_start = amrex::second();

            dt = mpm_pc.Calculate_time_step(specs);

            time += dt;
            output_time += dt;
            output_timePrint += dt;
            steps++;

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

            Reset_Nodaldata_to_Zero(nodaldata, ng_cells_nodaldata);            
            
            mpm_pc.deposit_onto_grid_momentum( nodaldata, specs.gravity, specs.external_loads_present, specs.force_slab_lo, specs.force_slab_hi, specs.extforce, 1, 1, specs.mass_tolerance, specs.order_scheme_directional, specs.periodic);            
			
            nodal_update(nodaldata, dt, specs.mass_tolerance);

			// impose bcs at nodes
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

            

#if USE_TEMP
            // Temperature steps start
            mpm_pc.deposit_onto_grid_temperature(
                nodaldata, true, true, specs.mass_tolerance,
                specs.order_scheme_directional, specs.periodic);
            backup_current_temperature(nodaldata);
            mpm_pc.interpolate_from_grid_temperature(
                nodaldata, true, true, specs.order_scheme_directional,
                specs.periodic, 1.0, dt);
            // Temperature steps end
            nodal_update_temperature(nodaldata, dt, specs.mass_tolerance);

            Array<Real, AMREX_SPACEDIM> temp_lo{AMREX_D_DECL(0.0, 0.0, 0.0)};
            Array<Real, AMREX_SPACEDIM> temp_hi{AMREX_D_DECL(1.0, 0.0, 0.0)};

            nodal_bcs_temperature(geom, nodaldata, specs.bclo.data(),
                                  specs.bchi.data(), temp_lo.data(),
                                  temp_hi.data(), dt);
            store_delta_temperature(nodaldata);
#endif
            
            
            mpm_pc.interpolate_from_grid(
                nodaldata, 1, 0, specs.order_scheme_directional, specs.periodic,
                specs.alpha_pic_flip, dt);

#if USE_TEMP
            mpm_pc.interpolate_from_grid_temperature(
                nodaldata, true, true, specs.order_scheme_directional,
                specs.periodic, 1.0, dt);
#endif            

            // Update particle position at t+dt
            mpm_pc.moveParticles(
                dt, specs.bclo.data(), specs.bchi.data(), specs.levelset_bc,
                specs.wall_mu_lo.data(), specs.wall_mu_hi.data(),
                specs.wall_vel_lo.data(), specs.wall_vel_hi.data(),
                specs.levelset_wall_mu);

            if (specs.stress_update_scheme == 1)
            {
                // MUSL scheme
                //  Calculate velocity on nodes
                mpm_pc.deposit_onto_grid_momentum(
                    nodaldata, specs.gravity, specs.external_loads_present,
                    specs.force_slab_lo, specs.force_slab_hi, specs.extforce, 1,
                    0, specs.mass_tolerance, specs.order_scheme_directional,
                    specs.periodic);

                nodal_bcs(geom, nodaldata, specs.bclo.data(), specs.bchi.data(),
                          specs.wall_mu_lo.data(), specs.wall_mu_hi.data(),
                          specs.wall_vel_lo.data(), specs.wall_vel_hi.data(),
                          dt);
                // nodal_bcs(	geom, nodaldata, dt);

                if (mpm_ebtools::using_levelset_geometry)
                {
                    nodal_levelset_bcs(nodaldata, geom, dt, specs.levelset_bc,
                                       specs.levelset_wall_mu);
                }
            }

            // find strainrate at material points at time t+dt
            mpm_pc.interpolate_from_grid(
                nodaldata, 0, 1, specs.order_scheme_directional, specs.periodic,
                specs.alpha_pic_flip, dt);
#if USE_TEMP
            mpm_pc.interpolate_from_grid_temperature(
                nodaldata, true, true, specs.order_scheme_directional,
                specs.periodic, 1.0, dt);
#endif
            mpm_pc.updateNeighbors();

            // mpm_pc.move_particles_from_nodevel(nodaldata,dt,
            // specs.bclo.data(),specs.bchi.data(),1);
            mpm_pc.updateVolume(dt);

            // update stress at material pointsat time t+dt
            if (time < specs.applied_strainrate_time)
            {
                if (specs.calculate_strain_based_on_delta == 1)
                {
                    mpm_pc.apply_constitutive_model_delta(
                        dt, specs.applied_strainrate);
                }
                else
                {
                    mpm_pc.apply_constitutive_model(dt,
                                                    specs.applied_strainrate);
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

            if (specs.levset_output)
            {
                mpm_pc.update_phase_field(levset_data, specs.levset_gridratio,
                                          specs.levset_smoothfactor);
            }

            if (fabs(output_time - specs.write_output_time) < dt * 0.5)
            {
                output_it++;
                Write_Particle_Grid_Levset_Output(
                    specs, mpm_pc, nodaldata, levset_data, nodaldata_names,
                    geom, geom_levset, ba, dm, time, steps, output_it, true);
                output_time = zero;
                BL_PROFILE_VAR_STOP(outputs);
            }

            auto time_per_iter = amrex::second() - iter_time_start;
            if (output_timePrint >= specs.screen_output_time)
            {
                Print() << "Iteration: " << std::setw(10) << steps << ",\t"
                        << "Time: " << std::fixed << std::setprecision(10)
                        << time << ",\tDt = " << std::scientific
                        << std::setprecision(5) << dt << std::fixed
                        << std::setprecision(10)
                        << ",\t Time/Iter = " << time_per_iter << "\n";
                output_timePrint = zero;
            }
        }

        Write_Particle_Grid_Levset_Output(specs, mpm_pc, nodaldata, levset_data,
                                          nodaldata_names, geom, geom_levset,
                                          ba, dm, time, steps, output_it, true);
    }

    amrex::Finalize();
}
