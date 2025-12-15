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
        // Rigid_Bodies *Rb=nullptr;
        specs.read_mpm_specs();

        // Declaring solver variables
        int steps = 0;
        Real dt;
        Real time = 0.0;
        // int num_of_rigid_bodies = 0;
        int output_it = 0;
        std::string pltfile;
        Real output_time = shunya;
        Real output_timePrint = shunya;
        Real diag_timePrint = shunya;
        // GpuArray<int, AMREX_SPACEDIM> order_surface_integral =
        // {AMREX_D_DECL(3, 3, 3)};
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

        Initialise_Diagnostic_Streams(specs);

        Initialise_Internal_Forces(specs, mpm_pc, nodaldata, levset_data);

        Write_Particle_Grid_Levset_Output(specs, mpm_pc, nodaldata, levset_data,
                                          nodaldata_names, geom, geom_levset,
                                          ba, dm, time, steps, output_it, true);

        amrex::Print() << "\n\nTimestepping begins\n\n";

        while ((steps < specs.maxsteps) and (time < specs.final_time))
        {
            steps++;
            auto iter_time_start = amrex::second();

            Redistribute_Fill_Update(specs, mpm_pc, steps);

            dt = mpm_pc.Calculate_time_step(specs);

            Reset_Nodaldata_to_Zero(nodaldata, ng_cells_nodaldata);

            P2G_Momentum(specs, mpm_pc, nodaldata, 1, 1, 1);

            backup_current_velocity(nodaldata);

            Nodal_Time_Update_Momentum(nodaldata, dt, specs.mass_tolerance);

            Apply_Nodal_BCs(geom, nodaldata, specs, dt);

            if (specs.stress_update_scheme == 0)
            {
                // Algo 1, step 18, 20, 21, 23 Vacoeboil;s paper
                G2P_Momentum(specs, mpm_pc, nodaldata, 1, 1, dt);
                Update_MP_Positions(specs, mpm_pc, dt); // step 19
            }

            // mpm_pc.updateNeighbors();

            if (specs.stress_update_scheme == 1)
            {
                // Algo 2, 19
                G2P_Momentum(specs, mpm_pc, nodaldata, 1, 0, dt);
                // 20
                P2G_Momentum(specs, mpm_pc, nodaldata, 0, 1, 0);
                // 21
                Apply_Nodal_BCs(geom, nodaldata, specs, dt);
                // 25
                G2P_Momentum(specs, mpm_pc, nodaldata, 0, 1, dt);
                Update_MP_Positions(specs, mpm_pc, dt); // step 18
            }

            Redistribute_Fill_Update(specs, mpm_pc, steps);

            // mpm_pc.updateNeighbors();

            Update_MP_Volume(mpm_pc);

            Calculate_MP_Stress_Strain(specs, mpm_pc, time, dt);

            if (specs.levset_output)
            {
                mpm_pc.update_phase_field(levset_data, specs.levset_gridratio,
                                          specs.levset_smoothfactor);
            }

            if (diag_timePrint >= specs.write_diag_output_time)
            {
                // amrex::Print()<<"\n Writing diagnostic files..";
                Do_All_Diagnostics(specs, mpm_pc, steps, time);
                diag_timePrint = shunya;
            }

            if (fabs(output_time - specs.write_output_time) < dt * 0.5)
            {
                output_it++;
                Write_Particle_Grid_Levset_Output(
                    specs, mpm_pc, nodaldata, levset_data, nodaldata_names,
                    geom, geom_levset, ba, dm, time, steps, output_it, true);
                output_time = shunya;
                BL_PROFILE_VAR_STOP(outputs);
            }

            time += dt;
            output_time += dt;
            output_timePrint += dt;
            diag_timePrint += dt;

            auto time_per_iter = amrex::second() - iter_time_start;
            if (output_timePrint >= specs.screen_output_time)
            {
                Print() << "Iteration: " << std::setw(10) << steps << ",\t"
                        << "Time: " << std::fixed << std::setprecision(10)
                        << time << ",\tDt = " << std::scientific
                        << std::setprecision(5) << dt << std::fixed
                        << std::setprecision(10)
                        << ",\t Time/Iter = " << time_per_iter << "\n";
                output_timePrint = shunya;
            }
        }

        Write_Particle_Grid_Levset_Output(specs, mpm_pc, nodaldata, levset_data,
                                          nodaldata_names, geom, geom_levset,
                                          ba, dm, time, steps, output_it, true);
        Close_Diagnostic_Streams(specs);
    }

    amrex::Finalize();
}
