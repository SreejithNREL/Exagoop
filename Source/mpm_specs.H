#ifndef MPM_PARTICLE_SPECS_H_
#define MPM_PARTICLE_SPECS_H_
// clang-format off
#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>
#include <constants.H>
// clang-format on

using namespace amrex;

struct realData
{
    enum
    { // Particle data stored in p.m_data
        radius = 0,
        xvel = 1,
        yvel = 2,
        zvel = 3,
        xvel_prime = 4, // For position update. Check the review paper page 34
        yvel_prime = 5,
        zvel_prime = 6,
        // For stress, strain and strainrate variables, we have 6 components
        // irrespective of AMREX_SPACEDIM. in 1D we use only the first
        // component, in 2D we use only the first 3 components etc.
        strainrate = 7, // 4
        strain = 13,    // 10
        stress = 19,    // 16
                        // for deformation gradient we have 9 components
                        // irrespective of AMREX_SPACEDIM
        deformation_gradient = 25, // 22
        volume = 34,               // 31
        mass = 35,                 // 32
        density = 36,              // 33
        jacobian = 37,             // 34
        pressure = 38,             // 35
        vol_init = 39,             // 36
        E = 40,                    // 37
        nu = 41,                   // 38
        Bulk_modulus = 42,         // 39
        Gama_pressure = 43,        // 40
        Dynamic_viscosity = 44,    // 41
        yacceleration =
            45, // 45	This is for storing the acceleration of each particle in
                // the y-direction. This variable is not generic and used only
                // for the spring-mass and membrane simulations only
#if USE_TEMP
        temperature = 46,
        specific_heat = 47,        // specific heat used in temperature equation
        thermal_conductivity = 48, // For now, treated as a scalar
        heat_flux = 49,            // AMREX_SPACEDIM size
        heat_source = 52,          // Used in the temperature equation
#endif
        count

    };
};

struct intData
{
    enum
    {
        phase = 0,
        rigid_body_id,
        constitutive_model,
        count
    };
};

struct Rigid_Bodies
{
    int Rigid_Body_Id;
    int num_of_mp;
    amrex::Real total_mass;
    amrex::Real total_volume;
    Array<amrex::Real, AMREX_SPACEDIM> gravity;
    Array<amrex::Real, AMREX_SPACEDIM> force_external;
    Array<amrex::Real, AMREX_SPACEDIM> force_internal;
    Array<amrex::Real, AMREX_SPACEDIM> velocity;
    Array<amrex::Real, AMREX_SPACEDIM> imposed_velocity;
    amrex::Real Damping_Coefficient;
    // Body forces
    int position_update_method; // 0-->Constant Velocity 1-->Particle dynamics
                                // solver
    bool enable_weight;
    bool enable_damping_force;
};

class MPMspecs
{
  public:
    Array<Real, AMREX_SPACEDIM> plo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    Array<Real, AMREX_SPACEDIM> phi{AMREX_D_DECL(0.0, 0.0, 0.0)};
    Array<int, AMREX_SPACEDIM> ncells{AMREX_D_DECL(0, 0, 0)};
    GpuArray<int, AMREX_SPACEDIM> periodic{AMREX_D_DECL(0, 0, 0)};
    Array<Real, AMREX_SPACEDIM> gravity{AMREX_D_DECL(0.0, -ACCG, 0.0)};

    Array<Real, AMREX_SPACEDIM> force_slab_lo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    Array<Real, AMREX_SPACEDIM> force_slab_hi{AMREX_D_DECL(0.0, 0.0, 0.0)};
    Array<Real, AMREX_SPACEDIM> extforce{AMREX_D_DECL(0.0, 0.0, 0.0)};

    Real timestep = 0.0;
    int num_redist = 10;
    int max_grid_size = 8;
    Real final_time;
    Real write_output_time = 1e200;
    Real write_diag_output_time = 1e200;
    Real screen_output_time;
    int maxsteps = 1;
    std::string particlefilename;
    std::string restart_checkfile;

    // Output parameters
    std::string prefix_particlefilename = "plt";
    std::string prefix_gridfilename = "plt";
    std::string prefix_densityfilename = "plt";
    std::string prefix_checkpointfilename = "chk";
    std::string prefix_asciifilename = "matpnt";
    std::string prefix_diagnosticname = "";

    bool write_ascii = false;

    std::string particle_output_folder = "./Solution/particle_files/";
    std::string grid_output_folder = "./Solution/grid_files/";
    std::string checkpoint_output_folder = "./Solution/checkpoint_files/";
    std::string ascii_output_folder = "./Solution/ascii_files/";
    std::string levset_output_folder = "./Solution/levset_files/";
    std::string diagnostic_output_folder = "./Diagnostics/";

    // Diagnostic output filenames
    std::string file_tke_tse = "Total_Energies.dat";
    std::string file_mwa_velcomp = "VelComponents.dat";
    std::string file_mwa_velmag = "VelMagnitudes.dat";
    std::string file_minmaxpos = "MinMaxPosition.dat";

    bool do_calculate_tke_tse = false;
    bool do_calculate_mwa_velcomp = false;
    bool do_calculate_mwa_velmag = false;
    bool do_calculate_minmaxpos = false;

    std::ofstream tmp_tke_tse;
    std::ofstream tmp_mwa_velcomp;
    std::ofstream tmp_mwa_velmag;
    std::ofstream tmp_minmaxpos;

    int num_of_digits_in_filenames = 6;

    // Diagnostic parameters
    int print_diagnostics = 0;
    int ifrigidnodespresent = 0;
    bool is_standard_test = true;
    int test_number = 0; // 0->is_standard_test=false,1-->axial vibration of
                         // bar, 2-->Dam break, 3--> Elastic collision of disks,
                         // 4--> Lateral deflection of beam
    int no_of_rigidbodies_present = 0;
    amrex::Real total_mass_phase_0 = 0.0;
    amrex::Real total_vol_phase_0 = 0.0;

    // Rigid body nodes
    Rigid_Bodies *Rb;

    // Parameters for membrane compaction
    amrex::Real mem_compaction_area = 1.0;
    amrex::Real mem_compaction_dampcoeff = 1.0;
    amrex::Real mem_compaction_L0 = 1.0;
    int mem_compaction_restoring_force_calc_method = 1;

    // Parameters for spring alone simulation
    amrex::Real spring_alone_area = 0.1;
    amrex::Real spring_alone_length = 0.0;
    amrex::Real spring_alone_E = 0.0;
    amrex::Real spring_alone_exact_deflection = 0.0;
    amrex::Real spring_alone_exact_delta = 0.0;

    // 1-->tent function, 3--> cubic spline shape function
    int order_scheme = 1;

    Real applied_strainrate = 0.0;
    Real applied_strainrate_time = 0.0;
    int levset_output = 0;
    int levset_gridratio = 1;
    Real levset_smoothfactor = 1.0;
    int external_loads_present = 0.0;
    int fixed_timestep =
        0; // Use fixed time step provided by user if this flag is 1
    Real dt_max_limit = 1e0;
    Real dt_min_limit = 1e-12;
    Real CFL = 0.1;
    Real mass_tolerance = 1e-8;
    Real alpha_pic_flip = 0.95;
    Real total_mass = 0.0;
    Real total_rigid_mass = 0.0;
    Real total_vol = 0.0;
    Real mem_compaction_vnew = 0.0;
    Real mem_compaction_vold = 0.0;
    int stress_update_scheme = 1;
    int calculate_strain_based_on_delta = 0;

    Vector<int> bclo;
    Vector<int> bchi;

#if USE_TEMP
    Vector<int> bclo_temp;
    Vector<int> bchi_temp;
    Vector<amrex::Real> bclo_tempval;
    Vector<amrex::Real> bchi_tempval;
#endif

    GpuArray<int, AMREX_SPACEDIM> order_scheme_directional{
        AMREX_D_DECL(1, 1, 1)};

    int use_autogen = 0;
    Vector<Real> autogen_mincoords;
    Vector<Real> autogen_maxcoords;
    Vector<Real> autogen_vel;
    int autogen_multi_part_per_cell = 1;
    Real autogen_dens = 1000.0;
    int autogen_constmodel = 1;
    Real autogen_E = 1e9;
    Real autogen_nu = 0.1;
    Real autogen_bulkmod = 0.0;
    Real autogen_Gama_pres = 1.4;
    Real autogen_visc = 0.001;
    Vector<int> autogen_ppc;

    Real autogen_T = 0.001;
    Real autogen_cp = 0.001;
    Real autogen_heatsrc = 0.001;
    Real autogen_thermcond = 0.001;

    int levelset_bc = 1; // no slip wall
    Real levelset_wall_mu = 0.1;

    Array<Real, AMREX_SPACEDIM> wall_mu_lo{AMREX_D_DECL(0.1, 0.1, 0.1)};
    Array<Real, AMREX_SPACEDIM> wall_mu_hi{AMREX_D_DECL(0.1, 0.1, 0.1)};

    Array<Real, AMREX_SPACEDIM * AMREX_SPACEDIM> wall_vel_lo{0.0};
    Array<Real, AMREX_SPACEDIM * AMREX_SPACEDIM> wall_vel_hi{0.0};

    Real levset_mu = 0.1;

    void read_mpm_specs()
    {
        bclo.resize(AMREX_SPACEDIM);
        bchi.resize(AMREX_SPACEDIM);
#if USE_TEMP
        bclo_temp.resize(AMREX_SPACEDIM);
        bchi_temp.resize(AMREX_SPACEDIM);
#endif

        autogen_mincoords.resize(AMREX_SPACEDIM);
        autogen_maxcoords.resize(AMREX_SPACEDIM);
        autogen_ppc.resize(AMREX_SPACEDIM);
        autogen_vel.resize(AMREX_SPACEDIM);

        ParmParse pp1("amr");
        restart_checkfile = "";
        pp1.query("restart_checkfile", restart_checkfile);

        ParmParse pp("mpm");
        pp.get("prob_lo", plo);
        pp.get("prob_hi", phi);
        pp.get("ncells", ncells);
        Array<int, AMREX_SPACEDIM> periodicarr{AMREX_D_DECL(0, 0, 0)};
        pp.get("is_it_periodic", periodicarr);
        periodic = {AMREX_D_DECL(periodicarr[XDIR], periodicarr[YDIR],
                                 periodicarr[ZDIR])};
        pp.get("max_grid_size", max_grid_size);

        AMREX_ASSERT(max_grid_size > 1);

        pp.get("max_steps", maxsteps);
        pp.get("final_time", final_time);

        pp.query("gravity", gravity);
        pp.query("calculate_strain_based_on_delta",
                 calculate_strain_based_on_delta);

        write_output_time = final_time / 10.0;
        pp.query("write_output_time", write_output_time);

        screen_output_time = write_output_time / 10.0;
        pp.query("screen_output_time", screen_output_time);

        pp.query("num_redist", num_redist);

        pp.query("use_autogen", use_autogen);

        // Read output parameters
        pp.query("prefix_particlefilename", prefix_particlefilename);
        pp.query("prefix_gridfilename", prefix_gridfilename);
        pp.query("prefix_densityfilename", prefix_densityfilename);
        pp.query("prefix_checkpointfilename", prefix_checkpointfilename);
        pp.query("prefix_asciifilename", prefix_asciifilename);
        pp.query("diagnostic_output_folder", diagnostic_output_folder);

        size_t pos = prefix_asciifilename.find('/');
        std::string temp_prefix;
        std::string temp_suffix;
        if (pos == std::string::npos)
        {
            temp_prefix = "";
            temp_suffix = prefix_asciifilename;
        }
        else
        {
            temp_prefix = prefix_asciifilename.substr(0, pos);
            temp_suffix = prefix_asciifilename.substr(pos + 1);
        }

        ascii_output_folder = ascii_output_folder + temp_prefix;
        prefix_asciifilename = temp_suffix;

        pp.query("num_of_digits_in_filenames", num_of_digits_in_filenames);
        pp.query("write_ascii", write_ascii);

        pp.query("mem_compaction_area", mem_compaction_area);
        pp.query("mem_compaction_dampcoeff", mem_compaction_dampcoeff);
        pp.query("mem_compaction_restoring_force_calc_method",
                 mem_compaction_restoring_force_calc_method);

        pp.query("spring_alone_area", spring_alone_area);
        pp.query("spring_alone_length", spring_alone_length);
        pp.query("spring_alone_E", spring_alone_E);

        if (!use_autogen)
        {
            particlefilename = "mpm_particles.dat";
            pp.query("particle_file", particlefilename);
        }
        else
        {
            pp.getarr("mincoords_autogen", autogen_mincoords);
            pp.getarr("maxcoords_autogen", autogen_maxcoords);
            pp.getarr("vel_autogen", autogen_vel);
            pp.query("constmodel_autogen", autogen_constmodel);
            pp.query("dens_autogen", autogen_dens);
            pp.query("E_autogen", autogen_E);
            pp.query("nu_autogen", autogen_nu);
            pp.query("bulkmod_autogen", autogen_bulkmod);
            pp.query("Gama_pres_autogen", autogen_Gama_pres);
            pp.query("visc_autogen", autogen_visc);
            pp.queryarr("ppc", autogen_ppc);
#if USE_TEMP
            pp.query("T_autogen", autogen_T);
            pp.query("cp_autogen", autogen_cp);
            pp.query("thermcond_autogen", autogen_thermcond);
            pp.query("heatsrc_autogen", autogen_heatsrc);
#endif

            pp.query("multi_part_per_cell_autogen",
                     autogen_multi_part_per_cell);
        }

        pp.query("fixed_timestep", fixed_timestep);
        pp.query("stress_update_scheme", stress_update_scheme);
        if (fixed_timestep == 1)
        {
            pp.get("timestep", timestep);
        }
        else
        {
            timestep = 0.0;
        }

        pp.query("alpha_pic_flip", alpha_pic_flip);
        pp.query("dt_max_limit", dt_max_limit);
        pp.query("dt_min_limit", dt_min_limit);
        pp.query("CFL", CFL);

        pp.query("applied_strainrate", applied_strainrate);
        applied_strainrate_time = final_time;
        pp.query("applied_strainrate_time", applied_strainrate_time);

        pp.query("order_scheme", order_scheme);
        pp.query("mass_tolerance", mass_tolerance);

        pp.query("levset_output", levset_output);
        pp.query("levset_smoothfactor", levset_smoothfactor);
        pp.query("levset_gridratio", levset_gridratio);

        pp.query("external_loads", external_loads_present);
        if (external_loads_present)
        {
            pp.get("force_slab_lo", force_slab_lo);
            pp.get("force_slab_hi", force_slab_hi);
            pp.get("extforce", extforce);
        }
        pp.query("print_diagnostics", print_diagnostics);
        if (print_diagnostics != 0)
        {
            pp.query("do_calculate_tke_tse", do_calculate_tke_tse);
            pp.query("do_calculate_mwa_velcomp", do_calculate_mwa_velcomp);
            pp.query("do_calculate_mwa_velmag", do_calculate_mwa_velmag);
            pp.query("do_calculate_minmaxpos", do_calculate_minmaxpos);
            pp.query("write_diag_output_time", write_diag_output_time);
        }

        // by default it is periodic
        bclo[0] = BC_SLIPWALL;
        bchi[0] = BC_SLIPWALL;
#if (AMREX_SPACEDIM >= 2)
        bclo[1] = BC_SLIPWALL;
        bchi[1] = BC_SLIPWALL;
#endif
#if (AMREX_SPACEDIM == 3)
        bclo[2] = BC_SLIPWALL;
        bchi[2] = BC_SLIPWALL;
#endif
        pp.queryarr("bc_lower", bclo);
        pp.queryarr("bc_upper", bchi);

#if USE_TEMP
        pp.queryarr("bc_lower_temp", bclo_temp);
        pp.queryarr("bc_upper_temp", bchi_temp);

        pp.queryarr("bc_lower_tempval", bclo_tempval);
        pp.queryarr("bc_upper_tempval", bchi_tempval);
#endif

        pp.query("levelset_bc", levelset_bc);
        pp.query("levelset_wall_mu", levelset_wall_mu);

        pp.query("wall_mu_lo", wall_mu_lo);
        pp.query("wall_mu_hi", wall_mu_hi);

        for (int i = 0; i < AMREX_SPACEDIM * AMREX_SPACEDIM; i++)
        {
            wall_vel_lo[i] = 0.0;
            wall_vel_hi[i] = 0.0;
        }

        pp.query("wall_vel_lo", wall_vel_lo);
        pp.query("wall_vel_hi", wall_vel_hi);
    }

    std::string Convert_arr_string(Array<Real, AMREX_SPACEDIM> arr)
    {
        std::string str;
        str = "[";
        for (int i = 0; i < AMREX_SPACEDIM - 1; i++)
        {
            str = str + std::to_string(arr[i]) + ",";
        }
        str = str + std::to_string(arr[AMREX_SPACEDIM - 1]);
        return (str);
    }

    void PrintSimulationParams()
    {
        // To be completed, Sreejith
        std::string vect_str;
        vect_str = Convert_arr_string(plo);
        amrex::Print() << "\n\tBackground grid physical extents: " << vect_str;
    }
    void ThreeDOF_Solver(amrex::Real dt)
    {
        for (int i = 0; i < no_of_rigidbodies_present; i++)
        {
            if (Rb[i].position_update_method == 1)
            {
                for (int j = 0; j < AMREX_SPACEDIM; j++)
                {
                    Rb[i].velocity[j] +=
                        (Rb[i].force_internal[j] + Rb[i].force_external[j]) /
                        Rb[i].total_mass * dt;
                }
            }
            else
            {
                for (int j = 0; j < AMREX_SPACEDIM; j++)
                {
                    Rb[i].velocity[j] = Rb[i].imposed_velocity[j];
                }
            }
        }
    }
};

#endif
