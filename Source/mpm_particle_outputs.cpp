// clang-format off
#include <mpm_particle_container.H>
#include <interpolants.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_AmrMesh.H>
// clang-format on

void MPMParticleContainer::update_phase_field(MultiFab &phasedata,
                                              int refratio,
                                              amrex::Real smoothfactor)
{
    const int ng_phase = 3;
    const int lev = 0;

    const Geometry &geom = Geom(lev);
    auto &plev = GetParticles(lev);

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxi = geom.InvCellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto phi = geom.ProbHiArray();

    // Compute a domain-scale max distance in a dimension-aware way
    amrex::Real maxdist = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        const amrex::Real Ld = phi[d] - plo[d];
        maxdist += Ld * Ld;
    }
    maxdist = 10.0 * std::sqrt(maxdist);
    phasedata.setVal(maxdist, ng_phase);

    // Refine domain and scale cell metrics for the phase grid
    Box domain = geom.Domain();
    domain.refine(refratio);

    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        dxi[d] *= refratio;
        dx[d] /= refratio;
    }

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        Box box = mfi.tilebox();
        Box &refbox = box.refine(refratio);
        const Box &refboxgrow = amrex::grow(refbox, ng_phase);

        const int gid = mfi.index();
        const int tid = mfi.LocalTileIndex();
        auto index = std::make_pair(gid, tid);

        auto &ptile = plev[index];
        auto &aos = ptile.GetArrayOfStructs();

        const int np = aos.numRealParticles();
        Array4<amrex::Real> phase_data_arr = phasedata.array(mfi);
        ParticleType *pstruct = aos().dataPtr();

        amrex::ParallelFor(
            np,
            [=] AMREX_GPU_DEVICE(int ip) noexcept
            {
                ParticleType &p = pstruct[ip];

                // Particle cell index at refined resolution
                IntVect iv = getParticleCell(p, plo, dxi, domain);

                // Neighborhood stencil half-width
                const int hw = 3;

#if (AMREX_SPACEDIM == 1)
                for (int l = -hw; l <= hw; ++l)
                {
                    const IntVect ivlocal(iv[0] + l);
                    if (refboxgrow.contains(ivlocal))
                    {
                        amrex::Real xp[1] = {p.pos(0)};
                        amrex::Real xi[1] = {plo[0] +
                                             (ivlocal[0] + HALF_CONST) * dx[0]};

                        amrex::Real dist = levelset(
                            xi, xp, smoothfactor * p.rdata(realData::radius),
                            -1, maxdist);
                        amrex::Gpu::Atomic::Min(&phase_data_arr(ivlocal), dist);
                    }
                }
#elif (AMREX_SPACEDIM == 2)
                for (int m = -hw; m <= hw; ++m)
                {
                    for (int l = -hw; l <= hw; ++l)
                    {
                        const IntVect ivlocal(iv[0] + l, iv[1] + m);
                        if (refboxgrow.contains(ivlocal))
                        {
                            amrex::Real xp[2] = {p.pos(0), p.pos(1)};
                            amrex::Real xi[2] = {
                                plo[0] + (ivlocal[0] + HALF_CONST) * dx[0],
                                plo[1] + (ivlocal[1] + HALF_CONST) * dx[1]};

                            amrex::Real dist = levelset(
                                xi, xp,
                                smoothfactor * p.rdata(realData::radius), -1,
                                maxdist);
                            amrex::Gpu::Atomic::Min(&phase_data_arr(ivlocal),
                                                    dist);
                        }
                    }
                }
#else
                for (int n = -hw; n <= hw; ++n)
                {
                    for (int m = -hw; m <= hw; ++m)
                    {
                        for (int l = -hw; l <= hw; ++l)
                        {
                            const IntVect ivlocal(
                                AMREX_D_DECL(iv[0] + l, iv[1] + m, iv[2] + n));
                            if (refboxgrow.contains(ivlocal))
                            {
                                amrex::Real xp[3] = {p.pos(0), p.pos(1),
                                                     p.pos(2)};
                                amrex::Real xi[3] = {
                                    plo[0] + (ivlocal[0] + HALF_CONST) * dx[0],
                                    plo[1] + (ivlocal[1] + HALF_CONST) * dx[1],
                                    plo[2] + (ivlocal[2] + HALF_CONST) * dx[2]};

                                amrex::Real dist = levelset(
                                    xi, xp,
                                    smoothfactor * p.rdata(realData::radius),
                                    -1, maxdist);
                                amrex::Gpu::Atomic::Min(
                                    &phase_data_arr(ivlocal), dist);
                            }
                        }
                    }
                }
#endif
            });
    }

    // Optional boundary min-reduction (AMReX lacks min boundary sum),
    // but with sufficiently larger levset grid than particle radii
    // (dx > 3 * radius), this often isn’t required.
    // phasedata.SumBoundary(geom.periodicity());
}

void MPMParticleContainer::writeAsciiFiles(std::string prefix_particlefilename,
                                           int num_of_digits_in_filenames,
                                           amrex::Real time)
{
    std::ostringstream oss;
    oss << prefix_particlefilename << "_t" << std::fixed
        << std::setprecision(num_of_digits_in_filenames) << time;
    WriteAsciiFile(oss.str());
}

void MPMParticleContainer::writeParticles(std::string prefix_particlefilename,
                                          int num_of_digits_in_filenames,
                                          const int n)
{
    BL_PROFILE("MPMParticleContainer::writeParticles");

    const std::string &pltfile = amrex::Concatenate(prefix_particlefilename, n,
                                                    num_of_digits_in_filenames);

    Vector<int> writeflags_real(realData::count, 1);
    Vector<int> writeflags_int(intData::count, 0);

    Vector<std::string> real_data_names;
    Vector<std::string> int_data_names;

    // Always include radius
    real_data_names.push_back("radius");

    real_data_names.push_back("xvel");
    real_data_names.push_back("yvel");
    real_data_names.push_back("zvel");

    real_data_names.push_back("xvel_prime");
    real_data_names.push_back("yvel_prime");
    real_data_names.push_back("zvel_prime");

    // Strainrate, strain, stress tensors (NCOMP_TENSOR entries)
    for (int c = 0; c < NCOMP_TENSOR; ++c)
    {
        real_data_names.push_back(amrex::Concatenate("strainrate_", c, 1));
    }
    for (int c = 0; c < NCOMP_TENSOR; ++c)
    {
        real_data_names.push_back(amrex::Concatenate("strain_", c, 1));
    }
    for (int c = 0; c < NCOMP_TENSOR; ++c)
    {
        real_data_names.push_back(amrex::Concatenate("stress_", c, 1));
    }

    // Deformation gradient (NCOMP_FULLTENSOR entries)
    for (int c = 0; c < NCOMP_FULLTENSOR; ++c)
    {
        real_data_names.push_back(
            amrex::Concatenate("deformation_gradient_", c, 1));
    }

    // Scalar material properties
    real_data_names.push_back("volume");
    real_data_names.push_back("mass");
    real_data_names.push_back("density");
    real_data_names.push_back("jacobian");
    real_data_names.push_back("pressure");
    real_data_names.push_back("vol_init");
    real_data_names.push_back("E");
    real_data_names.push_back("nu");
    real_data_names.push_back("Bulk_modulus");
    real_data_names.push_back("Gama_pressure");
    real_data_names.push_back("Dynamic_viscosity");
    real_data_names.push_back("yacceleration");

#if USE_TEMP
    // Thermal fields
    real_data_names.push_back("temperature");
    real_data_names.push_back("specific_heat");
    real_data_names.push_back("thermal_conductivity");
    for (int d = 0; d < 3; ++d)
    {
        real_data_names.push_back(amrex::Concatenate("heat_flux_", d, 1));
    }
    real_data_names.push_back("heat_source");

#endif

    // Integer data fields
    int_data_names.push_back("phase");
    int_data_names.push_back("rigid_body_id");
    int_data_names.push_back("constitutive_model");

    // Flags: mark which fields to write
    writeflags_int[intData::phase] = 1;
    writeflags_int[intData::constitutive_model] = 1;
    writeflags_int[intData::rigid_body_id] = 1;

    writeflags_real[realData::radius] = 1;
    // Dimension‑aware velocity flags
    writeflags_real[realData::xvel] = 1;
    writeflags_real[realData::yvel] = 1;
    writeflags_real[realData::zvel] = 1;

    writeflags_real[realData::mass] = 1;
    writeflags_real[realData::jacobian] = 1;
    writeflags_real[realData::pressure] = 1;
    writeflags_real[realData::vol_init] = 1;

    // Optional material properties (disabled by default)
    writeflags_real[realData::E] = 0;
    writeflags_real[realData::nu] = 0;
    writeflags_real[realData::Bulk_modulus] = 0;
    writeflags_real[realData::Gama_pressure] = 0;
    writeflags_real[realData::Dynamic_viscosity] = 0;

#if USE_TEMP
    writeflags_real[realData::temperature] = 1;
    writeflags_real[realData::specific_heat] = 0;
    writeflags_real[realData::thermal_conductivity] = 0;
    writeflags_real[realData::heat_source] = 0;
    for (int d = 0; d < 3; ++d)
    {
        writeflags_real[realData::heat_flux + d] = 1;
    }
#endif

    // Write plotfile
    WritePlotFile(pltfile, "particles", writeflags_real, writeflags_int,
                  real_data_names, int_data_names);
}

void MPMParticleContainer::WriteHeader(const std::string &name,
                                       bool is_checkpoint,
                                       amrex::Real cur_time,
                                       int nstep,
                                       int EB_generate_max_level,
                                       int output_it) const
{
    if (ParallelDescriptor::IOProcessor())
    {
        std::string HeaderFileName(name + "/Header");
        VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
        std::ofstream HeaderFile;

        HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
        HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out |
                                                    std::ofstream::trunc |
                                                    std::ofstream::binary);

        if (!HeaderFile.good())
        {
            amrex::FileOpenFailed(HeaderFileName);
        }

        HeaderFile.precision(17);
        HeaderFile << (is_checkpoint ? "Checkpoint version: 1\n"
                                     : "HyperCLaw-V1.1\n");
        HeaderFile << nstep << "\n";
        HeaderFile << output_it << "\n";
#ifdef AMREX_USE_EB
        HeaderFile << EB_generate_max_level << "\n";
#endif
        HeaderFile << cur_time << "\n";
    }
}

void MPMParticleContainer::writeCheckpointFile(
    std::string prefix_particlefilename,
    int num_of_digits_in_filenames,
    amrex::Real cur_time,
    int nstep,
    int output_it)
{
    BL_PROFILE("MPMParticleContainer::writeCheckpointFile");
    const int finest_level = 0;
    const int EB_generate_max_level = 0;
    const std::string &checkpointname = amrex::Concatenate(
        prefix_particlefilename, output_it, num_of_digits_in_filenames);

    amrex::PreBuildDirectorHierarchy(checkpointname, "Level_", finest_level + 1,
                                     true);

    WriteHeader(checkpointname, /*is_checkpoint=*/true, cur_time, nstep,
                EB_generate_max_level, output_it);

    amrex::Vector<std::string> real_data_names;

    // Basic scalars
    real_data_names.push_back("radius");

    // Dimension‑aware velocities
    real_data_names.push_back("radius");
    real_data_names.push_back("xvel");
    real_data_names.push_back("yvel");
    real_data_names.push_back("zvel");
    real_data_names.push_back("xvel_prime");
    real_data_names.push_back("yvel_prime");
    real_data_names.push_back("zvel_prime");

    // Strainrate, strain, stress tensors
    for (int c = 0; c < NCOMP_TENSOR; ++c)
    {
        real_data_names.push_back(amrex::Concatenate("strainrate_", c, 1));
    }
    for (int c = 0; c < NCOMP_TENSOR; ++c)
    {
        real_data_names.push_back(amrex::Concatenate("strain_", c, 1));
    }
    for (int c = 0; c < NCOMP_TENSOR; ++c)
    {
        real_data_names.push_back(amrex::Concatenate("stress_", c, 1));
    }

    // Deformation gradient
    for (int c = 0; c < NCOMP_FULLTENSOR; ++c)
    {
        real_data_names.push_back(
            amrex::Concatenate("deformation_gradient_", c, 1));
    }

    // Material properties
    real_data_names.push_back("volume");
    real_data_names.push_back("mass");
    real_data_names.push_back("density");
    real_data_names.push_back("jacobian");
    real_data_names.push_back("pressure");
    real_data_names.push_back("vol_init");
    real_data_names.push_back("E");
    real_data_names.push_back("nu");
    real_data_names.push_back("Bulk_modulus");
    real_data_names.push_back("Gama_pressure");
    real_data_names.push_back("Dynamic_viscosity");
    real_data_names.push_back("yacceleration");

#if USE_TEMP
    real_data_names.push_back("temperature");
    real_data_names.push_back("specific_heat");
    real_data_names.push_back("thermal_conductivity");
    real_data_names.push_back("heat_source");
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        real_data_names.push_back(amrex::Concatenate("heat_flux_", d, 1));
    }
#endif

    amrex::Vector<std::string> int_data_names;
    int_data_names.push_back("phase");
    int_data_names.push_back("constitutive_model");
    int_data_names.push_back("rigid_body_id");

    Checkpoint(checkpointname, "particles", /*is_checkpoint=*/true,
               real_data_names, int_data_names);
}

void GotoNextLine(std::istream &is)
{
    constexpr std::streamsize bl_ignore_max{100000};
    is.ignore(bl_ignore_max, '\n');
}

void MPMParticleContainer::readCheckpointFile(std::string &restart_chkfile,
                                              int &nstep,
                                              double &cur_time,
                                              int &output_it)
{
    BL_PROFILE("MPMParticleContainer::readCheckpointFile");

    amrex::Print() << "Restarting from checkpoint " << restart_chkfile << "\n";

    std::string File(restart_chkfile + "/Header");
    VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line;

    // Title line
    std::getline(is, line);

    // Step count
    is >> nstep;
    GotoNextLine(is);

    // Output number
    is >> output_it;
    GotoNextLine(is);

#ifdef AMREX_USE_EB
    std::getline(is, line);
    if (line.find('.') != std::string::npos)
    {
        cur_time = std::stod(line);
    }
    else
    {
        is >> cur_time;
        GotoNextLine(is);
    }
#else
    is >> cur_time;
    GotoNextLine(is);
#endif

    Restart(restart_chkfile, "particles", true);

    if (m_verbose)
    {
        amrex::Print() << "Restart complete" << std::endl;
    }
}
