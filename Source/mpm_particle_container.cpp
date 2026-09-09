// clang-format off
#include <mpm_particle_container.H>
#include <interpolants.H>
#include <constitutive_models.H>
#include <fstream>
// clang-format on

using namespace amrex;

/**
 * @brief Applies the constitutive model to all active (phase=0) particles.
 *
 * This routine performs a full stress update using the *total strain*
 * formulation. For each particle:
 *
 *   1. **Accumulates strain** using the current strain‑rate:
 *        ε ← ε + dt * ε̇
 *
 *   2. **Applies externally imposed axial strain‑rate** (if any) in all active
 *      spatial dimensions (XX, YY, ZZ depending on AMREX_SPACEDIM).
 *
 *   3. **Builds local copies** of strain and strain‑rate for the constitutive
 * law.
 *
 *   4. **Evaluates the constitutive model**:
 *        - If constitutive_model = 0 → linear elastic solid
 *        - If constitutive_model = 1 → Newtonian fluid with pressure from
 *          a weakly‑compressible equation of state
 *
 *   5. **Writes back the updated stress tensor** to particle storage.
 *
 * @param[in] dt                   Time step used for strain integration.
 * @param[in] applied_strainrate   Optional externally applied axial
 * strain‑rate.
 *
 * @note Neighbor particles (ghost particles) are also updated because nt = np +
 * ng.
 * @note Only particles with phase = 0 (material points) receive constitutive
 * updates.
 *
 * @return None.
 */

void MPMParticleContainer::copy_to_device_material_table(
    const amrex::Vector<MaterialParams> &host_table)
{
    m_device_material_table.resize(host_table.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_table.begin(),
                     host_table.end(), m_device_material_table.begin());
}

void MPMParticleContainer::build_material_table_from_input()
{
    amrex::ParmParse pp("mpm");
    int nmat = 0;
    pp.get("num_materials", nmat); // aborts if absent
    if (nmat < 1)
        amrex::Abort("mpm.num_materials must be >= 1");

    Validate_Constitutive_Model_Registry();

    const auto &reg = Get_Constitutive_Model_Registry();
    amrex::Vector<MaterialParams> table(nmat);
    for (int m = 0; m < nmat; ++m)
    {
        const std::string prefix = "mpm.material_" + std::to_string(m);
        amrex::ParmParse ppm(prefix.c_str());
        std::string model;
        ppm.get("model", model);

        const ConstitutiveModelInfo *info = nullptr;
        for (const auto &mi : reg)
            if (model == mi.name)
            {
                info = &mi;
                break;
            }
        if (info == nullptr)
            amrex::Abort("Unknown constitutive model '" + model + "' for " +
                         prefix);

        if (!info->kernel_implemented)
            amrex::Abort("Constitutive model '" + model + "' (" + prefix +
                         ") is registered but its stress kernel is not "
                         "implemented on this branch");

        table[m].model = info->id;
        table[m].material_id = m;
        // Required parameters abort if missing; optional ones default to 0.
        for (int s = 0; s < static_cast<int>(info->param_names.size()); ++s)
        {
            if (s < info->n_required)
                ppm.get(info->param_names[s], table[m].p[s]);
            else
                ppm.query(info->param_names[s], table[m].p[s]);
        }
    }
    m_host_material_table = table;
    copy_to_device_material_table(m_host_material_table);
}

int MPMParticleContainer::num_isv_slots_used() const
{
    const auto &reg = Get_Constitutive_Model_Registry();
    int n = 0;
    for (const auto &mp : m_host_material_table)
        for (const auto &mi : reg)
            if (mi.id == mp.model)
                n = std::max(n, static_cast<int>(mi.isv_names.size()));
    return n;
}

void MPMParticleContainer::write_material_table(
    const std::string &filename) const
{
    if (!amrex::ParallelDescriptor::IOProcessor())
        return;

    const auto &reg = Get_Constitutive_Model_Registry();
    std::ofstream ofs(filename);
    if (!ofs)
        amrex::Abort("Could not open " + filename + " for writing");

    ofs << "# ExaGOOP material table (single source of truth: input file)\n"
        << "# Particles carry material_indx; plotfile columns isv_<k> are\n"
        << "# interpreted per material via the isv slot names below.\n"
        << "# Unused isv slots hold 0.\n"
        << "num_materials " << m_host_material_table.size() << "\n"
        << "num_isv_slots " << EXAGOOP_NISV << "\n";

    for (const auto &mp : m_host_material_table)
    {
        const ConstitutiveModelInfo *info = nullptr;
        for (const auto &mi : reg)
            if (mi.id == mp.model)
            {
                info = &mi;
                break;
            }
        ofs << "\nmaterial " << mp.material_id << "\n"
            << "  model " << (info ? info->name : "unknown") << "\n";
        if (info)
        {
            ofs << "  params";
            for (std::size_t k = 0; k < info->param_names.size(); ++k)
                ofs << " " << info->param_names[k] << "=" << mp.p[k];
            ofs << "\n  isv";
            for (std::size_t k = 0; k < info->isv_names.size(); ++k)
                ofs << " isv_" << k << "=" << info->isv_names[k];
            ofs << "\n";
        }
    }
}

void MPMParticleContainer::validate_material_indices()
{
    const int nmat = num_materials();
    const int lev = 0;
    using PType = typename MPMParticleContainer::SuperParticleType;

    int imin = amrex::ReduceMin(
        *this, lev,
        [=] AMREX_GPU_HOST_DEVICE(const PType &p) noexcept -> int
        {
            return (p.idata(intData::phase) == 0)
                       ? p.idata(intData::material_indx)
                       : 0;
        });
    int imax = amrex::ReduceMax(
        *this, lev,
        [=] AMREX_GPU_HOST_DEVICE(const PType &p) noexcept -> int
        {
            return (p.idata(intData::phase) == 0)
                       ? p.idata(intData::material_indx)
                       : 0;
        });
    amrex::ParallelDescriptor::ReduceIntMin(imin);
    amrex::ParallelDescriptor::ReduceIntMax(imax);

    if (imin < 0 || imax >= nmat)
        amrex::Abort("Particle material index out of range: found [" +
                     std::to_string(imin) + ", " + std::to_string(imax) +
                     "] but mpm.num_materials = " + std::to_string(nmat));
}

void MPMParticleContainer::apply_constitutive_model(
    const amrex::Real &dt, amrex::Real applied_strainrate /*=0.0*/)
{
    const int lev = 0;
    auto &plev = GetParticles(lev);

    const MaterialParams *mat = m_device_material_table.dataPtr();

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        int gid = mfi.index();
        int tid = mfi.LocalTileIndex();
        auto index = std::make_pair(gid, tid);

        auto &ptile = plev[index];
        auto &aos = ptile.GetArrayOfStructs();

        const int nt = aos.numRealParticles();

        ParticleType *pstruct = aos().dataPtr();

        amrex::ParallelFor(
            nt,
            [=] AMREX_GPU_DEVICE(int i) noexcept
            {
                ParticleType &p = pstruct[i];
                if (p.idata(intData::phase) == 0)
                {
                    amrex::Real strainrate[NCOMP_TENSOR];
                    amrex::Real strain[NCOMP_TENSOR];
                    amrex::Real stress[NCOMP_TENSOR];
                    amrex::Real deformation_gradient[NCOMP_FULLTENSOR];

                    // Update strain from strainrate
                    for (int d = 0; d < NCOMP_TENSOR; ++d)
                    {
                        p.rdata(realData::strain + d) +=
                            dt * p.rdata(realData::strainrate + d);
                    }

#if (AMREX_SPACEDIM >= 1)
                    p.rdata(realData::strain + XX) += dt * applied_strainrate;
#endif
#if (AMREX_SPACEDIM >= 2)
                    p.rdata(realData::strain + YY) += dt * applied_strainrate;
#endif
#if (AMREX_SPACEDIM == 3)
                    p.rdata(realData::strain + ZZ) += dt * applied_strainrate;
#endif

                    // Copy strain/strainrate into local arrays
                    for (int d = 0; d < NCOMP_TENSOR; ++d)
                    {
                        strainrate[d] = p.rdata(realData::strainrate + d);
                        strain[d] = p.rdata(realData::strain + d);
                    }

                    for (int comp = 0; comp < NCOMP_FULLTENSOR; ++comp)
                    {
                        deformation_gradient[comp] =
                            p.rdata(realData::deformation_gradient + comp);
                    }

                    const int material_idx = p.idata(intData::material_indx);
                    const MaterialParams &mp = mat[material_idx];
                    if (mp.model == ConstitutiveModel::ELASTIC)
                    {
                        linear_elastic(strain, stress, mp.p[ElasticP::E],
                                       mp.p[ElasticP::nu]);
                    }
                    else if (mp.model == ConstitutiveModel::FLUID)
                    {
                        // Weakly compressible EOS: p = K[(1/J)^gamma - 1] +
                        // p_inf
                        amrex::Real &pres =
                            p.rdata(isv_slot(Fluid_ISV::pressure));
                        pres = mp.p[FluidP::bulk] *
                                   (std::pow(1.0 / p.rdata(realData::jacobian),
                                             mp.p[FluidP::gama]) -
                                    1.0) +
                               mp.p[FluidP::p_inf];
                        Newtonian_Fluid(strainrate, stress, mp.p[FluidP::visc],
                                        pres);
                    }
                    else if (mp.model == ConstitutiveModel::NEOHOOKEAN)
                    {
                        neo_hookean(stress, deformation_gradient,
                                    mp.p[NeoHookeanP::E],
                                    mp.p[NeoHookeanP::nu]);
                    }
                    else if (mp.model == ConstitutiveModel::JOHNSON_COOK)
                    {
                        amrex::Real F[9] = {1.0, 0.0, 0.0, 0.0, 1.0,
                                            0.0, 0.0, 0.0, 1.0};
                        for (int r = 0; r < AMREX_SPACEDIM; ++r)
                            for (int c = 0; c < AMREX_SPACEDIM; ++c)
                                F[r * 3 + c] = p.rdata(
                                    realData::deformation_gradient +
                                    r * AMREX_SPACEDIM + c);

                        // Per-particle state from the ISV block.
                        amrex::Real ep = p.rdata(isv_slot(JC_ISV::ep));
                        amrex::Real dmg = p.rdata(isv_slot(JC_ISV::damage));
                        amrex::Real sdev[NCOMP_TENSOR];
                        for (int c = 0; c < NCOMP_TENSOR; ++c)
                            sdev[c] = p.rdata(isv_slot(JC_ISV::sdev + c));

                        amrex::Real press = 0.0, hsrc = 0.0;
#if USE_TEMP
                        const amrex::Real Tcur =
                            p.rdata(realData::temperature);
#else
                        const amrex::Real Tcur = mp.p[JCP::Tr];
#endif
                        johnson_cook_stress_update(
                            F, strainrate, sdev, ep, stress, press, hsrc,
                            p.rdata(realData::density), mp.p[JCP::rho0],
                            mp.p[JCP::E], mp.p[JCP::nu], mp.p[JCP::A],
                            mp.p[JCP::B], mp.p[JCP::n], mp.p[JCP::C],
                            mp.p[JCP::m], mp.p[JCP::eps_dot_0], Tcur,
                            mp.p[JCP::Tr], mp.p[JCP::Tm], mp.p[JCP::chi],
                            mp.p[JCP::c0], mp.p[JCP::Salpha],
                            mp.p[JCP::Gamma0], mp.p[JCP::D1], mp.p[JCP::D2],
                            mp.p[JCP::D3], mp.p[JCP::D4], mp.p[JCP::D5], dmg,
                            dt);

                        // Persist state.
                        p.rdata(isv_slot(JC_ISV::ep)) = ep;
                        p.rdata(isv_slot(JC_ISV::damage)) = dmg;
                        for (int c = 0; c < NCOMP_TENSOR; ++c)
                            p.rdata(isv_slot(JC_ISV::sdev + c)) = sdev[c];
                        p.rdata(isv_slot(JC_ISV::pressure)) = press;
#if USE_TEMP
                        p.rdata(realData::heat_source) = hsrc;
#endif
                    }
                    else
                    {
                        for (int d = 0; d < NCOMP_TENSOR; ++d)
                            stress[d] = 0.0;
                    }

                    // Write back stress
                    for (int d = 0; d < NCOMP_TENSOR; ++d)
                    {
                        p.rdata(realData::stress + d) = stress[d];
                    }
                }
            });
    }
}

/**
 * @brief Applies the constitutive model using an incremental (delta)
 * formulation.
 *
 * This routine performs a stress update based on *incremental strain*:
 *
 *   1. **Accumulates total strain**:
 *        ε ← ε + dt * ε̇
 *
 *   2. **Builds delta_strain**:
 *        Δε = dt * ε̇
 *      and optionally adds externally applied axial increments.
 *
 *   3. **Evaluates the constitutive model incrementally**:
 *        - If constitutive_model = 0 → linear elastic solid:
 *              Δσ = C : Δε
 *        - If constitutive_model = 1 → (not implemented)
 *
 *   4. **Accumulates stress**:
 *        σ ← σ + Δσ
 *
 * This formulation is useful for implicit or incremental constitutive updates
 * where only the strain increment is needed.
 *
 * @param[in] dt                   Time step used to compute Δε.
 * @param[in] applied_strainrate   Optional externally applied axial
 * strain‑rate.
 *
 * @note Only particles with phase = 0 (material points) are updated.
 * @note Weakly compressible fluid delta‑model is not implemented.
 *
 * @return None.
 */

void MPMParticleContainer::apply_constitutive_model_delta(
    const amrex::Real &dt, amrex::Real applied_strainrate /*= 0.0*/)
{
    const int lev = 0;
    auto &plev = GetParticles(lev);

    const MaterialParams *mat = m_device_material_table.dataPtr();

    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        const int gid = mfi.index();
        const int tid = mfi.LocalTileIndex();
        auto index = std::make_pair(gid, tid);

        auto &ptile = plev[index];
        auto &aos = ptile.GetArrayOfStructs();

        const int nt = aos.numRealParticles();

        ParticleType *pstruct = aos().dataPtr();

        amrex::ParallelFor(
            nt,
            [=] AMREX_GPU_DEVICE(int i) noexcept
            {
                ParticleType &p = pstruct[i];

                if (p.idata(intData::phase) == 0)
                {
                    amrex::Real delta_strain[NCOMP_TENSOR] = {};
                    amrex::Real delta_stress[NCOMP_TENSOR];

                    // Accumulate strain from current strainrate
                    for (int c = 0; c < NCOMP_TENSOR; ++c)
                    {
                        p.rdata(realData::strain + c) +=
                            dt * p.rdata(realData::strainrate + c);
                    }

                // Apply external axial strainrate in active spatial dims
#if (AMREX_SPACEDIM >= 1)
                    p.rdata(realData::strain + XX) += dt * applied_strainrate;
#endif
#if (AMREX_SPACEDIM >= 2)
                    p.rdata(realData::strain + YY) += dt * applied_strainrate;
#endif
#if (AMREX_SPACEDIM >= 3)
                    p.rdata(realData::strain + ZZ) += dt * applied_strainrate;
#endif

                    // Build delta_strain from strainrate
                    for (int c = 0; c < NCOMP_TENSOR; ++c)
                    {
                        delta_strain[c] =
                            dt * p.rdata(realData::strainrate + c);
                    }

                // Add external axial delta in active dims
#if (AMREX_SPACEDIM >= 1)
                    delta_strain[XX] += dt * applied_strainrate;
#endif
#if (AMREX_SPACEDIM >= 2)
                    delta_strain[YY] += dt * applied_strainrate;
#endif
#if (AMREX_SPACEDIM >= 3)
                    delta_strain[ZZ] += dt * applied_strainrate;
#endif

                    // Constitutive response for delta update
                    const int material_idx = p.idata(intData::material_indx);
                    const MaterialParams &mp = mat[material_idx];
                    if (mp.model == ConstitutiveModel::ELASTIC)
                    {
                        // Elastic solid: linear operator on delta_strain
                        linear_elastic_delta(delta_strain, delta_stress,
                                             mp.p[ElasticP::E],
                                             mp.p[ElasticP::nu]);
                    }
                    else if (mp.model == ConstitutiveModel::FLUID)
                    {
                        amrex::Abort(
                            "\nDelta strain model for weakly compressible "
                            "fluids not implemented yet.");
                    }
                    else if (mp.model == ConstitutiveModel::NEOHOOKEAN)
                    {
                        amrex::Abort("\nDelta strain model for neo hookean "
                                     "model not implemented yet.");
                    }
                    else if (mp.model == ConstitutiveModel::JOHNSON_COOK)
                    {
                        amrex::Abort("\nDelta strain model for Johnson-Cook "
                                     "not implemented (use the total-strain "
                                     "path).");
                    }
                    else
                    {
                        for (int c = 0; c < NCOMP_TENSOR; ++c)
                            delta_stress[c] = 0.0;
                    }

                    // Accumulate stress with delta contribution
                    for (int c = 0; c < NCOMP_TENSOR; ++c)
                    {
                        p.rdata(realData::stress + c) += delta_stress[c];
                    }
                }
            });
    }
}
