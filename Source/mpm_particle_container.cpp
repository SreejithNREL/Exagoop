// clang-format off
#include <mpm_particle_container.H>
#include <interpolants.H>
#include <constitutive_models.H>
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

void MPMParticleContainer::record_new_material_elastic(int material_id, amrex::Real E,
                                                   amrex::Real nu)
{
    if (material_id < 0)
        return;
    if (material_id >= static_cast<int>(m_host_material_table.size()))
        m_host_material_table.resize(material_id + 1);
    m_host_material_table[material_id].model = ConstitutiveModel::ELASTIC;
    m_host_material_table[material_id].p[ElasticP::E] = E;
    m_host_material_table[material_id].p[ElasticP::nu] = nu;
}

void MPMParticleContainer::record_new_material_neohookean(int cm, amrex::Real E,
                                                   amrex::Real nu)
{
    if (cm < 0)
        return;
    if (cm >= static_cast<int>(m_host_material_table.size()))
        m_host_material_table.resize(cm + 1);
    m_host_material_table[cm].model = ConstitutiveModel::NEOHOOKEAN;
    m_host_material_table[cm].p[NeoHookeanP::E] = E;
    m_host_material_table[cm].p[NeoHookeanP::nu] = nu;
}

void MPMParticleContainer::record_new_material_fluid(int cm, amrex::Real bulk,
                                                 amrex::Real gama,
                                                 amrex::Real visc)
{
    if (cm < 0)
        return;
    if (cm >= static_cast<int>(m_host_material_table.size()))
        m_host_material_table.resize(cm + 1);
    m_host_material_table[cm].model = ConstitutiveModel::FLUID;
    m_host_material_table[cm].p[FluidP::bulk] = bulk;
    m_host_material_table[cm].p[FluidP::gama] = gama;
    m_host_material_table[cm].p[FluidP::visc] = visc;
}




bool MPMParticleContainer::build_material_table_from_input()
{
    amrex::ParmParse pp("mpm");
    int nmat = 0;
    pp.query("num_materials", nmat);
    if (nmat <= 0)
        return false;

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
            amrex::Abort("Unknown constitutive model '" + model + "' for " + prefix);

        table[m].model = info->id;
        for (std::size_t s = 0; s < info->param_names.size(); ++s)
            ppm.query(info->param_names[s], table[m].p[s]);
    }
    copy_to_device_material_table(table);
    return true;
}

void MPMParticleContainer::upload_material_table()
{
    int nmat = static_cast<int>(m_host_material_table.size());
#ifdef BL_USE_MPI
    amrex::ParallelDescriptor::ReduceIntMax(nmat);
    m_host_material_table.resize(nmat);
    for (int m = 0; m < nmat; ++m)
    {
        amrex::ParallelDescriptor::ReduceIntMax(m_host_material_table[m].model);
        for (int s = 0; s < MAX_MODEL_PARAMS; ++s)
            amrex::ParallelDescriptor::ReduceRealMax(
                m_host_material_table[m].p[s]);
    }
#endif
    copy_to_device_material_table(m_host_material_table);
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
                    	deformation_gradient[comp] = p.rdata(realData::deformation_gradient + comp);
                    }

                    const int material_idx = p.idata(intData::material_indx);
                    const MaterialParams &mp = mat[material_idx];
                    if (mp.model == ConstitutiveModel::ELASTIC)
                    {                        
                        linear_elastic(strain, stress, mp.p[ElasticP::E],mp.p[ElasticP::nu]);
                    }
                    else if (mp.model == ConstitutiveModel::FLUID)
                    {
                        p.rdata(realData::isv+Fluid_ISV::pressure) = mp.p[FluidP::bulk] * (std::pow(1.0 / p.rdata(realData::jacobian), mp.p[FluidP::gama]) - 1.0) + mp.p[FluidP::p_inf];
                        Newtonian_Fluid(strainrate, stress, mp.p[FluidP::visc], p.rdata(realData::isv+Fluid_ISV::pressure));
                    }
                    else if (mp.model == ConstitutiveModel::NEOHOOKEAN)
                    {                        
                        neo_hookean(stress, deformation_gradient, mp.p[NeoHookeanP::E], mp.p[NeoHookeanP::nu]);
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
                    amrex::Real delta_strain[NCOMP_TENSOR];
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

                    // Accumulate stress with delta contribution
                    for (int c = 0; c < NCOMP_TENSOR; ++c)
                    {
                        p.rdata(realData::stress + c) += delta_stress[c];
                    }
                }
            });
    }
}
