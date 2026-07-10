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

void MPMParticleContainer::build_material_table(
    const amrex::Vector<MaterialParams> &host_table)
{
    m_material_table.resize(host_table.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_table.begin(),
                     host_table.end(), m_material_table.begin());
}

void MPMParticleContainer::record_material_elastic(int cm, amrex::Real E,
                                                   amrex::Real nu)
{
    if (cm < 0)
        return;
    if (cm >= static_cast<int>(m_host_material_table.size()))
        m_host_material_table.resize(cm + 1);
    m_host_material_table[cm].model = CModel::ELASTIC;
    m_host_material_table[cm].p[ElasticP::E] = E;
    m_host_material_table[cm].p[ElasticP::nu] = nu;
}

void MPMParticleContainer::record_material_fluid(int cm, amrex::Real bulk,
                                                 amrex::Real gama,
                                                 amrex::Real visc)
{
    if (cm < 0)
        return;
    if (cm >= static_cast<int>(m_host_material_table.size()))
        m_host_material_table.resize(cm + 1);
    m_host_material_table[cm].model = CModel::FLUID;
    m_host_material_table[cm].p[FluidP::bulk] = bulk;
    m_host_material_table[cm].p[FluidP::gama] = gama;
    m_host_material_table[cm].p[FluidP::visc] = visc;
}

namespace
{
// Host-side constitutive-model registry (ADR-0001 Phase 3b). Single source of
// truth mapping a model's input name to its id and its parameter names in
// MaterialParams::p[] slot order. Adding a model = one entry here (+ its id in
// CModel, its slot enum, and its device update function). The input parser and
// (later) the generator/IO are driven by this table rather than hard-coding.
struct ModelInfo
{
    const char *name;
    int id;
    std::vector<const char *> param_names; // in slot order
};
const std::vector<ModelInfo> &model_registry()
{
    static const std::vector<ModelInfo> reg = {
        {"elastic", CModel::ELASTIC, {"E", "nu"}},
        {"fluid",
         CModel::FLUID,
         {"Bulk_modulus", "Gama_pressure", "Dynamic_viscosity"}},
        {"johnson_cook",
         CModel::JOHNSON_COOK,
         // slot order must match JCP:: in constitutive_models.H
         {"E", "nu", "JC_A", "JC_B", "JC_n", "JC_C", "JC_m", "JC_eps_dot_0",
          "JC_Tr", "JC_Tm", "JC_chi", "JC_c0", "JC_Salpha", "JC_Gamma0",
          "density"}},
    };
    return reg;
}
} // namespace

bool MPMParticleContainer::build_material_table_from_input()
{
    amrex::ParmParse pp("mpm");
    int nmat = 0;
    pp.query("num_materials", nmat);
    if (nmat <= 0)
        return false; // no material block -> caller falls back

    const auto &reg = model_registry();
    amrex::Vector<MaterialParams> table(nmat);
    for (int m = 0; m < nmat; ++m)
    {
        const std::string prefix = "mpm.material_" + std::to_string(m);
        amrex::ParmParse ppm(prefix.c_str());
        std::string model;
        ppm.get("model", model);

        const ModelInfo *info = nullptr;
        for (const auto &mi : reg)
            if (model == mi.name)
            {
                info = &mi;
                break;
            }
        if (info == nullptr)
            amrex::Abort("Unknown material model '" + model + "' for " + prefix);

        table[m].model = info->id;
        for (std::size_t s = 0; s < info->param_names.size(); ++s)
            ppm.get(info->param_names[s], table[m].p[s]);
    }
    build_material_table(table);
    return true;
}

void MPMParticleContainer::upload_material_table()
{
    // MPI-complete: each rank may have read only a subset of materials. Agree
    // on the table size, then reduce-max each entry (params are homogeneous and
    // non-negative per cm_id, so max recovers the value on ranks that saw it).
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
    build_material_table(m_host_material_table);
}

void MPMParticleContainer::apply_constitutive_model(
    const amrex::Real &dt, amrex::Real applied_strainrate /*=0.0*/)
{
    const int lev = 0;
    auto &plev = GetParticles(lev);

    // Device-resident per-model parameter table (ADR-0001), indexed by cm_id.
    const MaterialParams *mat = m_material_table.dataPtr();

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

                    const int cm = p.idata(intData::constitutive_model);
                    const MaterialParams &mp = mat[cm];
                    if (mp.model == CModel::ELASTIC)
                    {
                        // Elastic solid
                        linear_elastic(strain, stress, mp.p[ElasticP::E],
                                       mp.p[ElasticP::nu]);
                    }
                    else if (mp.model == CModel::FLUID)
                    {
                        // Viscous fluid with approximate EoS
                        const amrex::Real p_inf = 0.0;
                        p.rdata(realData::pressure) =
                            mp.p[FluidP::bulk] *
                                (std::pow(1.0 / p.rdata(realData::jacobian),
                                          mp.p[FluidP::gama]) -
                                 1.0) +
                            p_inf;

                        Newtonian_Fluid(strainrate, stress, mp.p[FluidP::visc],
                                        p.rdata(realData::pressure));
                    }
                    else if (mp.model == CModel::JOHNSON_COOK)
                    {
                        // Deformation gradient storage is DIM x DIM row-major
                        // (stride AMREX_SPACEDIM). Expand into a full row-major
                        // 3x3 (out-of-plane component = 1 for the plane case)
                        // for the 3x3 polar decomposition. In 3D this is the
                        // identity mapping; in 2D it places F correctly instead
                        // of reading a singular matrix.
                        amrex::Real F[9] = {1.0, 0.0, 0.0, 0.0, 1.0,
                                            0.0, 0.0, 0.0, 1.0};
                        for (int r = 0; r < AMREX_SPACEDIM; ++r)
                            for (int c = 0; c < AMREX_SPACEDIM; ++c)
                                F[r * 3 + c] = p.rdata(
                                    realData::deformation_gradient +
                                    r * AMREX_SPACEDIM + c);

                        // Per-particle state from the ISV block.
                        amrex::Real ep = p.rdata(realData::isv + JC_ISV::ep);
                        amrex::Real sdev[NCOMP_TENSOR];
                        for (int c = 0; c < NCOMP_TENSOR; ++c)
                            sdev[c] =
                                p.rdata(realData::isv + JC_ISV::sdev + c);

                        amrex::Real press = 0.0, hsrc = 0.0;
#if USE_TEMP
                        amrex::Real Tcur = p.rdata(realData::temperature);
#else
                        amrex::Real Tcur = mp.p[JCP::Tr];
#endif
                        johnson_cook_stress_update(
                            F, strainrate, sdev, ep, stress, press, hsrc,
                            p.rdata(realData::density), mp.p[JCP::rho0],
                            mp.p[JCP::E], mp.p[JCP::nu], mp.p[JCP::A],
                            mp.p[JCP::B], mp.p[JCP::n], mp.p[JCP::C],
                            mp.p[JCP::m], mp.p[JCP::eps_dot_0], Tcur,
                            mp.p[JCP::Tr], mp.p[JCP::Tm], mp.p[JCP::chi],
                            mp.p[JCP::c0], mp.p[JCP::Salpha], mp.p[JCP::Gamma0],
                            dt);

                        // Persist state.
                        p.rdata(realData::isv + JC_ISV::ep) = ep;
                        for (int c = 0; c < NCOMP_TENSOR; ++c)
                            p.rdata(realData::isv + JC_ISV::sdev + c) = sdev[c];
                        p.rdata(realData::pressure) = press;
#if USE_TEMP
                        p.rdata(realData::heat_source) = hsrc;
#endif
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

    const MaterialParams *mat = m_material_table.dataPtr();

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
                    const int cm = p.idata(intData::constitutive_model);
                    if (mat[cm].model == CModel::ELASTIC)
                    {
                        // Elastic solid: linear operator on delta_strain
                        linear_elastic_delta(delta_strain, delta_stress,
                                             mat[cm].p[ElasticP::E],
                                             mat[cm].p[ElasticP::nu]);
                    }
                    else if (mat[cm].model == CModel::FLUID)
                    {
                        amrex::Abort(
                            "\nDelta strain model for weakly compressible "
                            "fluids not implemented yet.");
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
