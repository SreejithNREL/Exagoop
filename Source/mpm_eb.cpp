// clang-format off
/**
 * @file mpm_eb.cpp
 *
 * @brief Initializes EB geometry and nodal level-set MultiFabs for ExaGOOP.
 *
 * Supports multiple named level set bodies, each with an independent signed-distance
 * MultiFab and refinement factor. Three options are provided now: a UDF (user-defined
 * function, STL file and amrex built in geometry. Check documentation for how to
 * specify these in input files.
 */
// clang-format on

#define EXAGOOP_INCLUDE_EB2_IF

#include <mpm_eb.H>

#if USE_EB
#include <mpm_udf_loader.H>
#include <AMReX_MultiFabUtil.H>
#endif

#include <AMReX_PlotFileUtil.H>

#if USE_EB
namespace mpm_ebtools
{

std::vector<LevelSetBody> ls_bodies;
EBFArrayBoxFactory *ebfactory = nullptr;
bool using_levelset_geometry = false;

static int coarsening_level_for_refinement(int ls_ref)
{
	//Returns the amrex refinement level: 1->0, 2->1,4->2
    int level = 0;
    if (ls_ref > 1)
    {
        int tmp = ls_ref;
        while (tmp >>= 1)
            ++level;
    }
    return level;
}

static Geometry refined_geom(const Geometry &geom, int ls_ref)
{
	//returns a refined (ls_ref) geometry
    Box dom_ls = geom.Domain();
    dom_ls.refine(ls_ref);
    return Geometry(dom_ls);
}

/**
 * @brief Rebuilds the global EBFArrayBoxFactory from the current top
 *        EB2::IndexSpace.  Deletes any previously allocated factory first.
 */
static void build_factory(const Geometry &geom,
                           const BoxArray &ba,
                           const DistributionMapping &dm,
                           int nghost)
{
    delete ebfactory;
    ebfactory = nullptr;

    const EB2::IndexSpace &ebis = EB2::IndexSpace::top();
    const EB2::Level &eblev = ebis.getLevel(geom);

    ebfactory = new EBFArrayBoxFactory(
        eblev, geom, ba, dm, {nghost, nghost, nghost}, EBSupport::full);
}

/**
 * @brief Allocates a nodal MultiFab for one body's signed-distance field.
 *        The returned pointer is owned by the caller (stored in LevelSetBody).
 */
static MultiFab *allocate_body_lsphi(const BoxArray &ba,
                                      const DistributionMapping &dm,
                                      int nghost,
                                      int ls_ref)
{
    BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
    ls_ba.refine(ls_ref);

    MultiFab *mf = new MultiFab;
    mf->define(ls_ba, dm, /*ncomp=*/1, nghost);
    return mf;
}

// ---------------------------------------------------------------
// Option 1 — UDF level set
// ---------------------------------------------------------------

/**
 * @brief Builds EB and fills lsphi from a UDF shared library.
 *
 * @param pp_prefix  ParmParse prefix for this body's keys (e.g. "sphere_1"
 *                   or "eb2" for the legacy single-body path).
 */
static MultiFab *build_udf_levelset(const std::string &name,
                                     const std::string &pp_prefix,
                                     const Geometry &geom,
                                     const BoxArray &ba,
                                     const DistributionMapping &dm,
                                     int nghost,
                                     int ls_ref)
{
    std::string so_file;
    amrex::ParmParse pp(pp_prefix);
    pp.get("udf_so_file", so_file);

    amrex::Print() << "\n\tBody '" << name << "' — UDF: " << so_file << "\n";

    UDFLoader loader(so_file);
    UDFImplicitFunction udf_if(loader);

    MultiFab *lsphi_out = nullptr;
    build_udf_eb(udf_if, geom, ba, dm, nghost, ls_ref, lsphi_out, ebfactory);

    Geometry geom_ls = refined_geom(geom, ls_ref);
    const auto plo = geom_ls.ProbLoArray();
    const auto dx_ls = geom_ls.CellSizeArray();

    for (MFIter mfi(*lsphi_out); mfi.isValid(); ++mfi)
    {
        auto arr = lsphi_out->array(mfi);
        const Box &bx = mfi.fabbox();

        amrex::LoopOnCpu(bx,
                         [&](int i, int j, int k)
                         {
                             amrex::RealArray p;
                             p[0] = plo[0] + i * dx_ls[0];
#if (AMREX_SPACEDIM >= 2)
                             p[1] = plo[1] + j * dx_ls[1];
#endif
#if (AMREX_SPACEDIM == 3)
                             p[2] = plo[2] + k * dx_ls[2];
#endif
                             arr(i, j, k) = udf_if(p);
                         });
    }
    lsphi_out->FillBoundary(geom_ls.periodicity());
    return lsphi_out;
}

// ---------------------------------------------------------------
// Option 2 — user-provied stl file
// ---------------------------------------------------------------

/**
 * @brief Builds EB and fills lsphi from an STL surface mesh.
 */
static MultiFab *build_stl_levelset(const std::string &name,
                                     const std::string &pp_prefix,
                                     const Geometry &geom,
                                     const BoxArray &ba,
                                     const DistributionMapping &dm,
                                     int nghost,
                                     int ls_ref)
{
#ifndef AMREX_USE_EB
    amrex::Abort("build_stl_levelset: AMReX was not compiled with EB support");
    return nullptr;
#else
    std::string stl_file;
    amrex::ParmParse pp(pp_prefix);
    pp.get("stl_file", stl_file);

    amrex::Print() << "\n[EB] Body '" << name << "' — STL: " << stl_file << "\n";

    Geometry geom_ls = refined_geom(geom, ls_ref);
    int req_coarsen = coarsening_level_for_refinement(ls_ref);

    amrex::EB2::Build(geom_ls, req_coarsen, 10);

    build_factory(geom, ba, dm, nghost);

    MultiFab *lsphi_out = allocate_body_lsphi(ba, dm, nghost, ls_ref);

    const EB2::IndexSpace &ebis = EB2::IndexSpace::top();
    const EB2::Level &lslev = ebis.getLevel(geom_ls);

    amrex::FillSignedDistance(*lsphi_out, lslev, *ebfactory, ls_ref);
    lsphi_out->FillBoundary(geom_ls.periodicity());
    return lsphi_out;
#endif
}

// ---------------------------------------------------------------
// Option 3: AMREX built in geom
// ---------------------------------------------------------------

/**
 * @brief Builds EB and fills lsphi from AMReX built-in shapes.
 *
 * For named bodies (pp_prefix != "eb2"), common shapes (sphere, plane,
 * cylinder) are constructed from per-body ParmParse keys so that each body
 * can have independent parameters.
 *
 * "wedge_hopper" and any unrecognised geom_type fall back to
 * EB2::Build(geom_ls, ...) which reads from the eb2.* namespace; this is
 * the legacy behaviour and is only correct for single-body simulations.
 */
static MultiFab *build_analytic_levelset(const std::string &name,
                                          const std::string &pp_prefix,
                                          const std::string &geom_type,
                                          const Geometry &geom,
                                          const BoxArray &ba,
                                          const DistributionMapping &dm,
                                          int nghost,
                                          int ls_ref)
{
    amrex::Print() << "[EB] Body '" << name << "' — analytic geometry: "
                   << geom_type << "\n";

    Geometry geom_ls = refined_geom(geom, ls_ref);
    int req_coarsen = coarsening_level_for_refinement(ls_ref);

    amrex::ParmParse pp(pp_prefix);

    if (geom_type == "sphere")
    {
        amrex::Real radius = 1.0;
        pp.get("sphere_radius", radius);

        std::vector<amrex::Real> center_v(AMREX_SPACEDIM, 0.5);
        pp.getarr("sphere_center", center_v);
        amrex::RealArray center;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) center[d] = center_v[d];

        bool has_fluid_inside = false;
        pp.query("sphere_has_fluid_inside", has_fluid_inside);

        EB2::SphereIF sphere(radius, center, has_fluid_inside);
        auto shop = EB2::makeShop(sphere);
        EB2::Build(shop, geom_ls, req_coarsen, 10);
    }
    else if (geom_type == "plane")
    {
        std::vector<amrex::Real> point_v(AMREX_SPACEDIM, 0.0);
        std::vector<amrex::Real> normal_v(AMREX_SPACEDIM, 0.0);
        normal_v[1] = 1.0;
        pp.getarr("plane_point", point_v);
        pp.getarr("plane_normal", normal_v);
        amrex::RealArray point, normal;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            point[d]  = point_v[d];
            normal[d] = normal_v[d];
        }

        bool has_fluid_inside = false;
        pp.query("plane_has_fluid_inside", has_fluid_inside);

        EB2::PlaneIF plane(point, normal, has_fluid_inside);
        auto shop = EB2::makeShop(plane);
        EB2::Build(shop, geom_ls, req_coarsen, 10);
    }
    else if (geom_type == "cylinder")
    {
        amrex::Real radius = 1.0;
        amrex::Real height = 1.0;
        int direction = 2;

        pp.get("cylinder_radius", radius);
        pp.query("cylinder_height", height);
        pp.query("cylinder_direction", direction);

        std::vector<amrex::Real> center_v(AMREX_SPACEDIM, 0.5);
        pp.getarr("cylinder_center", center_v);
        amrex::RealArray center;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) center[d] = center_v[d];

        bool has_fluid_inside = false;
        pp.query("cylinder_has_fluid_inside", has_fluid_inside);

        EB2::CylinderIF cyl(radius, height, direction, center, has_fluid_inside);
        auto shop = EB2::makeShop(cyl);
        EB2::Build(shop, geom_ls, req_coarsen, 10);
    }
    else if (geom_type == "wedge_hopper")
    {
#if (AMREX_SPACEDIM == 3)
        const auto plo = geom.ProbLoArray();
        const auto phi_arr = geom.ProbHiArray();

        amrex::Real exit_size = 0.0002;
        amrex::Real bin_size = 0.0002;
        amrex::Real funnel_height = 0.0002;
        amrex::Real vertoffset = 0.5 * (plo[1] + phi_arr[1]);

        amrex::ParmParse pp_wh("wedge_hopper");
        pp_wh.get("exit_size", exit_size);
        pp_wh.get("bin_size", bin_size);
        pp_wh.get("funnel_height", funnel_height);
        pp_wh.get("vertical_offset", vertoffset);

        Array<amrex::Real, 3> fp1 = {0.5f * exit_size, 0.0f, 0.0f};
        Array<amrex::Real, 3> fn1 = {funnel_height,
                                     0.5f * (exit_size - bin_size), 0.0f};
        EB2::PlaneIF funnel1(fp1, fn1);

        Array<amrex::Real, 3> bp1 = {0.5f * bin_size, funnel_height, 0.0f};
        Array<amrex::Real, 3> bn1 = {1.0f, 0.0f, 0.0f};
        EB2::PlaneIF bin1(bp1, bn1);

        Array<amrex::Real, 3> fp2 = {-0.5f * exit_size, 0.0f, 0.0f};
        Array<amrex::Real, 3> fn2 = {-funnel_height,
                                     0.5f * (exit_size - bin_size), 0.0f};
        EB2::PlaneIF funnel2(fp2, fn2);

        Array<amrex::Real, 3> bp2 = {-0.5f * bin_size, funnel_height, 0.0f};
        Array<amrex::Real, 3> bn2 = {-1.0f, 0.0f, 0.0f};
        EB2::PlaneIF bin2(bp2, bn2);

        Array<Real, 3> center = {0.5f * (plo[0] + phi_arr[0]), vertoffset,
                                  0.5f * (plo[2] + phi_arr[2])};

        auto hopper_alone = EB2::translate(
            EB2::makeUnion(funnel1, bin1, funnel2, bin2), center);

        amrex::Real len[AMREX_SPACEDIM] = {phi_arr[0] - plo[0],
                                            phi_arr[1] - plo[1],
                                            phi_arr[2] - plo[2]};
        RealArray lo_box, hi_box;
        lo_box[0] = plo[0] - len[0];
        lo_box[1] = plo[1] - len[1];
        lo_box[2] = plo[2] - len[2];
        hi_box[0] = phi_arr[0] + len[0];
        hi_box[1] = vertoffset;
        hi_box[2] = phi_arr[2] + len[2];
        EB2::BoxIF box_below(lo_box, hi_box, false);

        auto hopper = EB2::makeComplement(
            EB2::makeUnion(EB2::makeComplement(hopper_alone), box_below));

        auto shop = EB2::makeShop(hopper);
        EB2::Build(shop, geom_ls, req_coarsen, 10);
#else
        amrex::Abort("wedge_hopper geometry is only implemented in 3D");
#endif
    }
    else
    {
        EB2::Build(geom_ls, req_coarsen, 10);
    }

    build_factory(geom, ba, dm, nghost);

    MultiFab *lsphi_out = allocate_body_lsphi(ba, dm, nghost, ls_ref);

    const EB2::IndexSpace &ebis = EB2::IndexSpace::top();
    const EB2::Level &lslev = ebis.getLevel(geom_ls);

    amrex::FillSignedDistance(*lsphi_out, lslev, *ebfactory, ls_ref);
    lsphi_out->FillBoundary(geom_ls.periodicity());
    return lsphi_out;
}

/**
 * @brief Initialises EB geometry and all body level-set MultiFabs.
 *
 * Reads eb2.body_names for multi-body mode, or falls back to eb2.geom_type
 * for single-body backward compatibility.  On exit, mpm_ebtools::ls_bodies
 * is populated and mpm_ebtools::ebfactory reflects the last body's geometry.
 *
 * Writes one "ebplt_<name>" plotfile per body for visualisation.
 *
 * @param[in] geom  Coarse-level geometry.
 * @param[in] ba    BoxArray for the coarse level.
 * @param[in] dm    DistributionMapping.
 */
void init_eb(const Geometry &geom,
             const BoxArray &ba,
             const DistributionMapping &dm)
{
    constexpr int nghost = 1;

    amrex::ParmParse pp("eb2");

    std::vector<std::string> body_names;
    pp.queryarr("body_names", body_names);

    bool legacy_single_body = false;

    if (body_names.empty())
    {
        std::string geom_type = "all_regular";
        pp.query("geom_type", geom_type);

        if (geom_type == "all_regular")
        {
            amrex::Print() << "\n[EB] geom_type = all_regular — no EB geometry\n";
            return;
        }

        body_names.push_back("body_0");
        legacy_single_body = true;
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<int>(body_names.size()) <= EXAGOOP_MAX_LS_BODIES,
        "Number of EB bodies exceeds EXAGOOP_MAX_LS_BODIES");

    using_levelset_geometry = true;
    ls_bodies.reserve(body_names.size());

    for (const auto &name : body_names)
    {
        const std::string pp_prefix = legacy_single_body ? "eb2" : name;
        amrex::ParmParse pp_body(pp_prefix);

        std::string geom_type;
        pp_body.get("geom_type", geom_type);

        int ls_ref = 1;
        pp_body.query("ls_refinement", ls_ref);

        MultiFab *body_lsphi = nullptr;

        if (geom_type == "udf_cpp")
        {
            body_lsphi =
                build_udf_levelset(name, pp_prefix, geom, ba, dm, nghost, ls_ref);
        }
        else if (geom_type == "stl")
        {
            body_lsphi =
                build_stl_levelset(name, pp_prefix, geom, ba, dm, nghost, ls_ref);
        }
        else
        {
            body_lsphi = build_analytic_levelset(name, pp_prefix, geom_type,
                                                  geom, ba, dm, nghost, ls_ref);
        }

        
        std::string mom_bc = "noslipwall";
        pp_body.query("levelset_mom", mom_bc);


        if (mom_bc == "noslipwall" && !pp_body.contains("levelset_mom")
            && legacy_single_body)
        {
            amrex::ParmParse pp_mpm("mpm");
            int legacy_int = -1;
            if (pp_mpm.query("levelset_bc", legacy_int))
            {
                amrex::Print()
                    << "[EB] Warning: mpm.levelset_bc is deprecated. "
                       "Use eb2.levelset_mom = noslipwall|slipwall|partialslip "
                       "instead.\n";
                if (legacy_int == 2)      mom_bc = "slipwall";
                else if (legacy_int == 3) mom_bc = "partialslip";
                else                      mom_bc = "noslipwall";
            }
        }

        amrex::Real wall_mu = 0.0;
        pp_body.query("lset_wall_mu", wall_mu);


        if (wall_mu == 0.0 && !pp_body.contains("lset_wall_mu")
            && legacy_single_body)
        {
            amrex::ParmParse pp_mpm("mpm");
            pp_mpm.query("levelset_wall_mu", wall_mu);
        }

        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> wall_vel =
            {AMREX_D_DECL(0.0, 0.0, 0.0)};
        {
            std::vector<amrex::Real> wv(AMREX_SPACEDIM, 0.0);
            if (pp_body.queryarr("lset_wall_vel", wv))
            {
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    wall_vel[d] = wv[d];
            }
        }

        LevelSetBody body;
        body.name = name;
        body.lsphi = body_lsphi;
        body.ls_refinement = ls_ref;
        body.mom_bc_type = mom_bc;
        body.wall_mu = wall_mu;
        body.wall_vel = wall_vel;

        std::string temp_bc = "adiabatic";
        pp_body.query("temp_bc_type", temp_bc);

        amrex::Real T_wall = 0.0;
        pp_body.query("lset_T_wall", T_wall);

        amrex::Real heat_flux = 0.0;
        pp_body.query("lset_heat_flux", heat_flux);

        amrex::Real h_conv = 0.0;
        pp_body.query("lset_h_conv", h_conv);

        amrex::Real T_inf_val = 0.0;
        pp_body.query("lset_T_inf", T_inf_val);

        body.temp_bc_type = temp_bc;
        body.T_wall       = T_wall;
        body.heat_flux    = heat_flux;
        body.h_conv       = h_conv;
        body.T_inf        = T_inf_val;

        amrex::Print() << "  [EB] Body '" << name
                       << "': levelset_mom=" << mom_bc
                       << "  lset_wall_mu=" << wall_mu
                       << "  temp_bc_type=" << temp_bc << "\n";

        ls_bodies.push_back(std::move(body));
    }

    for (const auto &body : ls_bodies)
    {
        Geometry geom_ls = refined_geom(geom, body.ls_refinement);
        BoxArray plot_ba = ba;
        plot_ba.refine(body.ls_refinement);

        MultiFab plotmf(plot_ba, dm, body.lsphi->nComp(), 0);
        amrex::average_node_to_cellcenter(plotmf, 0, *body.lsphi, 0,
                                          body.lsphi->nComp());

        std::string pltname = "ebplt_" + body.name;
        WriteSingleLevelPlotfile(pltname, plotmf, {"phi"}, geom_ls, 0.0, 0);
    }
}

} // namespace mpm_ebtools
#endif // USE_EB
