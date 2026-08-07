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
#include <AMReX_MultiFabUtil.H>
#include <mpm_udf_loader.H>
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
    // Returns the amrex refinement level: 1->0, 2->1,4->2
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
    // returns a refined (ls_ref) geometry
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

    amrex::Print() << "\n[EB] Body '" << name << "' — STL: " << stl_file
                   << "\n";

    Geometry geom_ls = refined_geom(geom, ls_ref);
    int req_coarsen = coarsening_level_for_refinement(ls_ref);

    {
        amrex::ParmParse ppeb("eb2");
        ppeb.add("geom_type", std::string("stl"));
        ppeb.add("stl_file", stl_file);
    }

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
    amrex::Print() << "[EB] Body '" << name
                   << "' — analytic geometry: " << geom_type << "\n";

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
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
            center[d] = center_v[d];

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
            point[d] = point_v[d];
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
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
            center[d] = center_v[d];

        bool has_fluid_inside = false;
        pp.query("cylinder_has_fluid_inside", has_fluid_inside);

        EB2::CylinderIF cyl(radius, height, direction, center,
                            has_fluid_inside);
        auto shop = EB2::makeShop(cyl);
        EB2::Build(shop, geom_ls, req_coarsen, 10);
    }
    else if (geom_type == "box")
    {
        std::vector<amrex::Real> lo_v(AMREX_SPACEDIM, 0.0);
        std::vector<amrex::Real> hi_v(AMREX_SPACEDIM, 1.0);
        pp.getarr("box_lo", lo_v);
        pp.getarr("box_hi", hi_v);
        amrex::RealArray lo_box, hi_box;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            lo_box[d] = lo_v[d];
            hi_box[d] = hi_v[d];
        }

        bool has_fluid_inside = false;
        pp.query("box_has_fluid_inside", has_fluid_inside);

        EB2::BoxIF box_if(lo_box, hi_box, has_fluid_inside);
        auto shop = EB2::makeShop(box_if);
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

        amrex::Real len[AMREX_SPACEDIM] = {
            phi_arr[0] - plo[0], phi_arr[1] - plo[1], phi_arr[2] - plo[2]};
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
    else if (geom_type == "wedge")
    {
        // 2D cutting-tool wedge: the solid is the intersection of two
        // half-spaces (rake face and flank face) meeting at the tool tip.
        // Both planes pass through wedge_tip; wedge_normal1/2 are the OUTWARD
        // normals of the two faces (point away from the solid). Moving via the
        // usual motion_* keys turns this into an advancing cutting tool.
        std::vector<amrex::Real> tip_v(AMREX_SPACEDIM, 0.0);
        std::vector<amrex::Real> n1_v(AMREX_SPACEDIM, 0.0);
        std::vector<amrex::Real> n2_v(AMREX_SPACEDIM, 0.0);
        n1_v[0] = 1.0;
        n2_v[1] = -1.0;
        pp.getarr("wedge_tip", tip_v);
        pp.getarr("wedge_normal1", n1_v);
        pp.getarr("wedge_normal2", n2_v);
        amrex::RealArray tip, n1, n2;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            tip[d] = tip_v[d];
            n1[d] = n1_v[d];
            n2[d] = n2_v[d];
        }
        // PlaneIF(point, normal, fluid_inside=false): solid on the side the
        // normal points toward. Intersection keeps the region solid in BOTH.
        EB2::PlaneIF face1(tip, n1, false);
        EB2::PlaneIF face2(tip, n2, false);
        auto wedge = EB2::makeIntersection(face1, face2);
        auto shop = EB2::makeShop(wedge);
        EB2::Build(shop, geom_ls, req_coarsen, 10);
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
             const DistributionMapping &dm,
             MPMspecs &specs)
{
    constexpr int nghost = 4;

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
            amrex::Print()
                << "\n[EB] geom_type = all_regular — no EB geometry\n";
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
            body_lsphi = build_udf_levelset(name, pp_prefix, geom, ba, dm,
                                            nghost, ls_ref);
        }
        else if (geom_type == "stl")
        {
            body_lsphi = build_stl_levelset(name, pp_prefix, geom, ba, dm,
                                            nghost, ls_ref);
        }
        else
        {
            body_lsphi = build_analytic_levelset(name, pp_prefix, geom_type,
                                                 geom, ba, dm, nghost, ls_ref);
        }

        std::string mom_bc = "noslipwall";
        pp_body.query("levelset_mom", mom_bc);

        if (mom_bc == "noslipwall" && !pp_body.contains("levelset_mom") &&
            legacy_single_body)
        {
            amrex::ParmParse pp_mpm("mpm");
            int legacy_int = -1;
            if (pp_mpm.query("levelset_bc", legacy_int))
            {
                amrex::Print()
                    << "[EB] Warning: mpm.levelset_bc is deprecated. "
                       "Use eb2.levelset_mom = noslipwall|slipwall|partialslip "
                       "instead.\n";
                if (legacy_int == 2)
                    mom_bc = "slipwall";
                else if (legacy_int == 3)
                    mom_bc = "partialslip";
                else
                    mom_bc = "noslipwall";
            }
        }

        amrex::Real wall_mu = 0.0;
        pp_body.query("lset_wall_mu", wall_mu);

        if (wall_mu == 0.0 && !pp_body.contains("lset_wall_mu") &&
            legacy_single_body)
        {
            amrex::ParmParse pp_mpm("mpm");
            pp_mpm.query("levelset_wall_mu", wall_mu);
        }

        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> wall_vel = {
            AMREX_D_DECL(0.0, 0.0, 0.0)};
        {
            std::vector<amrex::Real> wv(AMREX_SPACEDIM, 0.0);
            if (pp_body.queryarr("lset_wall_vel", wv))
            {
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    wall_vel[d] = wv[d];
            }
        }

        RigidMotion motion;
        {
            std::string mtype = "static";
            pp_body.query("motion_type", mtype);
            if (mtype == "static")
                motion.type = MOTION_STATIC;
            else if (mtype == "translate")
                motion.type = MOTION_TRANSLATE;
            else if (mtype == "rotate")
                motion.type = MOTION_ROTATE;
            else if (mtype == "udf")
                motion.type = MOTION_UDF;
            else
                amrex::Abort("LevelSetBody '" + name +
                             "': unknown motion_type '" + mtype +
                             "'. Valid: static, translate, rotate, udf.");

            std::vector<amrex::Real> mc(AMREX_SPACEDIM, 0.0);
            if (pp_body.queryarr("motion_center", mc))
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    motion.center[d] = mc[d];

            std::vector<amrex::Real> mv(AMREX_SPACEDIM, 0.0);
            if (pp_body.queryarr("motion_vel", mv))
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    motion.vel[d] = mv[d];

            std::vector<amrex::Real> mw;
            if (pp_body.queryarr("motion_omega", mw))
            {
                if (mw.size() == 1) // 2D convention: single z-component
                    motion.omega[2] = mw[0];
                else
                    for (std::size_t d = 0; d < mw.size() && d < 3; ++d)
                        motion.omega[d] = mw[d];
            }
        }

        LevelSetBody body;
        body.name = name;
        body.lsphi = body_lsphi;
        body.ls_refinement = ls_ref;
        body.mom_bc_type = mom_bc;
        body.wall_mu = wall_mu;
        body.wall_vel = wall_vel;
        body.motion = motion;

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
        body.T_wall = T_wall;
        body.heat_flux = heat_flux;
        body.h_conv = h_conv;
        body.T_inf = T_inf_val;

        const char *mtype_str[] = {"static", "translate", "rotate", "udf"};
        amrex::Print() << "  [EB] Body '" << name
                       << "': levelset_mom=" << mom_bc
                       << "  lset_wall_mu=" << wall_mu
                       << "  temp_bc_type=" << temp_bc
                       << "  motion=" << mtype_str[motion.type] << "  vel=("
                       << motion.vel[0] << "," << motion.vel[1]
                       << ")  omega_z=" << motion.omega[2] << "\n";

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
        WriteSingleLevelPlotfile(specs.blevset_output_folder + pltname, plotmf,
                                 {"phi"}, geom_ls, 0.0, 0);
    }
}

// ============================================================
// Level-set transport (moving level set, Stage 1).
// Solves d(phi)/dt + v . grad(phi) = 0 on each moving body's refined nodal
// grid with first-order upwind + explicit Euler. v = rigid-body velocity.
// ============================================================
void advance_levelset_bodies(const Geometry &geom,
                             amrex::Real time,
                             amrex::Real dt)
{
    for (auto &body : ls_bodies)
    {
        const RigidMotion motion = body.motion;
        if (motion.type == MOTION_STATIC)
            continue;

        Geometry geom_ls = refined_geom(geom, body.ls_refinement);
        const auto plo = geom_ls.ProbLoArray();
        const auto dx = geom_ls.CellSizeArray();

        Box nddom = amrex::surroundingNodes(geom_ls.Domain());
        amrex::GpuArray<int, AMREX_SPACEDIM> nlo, nhi;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            nlo[d] = nddom.smallEnd(d);
            nhi[d] = nddom.bigEnd(d);
        }

        MultiFab &phi = *body.lsphi;
        phi.FillBoundary(geom_ls.periodicity());

        MultiFab phi_old(phi.boxArray(), phi.DistributionMap(), 1,
                         phi.nGrowVect());
        MultiFab::Copy(phi_old, phi, 0, 0, 1, phi.nGrowVect());

        const amrex::Real t = time;

        for (MFIter mfi(phi); mfi.isValid(); ++mfi)
        {
            const Box &nbx = mfi.validbox();
            Array4<amrex::Real> pn = phi.array(mfi);
            Array4<amrex::Real const> po = phi_old.const_array(mfi);

            amrex::ParallelFor(
                nbx,
                [=] AMREX_GPU_DEVICE(AMREX_D_DECL(int i, int j, int k)) noexcept
                {
#if AMREX_SPACEDIM == 2
                    const int k = 0;
#endif
                    amrex::Real xn[AMREX_SPACEDIM] = {AMREX_D_DECL(
                        plo[XDIR] + i * dx[XDIR], plo[YDIR] + j * dx[YDIR],
                        plo[ZDIR] + k * dx[ZDIR])};

                    amrex::Real v[AMREX_SPACEDIM];
                    motion.wall_velocity(xn, t, v);

                    amrex::Real adv = 0.0;
                    {
                        int im = amrex::max(i - 1, nlo[XDIR]);
                        int ip = amrex::min(i + 1, nhi[XDIR]);
                        amrex::Real g =
                            (v[XDIR] > 0.0)
                                ? (po(i, j, k) - po(im, j, k)) / dx[XDIR]
                                : (po(ip, j, k) - po(i, j, k)) / dx[XDIR];
                        adv += v[XDIR] * g;
                    }
#if AMREX_SPACEDIM >= 2
                    {
                        int jm = amrex::max(j - 1, nlo[YDIR]);
                        int jp = amrex::min(j + 1, nhi[YDIR]);
                        amrex::Real g =
                            (v[YDIR] > 0.0)
                                ? (po(i, j, k) - po(i, jm, k)) / dx[YDIR]
                                : (po(i, jp, k) - po(i, j, k)) / dx[YDIR];
                        adv += v[YDIR] * g;
                    }
#endif
#if AMREX_SPACEDIM == 3
                    {
                        int km = amrex::max(k - 1, nlo[ZDIR]);
                        int kp = amrex::min(k + 1, nhi[ZDIR]);
                        amrex::Real g =
                            (v[ZDIR] > 0.0)
                                ? (po(i, j, k) - po(i, j, km)) / dx[ZDIR]
                                : (po(i, j, kp) - po(i, j, k)) / dx[ZDIR];
                        adv += v[ZDIR] * g;
                    }
#endif
                    pn(i, j, k) = po(i, j, k) - dt * adv;
                });
        }

        phi.FillBoundary(geom_ls.periodicity());
    }
}

// ============================================================
// Level-set reinitialization (redistancing) — restore |grad phi| = 1.
// Sussman PDE  d phi/d tau + sign(phi0)(|grad phi| - 1) = 0  with a first-order
// Godunov upwind Hamiltonian, held fixed at the zero contour by the frozen
// sign(phi0). A few pseudo-time steps repair the field near the interface.
// ============================================================
void reinitialize_levelset_bodies(const Geometry &geom, int n_iters)
{
    for (auto &body : ls_bodies)
    {
        if (body.motion.type == MOTION_STATIC)
            continue;

        Geometry geom_ls = refined_geom(geom, body.ls_refinement);
        const auto dx = geom_ls.CellSizeArray();
        amrex::Real dxmin = dx[0];
        for (int d = 1; d < AMREX_SPACEDIM; ++d)
            dxmin = amrex::min(dxmin, dx[d]);
        const amrex::Real dtau = amrex::Real(0.5) * dxmin;
        const amrex::Real eps = dxmin; // smoothed-sign width

        Box nddom = amrex::surroundingNodes(geom_ls.Domain());
        amrex::GpuArray<int, AMREX_SPACEDIM> nlo, nhi;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            nlo[d] = nddom.smallEnd(d);
            nhi[d] = nddom.bigEnd(d);
        }

        MultiFab &phi = *body.lsphi;

        // Frozen field at entry: defines the zero contour and the subcell
        // distance used by the Russo-Smereka interface-preserving correction.
        MultiFab phi0(phi.boxArray(), phi.DistributionMap(), 1, phi.nGrowVect());
        MultiFab::Copy(phi0, phi, 0, 0, 1, phi.nGrowVect());
        phi0.FillBoundary(geom_ls.periodicity());

        for (int it = 0; it < n_iters; ++it)
        {
            phi.FillBoundary(geom_ls.periodicity());
            MultiFab phi_old(phi.boxArray(), phi.DistributionMap(), 1,
                             phi.nGrowVect());
            MultiFab::Copy(phi_old, phi, 0, 0, 1, phi.nGrowVect());

            for (MFIter mfi(phi); mfi.isValid(); ++mfi)
            {
                const Box &nbx = mfi.validbox();
                Array4<amrex::Real> pn = phi.array(mfi);
                Array4<amrex::Real const> po = phi_old.const_array(mfi);
                Array4<amrex::Real const> p0 = phi0.const_array(mfi);

                amrex::ParallelFor(
                    nbx,
                    [=] AMREX_GPU_DEVICE(AMREX_D_DECL(int i, int j, int k)) noexcept
                    {
#if AMREX_SPACEDIM == 2
                        const int k = 0;
#endif
                        const amrex::Real s0 = p0(i, j, k);
                        const amrex::Real phic = po(i, j, k);

                        const int im = amrex::max(i - 1, nlo[XDIR]);
                        const int ip = amrex::min(i + 1, nhi[XDIR]);
                        const amrex::Real s0xm = p0(im, j, k);
                        const amrex::Real s0xp = p0(ip, j, k);
#if AMREX_SPACEDIM >= 2
                        const int jm = amrex::max(j - 1, nlo[YDIR]);
                        const int jp = amrex::min(j + 1, nhi[YDIR]);
                        const amrex::Real s0ym = p0(i, jm, k);
                        const amrex::Real s0yp = p0(i, jp, k);
#endif
#if AMREX_SPACEDIM == 3
                        const int km = amrex::max(k - 1, nlo[ZDIR]);
                        const int kp = amrex::min(k + 1, nhi[ZDIR]);
                        const amrex::Real s0zm = p0(i, j, km);
                        const amrex::Real s0zp = p0(i, j, kp);
#endif
                        // Is this node adjacent to the zero contour in phi0?
                        bool nearIF = (s0 == amrex::Real(0.0)) ||
                                      (s0 * s0xm < amrex::Real(0.0)) ||
                                      (s0 * s0xp < amrex::Real(0.0));
#if AMREX_SPACEDIM >= 2
                        nearIF = nearIF || (s0 * s0ym < amrex::Real(0.0)) ||
                                 (s0 * s0yp < amrex::Real(0.0));
#endif
#if AMREX_SPACEDIM == 3
                        nearIF = nearIF || (s0 * s0zm < amrex::Real(0.0)) ||
                                 (s0 * s0zp < amrex::Real(0.0));
#endif

                        if (nearIF)
                        {
                            // Russo-Smereka: relax phi toward the subcell
                            // distance D computed from the frozen phi0, so the
                            // zero contour (and any sharp corner) stays fixed.
                            amrex::Real g0x = amrex::Real(0.5) * (s0xp - s0xm);
                            amrex::Real g0sq = g0x * g0x;
#if AMREX_SPACEDIM >= 2
                            amrex::Real g0y = amrex::Real(0.5) * (s0yp - s0ym);
                            g0sq += g0y * g0y;
#endif
#if AMREX_SPACEDIM == 3
                            amrex::Real g0z = amrex::Real(0.5) * (s0zp - s0zm);
                            g0sq += g0z * g0z;
#endif
                            // Floor at ~half a cell: |grad phi0|~1 for a signed
                            // distance, so the central difference is ~dx. This
                            // bounds D and avoids blow-up where the central
                            // difference cancels at a sharp corner.
                            amrex::Real grad0 = amrex::max(
                                std::sqrt(g0sq), amrex::Real(0.5) * dxmin);
                            amrex::Real D = dxmin * s0 / grad0; // subcell dist
                            amrex::Real sgn = (s0 > amrex::Real(0.0))
                                                  ? amrex::Real(1.0)
                                                  : ((s0 < amrex::Real(0.0))
                                                         ? amrex::Real(-1.0)
                                                         : amrex::Real(0.0));
                            pn(i, j, k) =
                                phic - (dtau / dxmin) *
                                           (sgn * amrex::Math::abs(phic) - D);
                        }
                        else
                        {
                            // Godunov upwind |grad phi|, branch by sign of phi0.
                            const bool pos = (s0 > amrex::Real(0.0));
                            amrex::Real g2 = amrex::Real(0.0);
                            {
                                amrex::Real a = (phic - po(im, j, k)) / dx[XDIR];
                                amrex::Real b = (po(ip, j, k) - phic) / dx[XDIR];
                                amrex::Real ap = amrex::max(a, amrex::Real(0.0));
                                amrex::Real bm = amrex::min(b, amrex::Real(0.0));
                                amrex::Real am = amrex::min(a, amrex::Real(0.0));
                                amrex::Real bp = amrex::max(b, amrex::Real(0.0));
                                g2 += pos ? amrex::max(ap * ap, bm * bm)
                                          : amrex::max(am * am, bp * bp);
                            }
#if AMREX_SPACEDIM >= 2
                            {
                                amrex::Real a = (phic - po(i, jm, k)) / dx[YDIR];
                                amrex::Real b = (po(i, jp, k) - phic) / dx[YDIR];
                                amrex::Real ap = amrex::max(a, amrex::Real(0.0));
                                amrex::Real bm = amrex::min(b, amrex::Real(0.0));
                                amrex::Real am = amrex::min(a, amrex::Real(0.0));
                                amrex::Real bp = amrex::max(b, amrex::Real(0.0));
                                g2 += pos ? amrex::max(ap * ap, bm * bm)
                                          : amrex::max(am * am, bp * bp);
                            }
#endif
#if AMREX_SPACEDIM == 3
                            {
                                amrex::Real a = (phic - po(i, j, km)) / dx[ZDIR];
                                amrex::Real b = (po(i, j, kp) - phic) / dx[ZDIR];
                                amrex::Real ap = amrex::max(a, amrex::Real(0.0));
                                amrex::Real bm = amrex::min(b, amrex::Real(0.0));
                                amrex::Real am = amrex::min(a, amrex::Real(0.0));
                                amrex::Real bp = amrex::max(b, amrex::Real(0.0));
                                g2 += pos ? amrex::max(ap * ap, bm * bm)
                                          : amrex::max(am * am, bp * bp);
                            }
#endif
                            const amrex::Real gradmag = std::sqrt(g2);
                            const amrex::Real S =
                                s0 / std::sqrt(s0 * s0 + eps * eps);
                            pn(i, j, k) =
                                phic - dtau * S * (gradmag - amrex::Real(1.0));
                        }
                    });
            }
        }

        phi.FillBoundary(geom_ls.periodicity());
    }
}

} // namespace mpm_ebtools
#endif // USE_EB
