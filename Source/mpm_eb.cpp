// clang-format off
#include <mpm_eb.H>
#include <mpm_udf_loader.H>
#if USE_EB
#include <AMReX_EB_utils.H>
#endif
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
// clang-format on

#if USE_EB
namespace mpm_ebtools
{
EBFArrayBoxFactory *ebfactory = NULL;
MultiFab *lsphi = NULL;
int ls_refinement = 1;
bool using_levelset_geometry = false;

// File-scope UDF loader — must outlive EB2::Build and FillSignedDistance.
// Declared here so it is valid for the entire simulation lifetime.
static UDFLoader g_udf_loader;

// ─────────────────────────────────────────────────────────────────────────────
/**
 * @brief Shared helper: builds EB index space and fills lsphi from a geometry
 * shop.
 *
 * Extracted so that all four geometry paths (analytic EB2, wedge_hopper,
 * udf_cpp, stl) call identical EB + level-set setup code.
 *
 * @param[in] gshop   Any AMReX EB2 geometry shop (result of EB2::makeShop).
 * @param[in] geom    Coarse geometry.
 * @param[in] ba      BoxArray.
 * @param[in] dm      DistributionMapping.
 * @param[in] nghost  Number of ghost cells for the EB factory and lsphi.
 */
template <typename GShop>
void build_eb_and_levelset(const GShop &gshop,
                           const Geometry &geom,
                           const BoxArray &ba,
                           const DistributionMapping &dm,
                           int nghost)
{
    Box dom_ls = geom.Domain();
    dom_ls.refine(ls_refinement);
    Geometry geom_ls(dom_ls);

    int required_coarsening_level = 0;
    if (ls_refinement > 1)
    {
        int tmp = ls_refinement;
        while (tmp >>= 1)
            ++required_coarsening_level;
    }

    EB2::Build(gshop, geom_ls, required_coarsening_level, 10);

    const EB2::IndexSpace &ebis = EB2::IndexSpace::top();
    const EB2::Level &eblev = ebis.getLevel(geom);
    const EB2::Level &lslev = ebis.getLevel(geom_ls);

    ebfactory = new EBFArrayBoxFactory(
        eblev, geom, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

    // ── lsphi needs enough ghost cells for FillSignedDistance to propagate ──
    // Use the max number of cells in any direction on the refined grid as
    // the ghost cell count — this guarantees full-domain coverage regardless
    // of box decomposition.
    const auto &domain = geom.Domain();
    int max_cells = 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        max_cells = std::max(max_cells, domain.length(d));
    int ls_nghost = max_cells * ls_refinement; // full refined domain width

    BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
    ls_ba.refine(ls_refinement);
    lsphi = new MultiFab;
    lsphi->define(ls_ba, dm, 1, ls_nghost);
    amrex::FillSignedDistance(*lsphi, lslev, *ebfactory, ls_refinement);
}

// ─────────────────────────────────────────────────────────────────────────────
#if (AMREX_SPACEDIM == 3)
/**
 * @brief Builds the wedge hopper EB and level-set (3D only).
 * Unchanged from original implementation.
 */
void make_wedge_hopper_levelset(const Geometry &geom,
                                const BoxArray &ba,
                                const DistributionMapping &dm)
{
    int ls_ref = ls_refinement;
    int nghost = 1;

    const auto plo = geom.ProbLoArray();
    const auto phi = geom.ProbHiArray();

    amrex::Real exit_size = 0.0002;
    amrex::Real bin_size = 0.0002;
    amrex::Real funnel_height = 0.0002;
    amrex::Real vertoffset = 0.5 * (plo[1] + phi[1]);

    amrex::ParmParse pp("wedge_hopper");
    pp.get("exit_size", exit_size);
    pp.get("bin_size", bin_size);
    pp.get("funnel_height", funnel_height);
    pp.get("vertical_offset", vertoffset);

    Array<amrex::Real, 3> funnel_point1 = {0.5 * exit_size, 0.0, 0.0};
    Array<amrex::Real, 3> funnel_normal1 = {funnel_height,
                                            0.5 * (exit_size - bin_size), 0.0};
    EB2::PlaneIF funnel1(funnel_point1, funnel_normal1);

    Array<amrex::Real, 3> bin_point1 = {0.5 * bin_size, funnel_height, 0.0};
    Array<amrex::Real, 3> bin_normal1 = {1.0, 0.0, 0.0};
    EB2::PlaneIF bin1(bin_point1, bin_normal1);

    Array<amrex::Real, 3> funnel_point2 = {-0.5 * exit_size, 0.0, 0.0};
    Array<amrex::Real, 3> funnel_normal2 = {-funnel_height,
                                            0.5 * (exit_size - bin_size), 0.0};
    EB2::PlaneIF funnel2(funnel_point2, funnel_normal2);

    Array<amrex::Real, 3> bin_point2 = {-0.5 * bin_size, funnel_height, 0.0};
    Array<amrex::Real, 3> bin_normal2 = {-1.0, 0.0, 0.0};
    EB2::PlaneIF bin2(bin_point2, bin_normal2);

    Array<Real, 3> center = {0.5 * (plo[0] + phi[0]), vertoffset,
                             0.5 * (plo[2] + phi[2])};
    auto hopper_alone =
        EB2::translate(EB2::makeUnion(funnel1, bin1, funnel2, bin2), center);

    amrex::Real len[AMREX_SPACEDIM] = {phi[0] - plo[0], phi[1] - plo[1],
                                       phi[2] - plo[2]};
    RealArray lo, hi;
    lo[0] = plo[0] - len[0];
    lo[1] = plo[1] - len[1];
    lo[2] = plo[2] - len[2];
    hi[0] = phi[0] + len[0];
    hi[1] = vertoffset;
    hi[2] = phi[2] + len[2];
    EB2::BoxIF box_below(lo, hi, false);

    auto hopper = EB2::makeComplement(
        EB2::makeUnion(EB2::makeComplement(hopper_alone), box_below));
    auto hopper_gshop = EB2::makeShop(hopper);

    // Reuse shared helper for EB + lsphi setup
    build_eb_and_levelset(hopper_gshop, geom, ba, dm, nghost);
}
#endif // AMREX_SPACEDIM == 3

// Replace FillSignedDistance with direct UDF evaluation for udf_cpp path
void fill_lsphi_from_udf(MultiFab &lsphi,
                         const Geometry &geom,
                         int ls_refinement,
                         LevelSetPhiFn phi_fn)
{
    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();

    for (MFIter mfi(lsphi); mfi.isValid(); ++mfi)
    {
        const Box &bx = mfi.fabbox(); // include ghost cells
        Array4<Real> phi = lsphi.array(mfi);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                // Physical position of this node on the refined grid
                amrex::Real x = plo[0] + i * dx[0] / ls_refinement;
                amrex::Real y = plo[1] + j * dx[1] / ls_refinement;
                amrex::Real z = (AMREX_SPACEDIM == 3)
                                    ? plo[2] + k * dx[2] / ls_refinement
                                    : 0.0;

                phi(i, j, k) = static_cast<amrex::Real>(phi_fn(x, y, z));
            });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
/**
 * @brief Initialises embedded boundary and level-set data structures.
 *
 * Reads eb2.geom_type from the input file and dispatches to the appropriate
 * geometry path:
 *
 *   all_regular   — no EB (default)
 *   wedge_hopper  — built-in 3-D wedge hopper geometry
 *   <amrex_type>  — any analytic AMReX EB2 shape (sphere, plane, cylinder …)
 *   udf_cpp       — user-compiled shared library defining levelset_phi()
 *   stl           — surface mesh loaded from an STL file
 *
 * For udf_cpp and stl the geometry is specified entirely in the input file;
 * no recompilation of ExaGOOP is required.
 *
 * @param[in] geom  Coarse-level geometry.
 * @param[in] ba    BoxArray.
 * @param[in] dm    DistributionMapping.
 */
void init_eb(const Geometry &geom,
             const BoxArray &ba,
             const DistributionMapping &dm)
{
    int nghost = 1;
    std::string geom_type = "all_regular";

    amrex::ParmParse pp("eb2");
    pp.query("geom_type", geom_type);
    pp.query("ls_refinement", ls_refinement);

    // ── all_regular: no EB
    // ────────────────────────────────────────────────────
    if (geom_type == "all_regular")
    {
        // Nothing to do — using_levelset_geometry stays false
    }

    // ── wedge_hopper: built-in 3-D geometry
    // ───────────────────────────────────
    else if (geom_type == "wedge_hopper")
    {
#if (AMREX_SPACEDIM == 3)
        using_levelset_geometry = true;
        make_wedge_hopper_levelset(geom, ba, dm);
#else
        amrex::Abort("wedge_hopper geometry is only available in 3D");
#endif
    }

    // ── udf_cpp: runtime user-defined level set
    // ───────────────────────────────
    /**
     * The user compiles their geometry .cpp into a shared library using the
     * exagoop-build-udf helper (or GNUmakefile.udf), then points the input
     * file at the resulting .so / .dylib / .dll.
     *
     * Required input keys:
     *   eb2.geom_type   = udf_cpp
     *   eb2.udf_so_file = /path/to/levelset_udf.so
     *
     * Optional:
     *   eb2.ls_refinement = 2   (default 1)
     */
    else if (geom_type == "udf_cpp")
    {
        using_levelset_geometry = true;

        std::string so_path;
        if (!pp.query("udf_so_file", so_path))
            amrex::Abort("[UDF] eb2.udf_so_file must be set");

        g_udf_loader.load(so_path);
        auto phi_fn = g_udf_loader.get_fn();

        // Build EB index space from UDF
        auto udf_if = [phi_fn](const amrex::RealArray &p) -> amrex::Real
        { return static_cast<amrex::Real>(phi_fn(p[0], p[1], p[2])); };
        auto gshop = EB2::makeShop(udf_if);

        // Build EB factory (same as before)
        Box dom_ls = geom.Domain();
        dom_ls.refine(ls_refinement);
        Geometry geom_ls(dom_ls);

        int required_coarsening_level = 0;
        if (ls_refinement > 1)
        {
            int tmp = ls_refinement;
            while (tmp >>= 1)
                ++required_coarsening_level;
        }

        EB2::Build(gshop, geom_ls, required_coarsening_level, 10);

        const EB2::IndexSpace &ebis = EB2::IndexSpace::top();
        const EB2::Level &eblev = ebis.getLevel(geom);

        ebfactory = new EBFArrayBoxFactory(
            eblev, geom, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

        // Fill lsphi directly from UDF — bypasses FillSignedDistance entirely.
        // The UDF is an analytic SDF so no redistancing is needed.
        BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
        ls_ba.refine(ls_refinement);
        lsphi = new MultiFab;
        lsphi->define(ls_ba, dm, 1, nghost); // nghost=1 is fine now

        const auto plo = geom.ProbLoArray();
        const auto dx = geom.CellSizeArray();
        const int lsr = ls_refinement;

        for (MFIter mfi(*lsphi); mfi.isValid(); ++mfi)
        {
            const Box &bx = mfi.fabbox();
            Array4<Real> phi = lsphi->array(mfi);

            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real x = plo[0] + i * dx[0] / lsr;
                    amrex::Real y = plo[1] + j * dx[1] / lsr;
                    amrex::Real z =
                        (AMREX_SPACEDIM == 3) ? plo[2] + k * dx[2] / lsr : 0.0;
                    phi(i, j, k) = static_cast<amrex::Real>(phi_fn(x, y, z));
                });
        }
    }

    // ── stl: surface mesh from file
    // ───────────────────────────────────────────
    /**
     * Loads an ASCII or binary STL surface mesh and computes the signed
     * distance field.
     *
     * AMReX >= 23.05 provides EB2::STLGeom directly.
     * Older AMReX versions use EB2::GeometryShop<EB2::TriangleMeshIF> instead.
     * Both paths are supported via the AMREX_VERSION preprocessor guard below.
     *
     * Required input keys:
     *   eb2.geom_type = stl
     *   eb2.stl_file  = /path/to/surface.stl
     *
     * Optional:
     *   eb2.stl_reverse   = false   (flip inside/outside if STL normals point
     * inward) eb2.ls_refinement = 2
     */
    else if (geom_type == "stl")
    {
        using_levelset_geometry = true;

        std::string stl_file;
        bool stl_reverse = false;

        if (!pp.query("stl_file", stl_file))
        {
            amrex::Abort("[STL] eb2.stl_file must be set when "
                         "eb2.geom_type = stl");
        }
        pp.query("stl_reverse", stl_reverse);

        amrex::Print() << "[STL] Loading: " << stl_file << "\n";

#if EXAGOOP_AMREX_HAS_STLGEOM
        // ── AMReX has EB2::STLGeom (detected at configure time) ──────────────
        auto stl_geom = amrex::EB2::STLGeom(stl_file);

        if (stl_reverse)
        {
            auto gshop = EB2::makeShop(EB2::makeComplement(stl_geom));
            build_eb_and_levelset(gshop, geom, ba, dm, nghost);
        }
        else
        {
            auto gshop = EB2::makeShop(stl_geom);
            build_eb_and_levelset(gshop, geom, ba, dm, nghost);
        }
#else
        // ── Older AMReX: delegate to AMReX's own built-in STL EB2 path ───────
        // AMReX reads eb2.stl_file from ParmParse when geom_type = stl.
        // stl_reverse is not supported on this path — flip STL normals in
        // your CAD tool if needed.
        if (stl_reverse)
        {
            amrex::Print() << "[STL] WARNING: stl_reverse is not supported on "
                           << "this AMReX version. Ignoring.\n";
        }
        amrex::Print() << "[STL] Using AMReX built-in STL EB2 path.\n";

        int required_coarsening_level = 0;
        if (ls_refinement > 1)
        {
            int tmp = ls_refinement;
            while (tmp >>= 1)
                ++required_coarsening_level;
        }
        Box dom_ls = geom.Domain();
        dom_ls.refine(ls_refinement);
        Geometry geom_ls(dom_ls);

        amrex::EB2::Build(geom_ls, required_coarsening_level, 10);

        const EB2::IndexSpace &ebis = EB2::IndexSpace::top();
        const EB2::Level &eblev = ebis.getLevel(geom);
        const EB2::Level &lslev = ebis.getLevel(geom_ls);

        ebfactory = new EBFArrayBoxFactory(
            eblev, geom, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

        BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
        ls_ba.refine(ls_refinement);
        lsphi = new MultiFab;
        lsphi->define(ls_ba, dm, 1, nghost);
        amrex::FillSignedDistance(*lsphi, lslev, *ebfactory, ls_refinement);
#endif
    }

    // ── anything else: pass directly to AMReX EB2 (sphere, plane, etc.)
    // ───────
    else
    {
        using_levelset_geometry = true;

        int required_coarsening_level = 0;
        if (ls_refinement > 1)
        {
            int tmp = ls_refinement;
            while (tmp >>= 1)
                ++required_coarsening_level;
        }

        Box dom_ls = geom.Domain();
        dom_ls.refine(ls_refinement);
        Geometry geom_ls(dom_ls);

        amrex::EB2::Build(geom_ls, required_coarsening_level, 10);

        const EB2::IndexSpace &ebis = EB2::IndexSpace::top();
        const EB2::Level &eblev = ebis.getLevel(geom);
        const EB2::Level &lslev = ebis.getLevel(geom_ls);

        ebfactory = new EBFArrayBoxFactory(
            eblev, geom, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

        BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
        ls_ba.refine(ls_refinement);
        lsphi = new MultiFab;
        lsphi->define(ls_ba, dm, 1, nghost);
        amrex::FillSignedDistance(*lsphi, lslev, *ebfactory, ls_refinement);
    }

    // ── write phi plotfile for all geometry types
    // ─────────────────────────────
    if (using_levelset_geometry)
    {
        const std::string &pltfile = "ebplt";
        Box dom_ls = geom.Domain();
        dom_ls.refine(ls_refinement);
        Geometry geom_ls(dom_ls);
        BoxArray plot_ba = ba;
        plot_ba.refine(ls_refinement);
        MultiFab plotmf(plot_ba, dm, lsphi->nComp(), 0);
        amrex::average_node_to_cellcenter(plotmf, 0, *lsphi, 0, lsphi->nComp());
        WriteSingleLevelPlotfile(pltfile, plotmf, {"phi"}, geom_ls, 0.0, 0);
    }
}

} // namespace mpm_ebtools
#endif // USE_EB
