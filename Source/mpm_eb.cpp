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
EBFArrayBoxFactory        *ebfactory              = NULL;
MultiFab                  *lsphi                  = NULL;
std::vector<MultiFab*>     lsphi_bodies;
int                        num_lsphi_bodies        = 0;
int                        ls_refinement           = 1;
bool                       using_levelset_geometry = false;

// File-scope UDF loaders — one per body.
// Index 0 = legacy single-body loader (existing behaviour preserved).
static constexpr int MAX_BODIES = 16;
static UDFLoader g_udf_loaders[MAX_BODIES];

// ─────────────────────────────────────────────────────────────────────────────
/**
 * @brief Shared helper: builds EB index space and fills lsphi from a geometry shop.
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
void build_eb_and_levelset(const GShop           &gshop,
                            const Geometry        &geom,
                            const BoxArray        &ba,
                            const DistributionMapping &dm,
                            int                    nghost)
{
    // Refine domain for level-set
    Box      dom_ls  = geom.Domain();
    dom_ls.refine(ls_refinement);
    Geometry geom_ls(dom_ls);

    // Required coarsening level so EB2 can reach the coarse grid
    int required_coarsening_level = 0;
    if (ls_refinement > 1)
    {
        int tmp = ls_refinement;
        while (tmp >>= 1) ++required_coarsening_level;
    }

    EB2::Build(gshop, geom_ls, required_coarsening_level, 10);

    const EB2::IndexSpace &ebis  = EB2::IndexSpace::top();
    const EB2::Level      &eblev = ebis.getLevel(geom);
    const EB2::Level      &lslev = ebis.getLevel(geom_ls);

    ebfactory = new EBFArrayBoxFactory(
        eblev, geom, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

    BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
    ls_ba.refine(ls_refinement);
    lsphi = new MultiFab;
    lsphi->define(ls_ba, dm, 1, nghost);
    amrex::FillSignedDistance(*lsphi, lslev, *ebfactory, ls_refinement);
}

// ─────────────────────────────────────────────────────────────────────────────
#if (AMREX_SPACEDIM == 3)
/**
 * @brief Builds the wedge hopper EB and level-set (3D only).
 * Unchanged from original implementation.
 */
void make_wedge_hopper_levelset(const Geometry        &geom,
                                const BoxArray        &ba,
                                const DistributionMapping &dm)
{
    int ls_ref  = ls_refinement;
    int nghost  = 1;

    const auto plo = geom.ProbLoArray();
    const auto phi = geom.ProbHiArray();

    amrex::Real exit_size    = 0.0002;
    amrex::Real bin_size     = 0.0002;
    amrex::Real funnel_height = 0.0002;
    amrex::Real vertoffset   = 0.5 * (plo[1] + phi[1]);

    amrex::ParmParse pp("wedge_hopper");
    pp.get("exit_size",      exit_size);
    pp.get("bin_size",       bin_size);
    pp.get("funnel_height",  funnel_height);
    pp.get("vertical_offset",vertoffset);

    Array<amrex::Real, 3> funnel_point1  = {0.5*exit_size, 0.0, 0.0};
    Array<amrex::Real, 3> funnel_normal1 = {funnel_height, 0.5*(exit_size-bin_size), 0.0};
    EB2::PlaneIF funnel1(funnel_point1, funnel_normal1);

    Array<amrex::Real, 3> bin_point1  = {0.5*bin_size, funnel_height, 0.0};
    Array<amrex::Real, 3> bin_normal1 = {1.0, 0.0, 0.0};
    EB2::PlaneIF bin1(bin_point1, bin_normal1);

    Array<amrex::Real, 3> funnel_point2  = {-0.5*exit_size, 0.0, 0.0};
    Array<amrex::Real, 3> funnel_normal2 = {-funnel_height, 0.5*(exit_size-bin_size), 0.0};
    EB2::PlaneIF funnel2(funnel_point2, funnel_normal2);

    Array<amrex::Real, 3> bin_point2  = {-0.5*bin_size, funnel_height, 0.0};
    Array<amrex::Real, 3> bin_normal2 = {-1.0, 0.0, 0.0};
    EB2::PlaneIF bin2(bin_point2, bin_normal2);

    Array<Real, 3> center = {0.5*(plo[0]+phi[0]), vertoffset, 0.5*(plo[2]+phi[2])};
    auto hopper_alone = EB2::translate(
        EB2::makeUnion(funnel1, bin1, funnel2, bin2), center);

    amrex::Real len[AMREX_SPACEDIM] = {phi[0]-plo[0], phi[1]-plo[1], phi[2]-plo[2]};
    RealArray lo, hi;
    lo[0] = plo[0]-len[0]; lo[1] = plo[1]-len[1]; lo[2] = plo[2]-len[2];
    hi[0] = phi[0]+len[0]; hi[1] = vertoffset;      hi[2] = phi[2]+len[2];
    EB2::BoxIF box_below(lo, hi, false);

    auto hopper = EB2::makeComplement(
        EB2::makeUnion(EB2::makeComplement(hopper_alone), box_below));
    auto hopper_gshop = EB2::makeShop(hopper);

    // Reuse shared helper for EB + lsphi setup
    build_eb_and_levelset(hopper_gshop, geom, ba, dm, nghost);
}
#endif  // AMREX_SPACEDIM == 3

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
void init_eb(const Geometry        &geom,
             const BoxArray        &ba,
             const DistributionMapping &dm)
{
    int         nghost    = 1;
    std::string geom_type = "all_regular";

    amrex::ParmParse pp("eb2");
    pp.query("geom_type",    geom_type);
    pp.query("ls_refinement", ls_refinement);

    // ── all_regular: no EB ────────────────────────────────────────────────────
    if (geom_type == "all_regular")
    {
        // Nothing to do — using_levelset_geometry stays false
    }

    // ── wedge_hopper: built-in 3-D geometry ───────────────────────────────────
    else if (geom_type == "wedge_hopper")
    {
#if (AMREX_SPACEDIM == 3)
        using_levelset_geometry = true;
        make_wedge_hopper_levelset(geom, ba, dm);
#else
        amrex::Abort("wedge_hopper geometry is only available in 3D");
#endif
    }

    // ── udf_cpp: runtime user-defined level set ───────────────────────────────
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
        {
            amrex::Abort("[UDF] eb2.udf_so_file must be set when "
                         "eb2.geom_type = udf_cpp");
        }

        // Load the user's shared library — g_udf_loader lives at file scope
        // so the handle stays open for the whole simulation.
        g_udf_loaders[0].load(so_path);

        // ── Build EB index space from UDF (used by ebfactory only) ───────────
        // The UDF function pointer is a CPU symbol — it cannot be called
        // inside a GPU kernel. We use it here only for EB2::Build (which
        // runs on the CPU) and for the lsphi fill below (LoopOnCpu).
        auto phi_fn = g_udf_loaders[0].get_fn();

        auto udf_if = [phi_fn](const amrex::RealArray& p) -> amrex::Real {
            return static_cast<amrex::Real>(phi_fn(p[0], p[1], p[2]));
        };
        auto gshop = EB2::makeShop(udf_if);

        Box dom_ls = geom.Domain();
        dom_ls.refine(ls_refinement);
        Geometry geom_ls(dom_ls);

        int required_coarsening_level = 0;
        if (ls_refinement > 1)
        {
            int tmp = ls_refinement;
            while (tmp >>= 1) ++required_coarsening_level;
        }

        EB2::Build(gshop, geom_ls, required_coarsening_level, 10);

        const EB2::IndexSpace &ebis  = EB2::IndexSpace::top();
        const EB2::Level      &eblev = ebis.getLevel(geom);

        ebfactory = new EBFArrayBoxFactory(
            eblev, geom, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

        // ── Fill lsphi directly from UDF using CPU loop ───────────────────────
        // WHY NOT FillSignedDistance:
        //   FillSignedDistance uses a tiled fast-marching method that only
        //   propagates within each box ghost-cell halo. With small boxes
        //   (max_grid_size=16) and geometry spanning the domain, most boxes
        //   see no EB surface and get zero/incorrect phi values.
        //
        // WHY NOT amrex::ParallelFor:
        //   ParallelFor dispatches to GPU kernels on CUDA/HIP/SYCL builds.
        //   phi_fn is a host-side function pointer loaded via dlopen —
        //   it cannot be called from GPU device code.
        //
        // SOLUTION — amrex::LoopOnCpu:
        //   Forces CPU-side execution regardless of GPU build.
        //   Since lsphi is filled once at initialisation and the UDF is
        //   analytic (O(1) per node), the cost is negligible.
        //   AMReX automatically syncs host->device when lsphi is next
        //   accessed in a GPU kernel (e.g. average_down_nodal).
        BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
        ls_ba.refine(ls_refinement);
        lsphi = new MultiFab;
        lsphi->define(ls_ba, dm, 1, nghost);

        const auto plo = geom.ProbLoArray();
        const auto dx  = geom.CellSizeArray();
        const int  lsr = ls_refinement;

        for (MFIter mfi(*lsphi); mfi.isValid(); ++mfi)
        {
            const Box&   bx  = mfi.fabbox();   // include ghost cells
            Array4<Real> phi = lsphi->array(mfi);

            // LoopOnCpu: always executes on host, never dispatched to GPU
            amrex::LoopOnCpu(bx, [&](int i, int j, int k)
            {
                amrex::Real x = plo[0] + i * dx[0] / lsr;
                amrex::Real y = plo[1] + j * dx[1] / lsr;
                amrex::Real z = (AMREX_SPACEDIM == 3)
                                ? plo[2] + k * dx[2] / lsr
                                : 0.0;
                phi(i, j, k) = static_cast<amrex::Real>(phi_fn(x, y, z));
            });
        }

        amrex::Print() << "[UDF] lsphi filled on CPU (GPU-safe). "
                       << "min=" << lsphi->min(0)
                       << " max=" << lsphi->max(0) << "\n";
    }

    // ── stl: surface mesh from file ───────────────────────────────────────────
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
     *   eb2.stl_reverse   = false   (flip inside/outside if STL normals point inward)
     *   eb2.ls_refinement = 2
     */
    else if (geom_type == "stl")
    {
        using_levelset_geometry = true;

        std::string stl_file;
        bool        stl_reverse = false;

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
            while (tmp >>= 1) ++required_coarsening_level;
        }
        Box      dom_ls = geom.Domain();
        dom_ls.refine(ls_refinement);
        Geometry geom_ls(dom_ls);

        amrex::EB2::Build(geom_ls, required_coarsening_level, 10);

        const EB2::IndexSpace &ebis  = EB2::IndexSpace::top();
        const EB2::Level      &eblev = ebis.getLevel(geom);
        const EB2::Level      &lslev = ebis.getLevel(geom_ls);

        ebfactory = new EBFArrayBoxFactory(
            eblev, geom, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

        BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
        ls_ba.refine(ls_refinement);
        lsphi = new MultiFab;
        lsphi->define(ls_ba, dm, 1, nghost);
        amrex::FillSignedDistance(*lsphi, lslev, *ebfactory, ls_refinement);
#endif
    }

    // ── anything else: pass directly to AMReX EB2 (sphere, plane, etc.) ───────
    else
    {
        using_levelset_geometry = true;

        int required_coarsening_level = 0;
        if (ls_refinement > 1)
        {
            int tmp = ls_refinement;
            while (tmp >>= 1) ++required_coarsening_level;
        }

        Box      dom_ls = geom.Domain();
        dom_ls.refine(ls_refinement);
        Geometry geom_ls(dom_ls);

        amrex::EB2::Build(geom_ls, required_coarsening_level, 10);

        const EB2::IndexSpace &ebis  = EB2::IndexSpace::top();
        const EB2::Level      &eblev = ebis.getLevel(geom);
        const EB2::Level      &lslev = ebis.getLevel(geom_ls);

        ebfactory = new EBFArrayBoxFactory(
            eblev, geom, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

        BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
        ls_ba.refine(ls_refinement);
        lsphi = new MultiFab;
        lsphi->define(ls_ba, dm, 1, nghost);
        amrex::FillSignedDistance(*lsphi, lslev, *ebfactory, ls_refinement);
    }

    // ── write phi plotfile for all geometry types ─────────────────────────────
    if (using_levelset_geometry)
    {
        const std::string &pltfile = "ebplt";
        Box      dom_ls = geom.Domain();
        dom_ls.refine(ls_refinement);
        Geometry geom_ls(dom_ls);
        BoxArray plot_ba = ba;
        plot_ba.refine(ls_refinement);
        MultiFab plotmf(plot_ba, dm, lsphi->nComp(), 0);
        amrex::average_node_to_cellcenter(
            plotmf, 0, *lsphi, 0, lsphi->nComp());
        WriteSingleLevelPlotfile(pltfile, plotmf, {"phi"}, geom_ls, 0.0, 0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
/**
 * @brief Fills lsphi for a single body from mpm.rb_N.* input keys.
 *
 * Reads mpm.rb_N.geom_type and dispatches to the appropriate geometry path.
 * Supported types: udf_cpp, stl, analytic EB2 (sphere, plane, etc.),
 * wedge_hopper.
 *
 * The resulting MultiFab is stored in lsphi_bodies[body_id].
 *
 * @param body_id  Index of the rigid body (0-based).
 * @param geom     Coarse geometry.
 * @param ba       BoxArray.
 * @param dm       DistributionMapping.
 */
static void fill_body_lsphi(int body_id,
                             const Geometry &geom,
                             const BoxArray &ba,
                             const DistributionMapping &dm)
{
    int nghost = 1;
    std::string rb_key   = "mpm.rb_" + std::to_string(body_id);
    std::string geom_type = "udf_cpp";   // default for bodies

    amrex::ParmParse pp_rb(("mpm.rb_" + std::to_string(body_id)).c_str());
    pp_rb.query("geom_type",     geom_type);
    pp_rb.query("ls_refinement", ls_refinement);

    amrex::Print() << "[EB] Body " << body_id
                   << ": geom_type = " << geom_type << "\n";

    // Helper lambda: allocate and fill lsphi_bodies[body_id] on CPU
    auto allocate_lsphi = [&]() -> MultiFab*
    {
        BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
        ls_ba.refine(ls_refinement);
        auto* mf = new MultiFab;
        mf->define(ls_ba, dm, 1, nghost);
        return mf;
    };

    if (geom_type == "udf_cpp")
    {
        std::string so_path;
        if (!pp_rb.query("udf_so_file", so_path))
            amrex::Abort("[UDF] mpm.rb_" + std::to_string(body_id)
                         + ".udf_so_file must be set");

        g_udf_loaders[body_id].load(so_path);
        auto phi_fn = g_udf_loaders[body_id].get_fn();

        // Build EB for ebfactory (only body 0 sets the global ebfactory)
        auto udf_if = [phi_fn](const amrex::RealArray& p) -> amrex::Real {
            return static_cast<amrex::Real>(phi_fn(p[0], p[1], p[2]));
        };
        auto gshop = EB2::makeShop(udf_if);

        Box dom_ls = geom.Domain();
        dom_ls.refine(ls_refinement);
        Geometry geom_ls(dom_ls);
        int required_coarsening_level = 0;
        if (ls_refinement > 1) {
            int tmp = ls_refinement;
            while (tmp >>= 1) ++required_coarsening_level;
        }
        EB2::Build(gshop, geom_ls, required_coarsening_level, 10);

        if (body_id == 0)
        {
            const EB2::IndexSpace &ebis = EB2::IndexSpace::top();
            const EB2::Level &eblev = ebis.getLevel(geom);
            ebfactory = new EBFArrayBoxFactory(
                eblev, geom, ba, dm,
                {nghost, nghost, nghost}, EBSupport::full);
        }

        // Fill lsphi on CPU via LoopOnCpu (GPU-safe)
        lsphi_bodies[body_id] = allocate_lsphi();
        const auto plo = geom.ProbLoArray();
        const auto dx  = geom.CellSizeArray();
        const int  lsr = ls_refinement;

        for (MFIter mfi(*lsphi_bodies[body_id]); mfi.isValid(); ++mfi)
        {
            const Box&   bx  = mfi.fabbox();
            Array4<Real> phi = lsphi_bodies[body_id]->array(mfi);
            amrex::LoopOnCpu(bx, [&](int i, int j, int k)
            {
                amrex::Real x = plo[0] + i * dx[0] / lsr;
                amrex::Real y = plo[1] + j * dx[1] / lsr;
                amrex::Real z = (AMREX_SPACEDIM == 3)
                                ? plo[2] + k * dx[2] / lsr : 0.0;
                phi(i, j, k) = static_cast<amrex::Real>(phi_fn(x, y, z));
            });
        }

        amrex::Print() << "[UDF] Body " << body_id << " lsphi: "
                       << "min=" << lsphi_bodies[body_id]->min(0)
                       << " max=" << lsphi_bodies[body_id]->max(0) << "\n";
    }
    else if (geom_type == "stl")
    {
        std::string stl_file;
        bool stl_reverse = false;
        if (!pp_rb.query("stl_file", stl_file))
            amrex::Abort("[STL] mpm.rb_" + std::to_string(body_id)
                         + ".stl_file must be set");
        pp_rb.query("stl_reverse", stl_reverse);

        amrex::Print() << "[STL] Body " << body_id
                       << " loading: " << stl_file << "\n";

        lsphi_bodies[body_id] = allocate_lsphi();

#if EXAGOOP_AMREX_HAS_STLGEOM
        auto stl_geom = amrex::EB2::STLGeom(stl_file);
        auto gshop = stl_reverse
                     ? EB2::makeShop(EB2::makeComplement(stl_geom))
                     : EB2::makeShop(stl_geom);

        Box dom_ls = geom.Domain();
        dom_ls.refine(ls_refinement);
        Geometry geom_ls(dom_ls);
        int required_coarsening_level = 0;
        if (ls_refinement > 1) {
            int tmp = ls_refinement;
            while (tmp >>= 1) ++required_coarsening_level;
        }
        EB2::Build(gshop, geom_ls, required_coarsening_level, 10);

        const EB2::IndexSpace &ebis = EB2::IndexSpace::top();
        const EB2::Level &eblev = ebis.getLevel(geom);
        const EB2::Level &lslev = ebis.getLevel(geom_ls);

        if (body_id == 0)
            ebfactory = new EBFArrayBoxFactory(
                eblev, geom, ba, dm,
                {nghost, nghost, nghost}, EBSupport::full);

        amrex::FillSignedDistance(*lsphi_bodies[body_id],
                                   lslev, *ebfactory, ls_refinement);
#else
        amrex::Abort("[STL] Body geometry requires AMReX with STLGeom support");
#endif
    }
    else
    {
        // Analytic AMReX EB2 shapes (sphere, plane, cylinder, etc.)
        // or wedge_hopper — use the existing init_eb path via eb2.* keys
        // but override with rb_N keys where present.
        // For simplicity, delegate to init_eb by temporarily setting
        // eb2.geom_type in ParmParse. This reuses all existing geometry code.
        amrex::Print() << "[EB] Body " << body_id
                       << ": using analytic EB2 type '" << geom_type << "'\n";

        // Build EB using existing analytic path
        Box dom_ls = geom.Domain();
        dom_ls.refine(ls_refinement);
        Geometry geom_ls(dom_ls);
        int required_coarsening_level = 0;
        if (ls_refinement > 1) {
            int tmp = ls_refinement;
            while (tmp >>= 1) ++required_coarsening_level;
        }

        // Re-use the eb2 ParmParse namespace for analytic shapes
        // The user sets eb2.geom_type = sphere etc. per body via
        // mpm.rb_N.geom_type, but the EB2 analytic shapes read their
        // parameters from eb2.* — for multiple analytic bodies the user
        // should use udf_cpp with the advanced template instead.
        amrex::EB2::Build(geom_ls, required_coarsening_level, 10);

        const EB2::IndexSpace &ebis = EB2::IndexSpace::top();
        const EB2::Level &eblev = ebis.getLevel(geom);
        const EB2::Level &lslev = ebis.getLevel(geom_ls);

        if (body_id == 0)
            ebfactory = new EBFArrayBoxFactory(
                eblev, geom, ba, dm,
                {nghost, nghost, nghost}, EBSupport::full);

        lsphi_bodies[body_id] = allocate_lsphi();
        amrex::FillSignedDistance(*lsphi_bodies[body_id],
                                   lslev, *ebfactory, ls_refinement);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
/**
 * @brief Initialises one lsphi per rigid body and builds the union lsphi.
 *
 * For each body N:
 *   1. Reads mpm.rb_N.geom_type and associated geometry parameters.
 *   2. Fills lsphi_bodies[N] with the signed distance field for body N.
 *
 * After all bodies are processed, computes the union lsphi:
 *   lsphi(x) = min over all N of lsphi_bodies[N](x)
 *
 * This union lsphi is used by removeParticlesInsideEB (particles inside ANY
 * body are removed) and by the global ebfactory (set from body 0).
 *
 * The existing init_eb() handles the legacy single-body eb2.* path.
 * Call init_eb_bodies() AFTER init_eb() when num_rigidbodies > 0.
 *
 * @param geom         Coarse geometry.
 * @param ba           BoxArray.
 * @param dm           DistributionMapping.
 * @param num_bodies   Number of rigid bodies to initialise.
 */
void init_eb_bodies(const Geometry &geom,
                    const BoxArray &ba,
                    const DistributionMapping &dm,
                    int num_bodies)
{
    if (num_bodies <= 0) return;

    amrex::Print() << "[EB] Initialising " << num_bodies
                   << " rigid body level sets\n";

    using_levelset_geometry = true;
    num_lsphi_bodies = num_bodies;
    lsphi_bodies.resize(num_bodies, nullptr);

    // ── Fill lsphi for each body ──────────────────────────────────────────────
    for (int b = 0; b < num_bodies; ++b)
        fill_body_lsphi(b, geom, ba, dm);

    // ── Build union lsphi = min over all bodies ───────────────────────────────
    // This replaces the global lsphi used by removeParticlesInsideEB.
    if (lsphi != nullptr) delete lsphi;

    BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
    ls_ba.refine(ls_refinement);
    lsphi = new MultiFab;
    lsphi->define(ls_ba, dm, 1, 1);

    // Initialise to +infinity so min() works correctly
    lsphi->setVal(1.0e30);

    // Take element-wise minimum across all bodies
    for (int b = 0; b < num_bodies; ++b)
    {
        for (MFIter mfi(*lsphi); mfi.isValid(); ++mfi)
        {
            const Box&   bx      = mfi.fabbox();
            Array4<Real> phi_union = lsphi->array(mfi);
            Array4<Real> phi_body  = lsphi_bodies[b]->array(mfi);

            amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    phi_union(i,j,k) = amrex::min(phi_union(i,j,k),
                                                   phi_body(i,j,k));
                });
        }
    }

    amrex::Print() << "[EB] Union lsphi: min=" << lsphi->min(0)
                   << " max=" << lsphi->max(0) << "\n";

    // ── Write plotfile for each body and the union ────────────────────────────
    Box      dom_ls = geom.Domain();
    dom_ls.refine(ls_refinement);
    Geometry geom_ls(dom_ls);
    BoxArray plot_ba = ba;
    plot_ba.refine(ls_refinement);

    for (int b = 0; b < num_bodies; ++b)
    {
        MultiFab plotmf(plot_ba, dm, 1, 0);
        amrex::average_node_to_cellcenter(plotmf, 0, *lsphi_bodies[b], 0, 1);
        std::string pltname = "ebplt_rb" + std::to_string(b);
        WriteSingleLevelPlotfile(pltname, plotmf, {"phi"}, geom_ls, 0.0, 0);
    }

    // Union plotfile
    {
        MultiFab plotmf(plot_ba, dm, 1, 0);
        amrex::average_node_to_cellcenter(plotmf, 0, *lsphi, 0, 1);
        WriteSingleLevelPlotfile("ebplt_union", plotmf, {"phi"}, geom_ls, 0.0, 0);
    }
}

} // namespace mpm_ebtools
#endif // USE_EB
