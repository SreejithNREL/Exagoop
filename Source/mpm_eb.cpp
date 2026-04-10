// clang-format off
/**
 * @file mpm_eb.cpp
 *
 * @brief Initializes the embedded boundary (EB) geometry and nodal level-set
 *        MultiFab (lsphi) for ExaGOOP.
 *
 * Supports three geometry input paths, selected by eb2.geom_type:
 *
 *   Path A  "udf_cpp"   — user-provided C/C++ shared library (.so/.dylib/.dll)
 *                         exporting:
 *                           extern "C" double levelset_phi(double x,
 *                                                          double y,
 *                                                          double z);
 *
 *   Path B  "stl"       — STL surface mesh file; uses AMReX EB2::STLGeom.
 *
 *   Path C  anything else — AMReX built-in analytic shapes (sphere, cylinder,
 *                           plane, box, wedge_hopper, etc.) read directly from
 *                           the eb2.* ParmParse namespace.
 *
 * All three paths produce the same outputs:
 *   mpm_ebtools::ebfactory   — EBFArrayBoxFactory at coarse resolution
 *   mpm_ebtools::lsphi       — nodal MultiFab at ls_refinement * coarse res
 *
 * Input keys (eb2 namespace):
 *   eb2.geom_type      = udf_cpp | stl | sphere | cylinder | ... (required)
 *   eb2.ls_refinement  = 1                 (optional, default 1)
 *   eb2.udf_so_file    = /path/to/lib.so   (Path A only)
 *   eb2.stl_file       = /path/to/geo.stl  (Path B only)
 *   (all other eb2.* keys forwarded to AMReX for Path C)
 */
// clang-format on

#include <mpm_eb.H>
#include <mpm_udf_loader.H>

#if USE_EB
#include <AMReX_EB2.H>
#include <AMReX_EB_utils.H>
#include <AMReX_MultiFabUtil.H>
#endif

#include <AMReX_PlotFileUtil.H>

#if USE_EB
namespace mpm_ebtools
{

EBFArrayBoxFactory *ebfactory = nullptr;
MultiFab *lsphi = nullptr;
int ls_refinement = 1;
bool using_levelset_geometry = false;


/**
 * @brief Computes required_coarsening_level = log2(ls_ref).
 *
 * AMReX's EB2::Build requires this to be at least log2(ls_refinement)
 * so it can coarsen the refined index space back to the coarse level.
 */
static int coarsening_level_for_refinement(int ls_ref)
{
    int level = 0;
    if (ls_ref > 1)
    {
        int tmp = ls_ref;
        while (tmp >>= 1)
            ++level;
    }
    return level;
}

/**
 * @brief Builds the EBFArrayBoxFactory from the current EB2::IndexSpace top,
 *        and allocates (but does NOT fill) the nodal lsphi MultiFab.
 *
 * Called by all three paths after EB2::Build (or its equivalent) has been
 * invoked and the desired index space is at EB2::IndexSpace::top().
 */
static void build_factory_and_lsphi(const Geometry &geom,
                                    const BoxArray &ba,
                                    const DistributionMapping &dm,
                                    int nghost,
                                    int ls_ref)
{
    const EB2::IndexSpace &ebis = EB2::IndexSpace::top();
    const EB2::Level &eblev = ebis.getLevel(geom);

    ebfactory = new EBFArrayBoxFactory(
        eblev, geom, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

    BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
    ls_ba.refine(ls_ref);

    lsphi = new MultiFab;
    lsphi->define(ls_ba, dm, /*ncomp=*/1, nghost);
}

/**
 * @brief Returns a refined Geometry with the domain grown by ls_ref.
 */
static Geometry refined_geom(const Geometry &geom, int ls_ref)
{
    Box dom_ls = geom.Domain();
    dom_ls.refine(ls_ref);
    return Geometry(dom_ls);
}

// ---------------------------------------------------------------
// Path A — UDF C/C++ shared library
// ---------------------------------------------------------------

/**
 * @brief Builds EB and fills lsphi from a user-provided shared-library UDF.
 *
 * The UDF must export:
 *   extern "C" double levelset_phi(double x, double y, double z);
 *
 * EB2::Build for the UDF is performed inside build_udf_eb() which lives in
 * mpm_eb_udf_build.cpp (CXX-only TU — never compiled by nvcc).
 *
 * lsphi is filled here via amrex::LoopOnCpu (CPU only — function pointers
 * cannot be called on the GPU).  FillBoundary is called afterwards (Bug 2 fix).
 */
static void build_udf_levelset(const Geometry &geom,
                               const BoxArray &ba,
                               const DistributionMapping &dm,
                               int nghost,
                               int ls_ref)
{
    std::string so_file;
    amrex::ParmParse pp("eb2");
    pp.get("udf_so_file", so_file);

    amrex::Print() << "\n[EB] Reading from UDF shared library: " << so_file << "\n";

    UDFLoader loader(so_file);
    UDFImplicitFunction udf_if(loader);

    build_udf_eb(udf_if, geom, ba, dm, nghost, ls_ref, lsphi, ebfactory);

    Geometry geom_ls = refined_geom(geom, ls_ref);
    const auto plo = geom_ls.ProbLoArray();
    const auto dx_ls = geom_ls.CellSizeArray();

    for (MFIter mfi(*lsphi); mfi.isValid(); ++mfi)
    {
        auto arr = lsphi->array(mfi);
        const Box &bx = mfi.fabbox(); // includes ghost cells

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
    lsphi->FillBoundary(geom_ls.periodicity());
}

/**
 * @brief Builds EB and fills lsphi from an STL surface mesh.
 *
 * Uses AMReX's EB2::STLGeom (requires AMReX compiled with STL support).
 * lsphi is filled via FillSignedDistance (same as Path C).
 */
static void build_stl_levelset(const Geometry &geom,
                               const BoxArray &ba,
                               const DistributionMapping &dm,
                               int nghost,
                               int ls_ref)
{
#ifndef AMREX_USE_EB
    amrex::Abort("build_stl_levelset: AMReX was not compiled with EB support");
#else
    std::string stl_file;
    amrex::ParmParse pp("eb2");
    pp.get("stl_file", stl_file);

    amrex::Print() << "\n[EB] Reading STL file: " << stl_file << "\n";


    Geometry geom_ls = refined_geom(geom, ls_ref);
    int req_coarsen = coarsening_level_for_refinement(ls_ref);

    amrex::EB2::Build(geom_ls, req_coarsen, /*max_coarsening_level=*/10);

    const EB2::IndexSpace &ebis = EB2::IndexSpace::top();
    const EB2::Level &eblev = ebis.getLevel(geom);
    const EB2::Level &lslev = ebis.getLevel(geom_ls);

    ebfactory = new EBFArrayBoxFactory(
        eblev, geom, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

    BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
    ls_ba.refine(ls_ref);
    lsphi = new MultiFab;
    lsphi->define(ls_ba, dm, 1, nghost);

    amrex::FillSignedDistance(*lsphi, lslev, *ebfactory, ls_ref);
    lsphi->FillBoundary(geom_ls.periodicity());
#endif
}

// ---------------------------------------------------------------
// Path C — AMReX built-in analytic geometry
// ---------------------------------------------------------------

/**
 * @brief Builds EB and fills lsphi using AMReX's built-in EB2 shapes.
 *
 * All eb2.* ParmParse keys (sphere_radius, cylinder_radius, etc.) are
 * read by AMReX::EB2::Build internally.  ExaGOOP only needs to drive
 * the build and fill lsphi.
 *
 * Special case: "wedge_hopper" is assembled from EB2 primitives here
 * rather than as a separate function, keeping all EB init in one place.
 */
static void build_analytic_levelset(const std::string &geom_type,
                                    const Geometry &geom,
                                    const BoxArray &ba,
                                    const DistributionMapping &dm,
                                    int nghost,
                                    int ls_ref)
{
    amrex::Print() << "[EB] Path C — AMReX built-in geometry: " << geom_type
                   << "\n";

    Geometry geom_ls = refined_geom(geom, ls_ref);
    int req_coarsen = coarsening_level_for_refinement(ls_ref);

    if (geom_type == "wedge_hopper")
    {
#if (AMREX_SPACEDIM == 3)
        // ---- wedge hopper assembled from EB2 primitives ----
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
    else
    {
        // All other AMReX built-in types: sphere, cylinder, plane, box, etc.
        // AMReX reads the shape parameters from the eb2.* ParmParse namespace.
        EB2::Build(geom_ls, req_coarsen, 10);
    }

    // Common: factory + lsphi allocation + FillSignedDistance
    const EB2::IndexSpace &ebis = EB2::IndexSpace::top();
    const EB2::Level &eblev = ebis.getLevel(geom);
    const EB2::Level &lslev = ebis.getLevel(geom_ls);

    ebfactory = new EBFArrayBoxFactory(
        eblev, geom, ba, dm, {nghost, nghost, nghost}, EBSupport::full);

    BoxArray ls_ba = amrex::convert(ba, IntVect::TheNodeVector());
    ls_ba.refine(ls_ref);
    lsphi = new MultiFab;
    lsphi->define(ls_ba, dm, 1, nghost);

    amrex::FillSignedDistance(*lsphi, lslev, *ebfactory, ls_ref);

    // Explicit FillBoundary for correctness across MPI ranks / periodicity
    lsphi->FillBoundary(geom_ls.periodicity());
}

// ---------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------

/**
 * @brief Initializes EB geometry and level-set MultiFab.
 *
 * Reads eb2.geom_type and dispatches to Path A (udf_cpp), Path B (stl),
 * or Path C (AMReX built-in).  On exit, mpm_ebtools::ebfactory and
 * mpm_ebtools::lsphi are fully initialised and ready for use.
 *
 * Also writes a plot file "ebplt" with the cell-centred signed distance field
 * for visualisation.
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

    std::string geom_type = "all_regular";
    amrex::ParmParse pp("eb2");
    pp.query("geom_type", geom_type);
    pp.query("ls_refinement", ls_refinement);

    if (geom_type == "all_regular")
    {
        amrex::Print() << "[EB] geom_type = all_regular — no EB geometry\n";
        return; // using_levelset_geometry stays false
    }

    using_levelset_geometry = true;

    // Dispatch to the appropriate path
    if (geom_type == "udf_cpp")
    {
        build_udf_levelset(geom, ba, dm, nghost, ls_refinement);
    }
    else if (geom_type == "stl")
    {
        build_stl_levelset(geom, ba, dm, nghost, ls_refinement);
    }
    else
    {
        build_analytic_levelset(geom_type, geom, ba, dm, nghost, ls_refinement);
    }

    // Diagnostic plotfile — cell-centred signed distance field
    {
        Geometry geom_ls = refined_geom(geom, ls_refinement);
        BoxArray plot_ba = ba;
        plot_ba.refine(ls_refinement);
        MultiFab plotmf(plot_ba, dm, lsphi->nComp(), 0);
        amrex::average_node_to_cellcenter(plotmf, 0, *lsphi, 0, lsphi->nComp());
        WriteSingleLevelPlotfile("ebplt", plotmf, {"phi"}, geom_ls, 0.0, 0);
    }
}

} // namespace mpm_ebtools
#endif // USE_EB
