#if USE_EB
#include <AMReX_EB2.H>
#include <functional>
#include <mpm_eb.H>

void build_udf_eb_only(

    std::function<amrex::Real(const amrex::RealArray &)> udf_if,
    const amrex::Geometry &geom,
    const amrex::BoxArray &ba,
    const amrex::DistributionMapping &dm,
    int nghost,
    int ls_refinement_in,
    amrex::MultiFab *&lsphi_out,
    amrex::EBFArrayBoxFactory *&ebfactory_out)
{
    auto gshop = amrex::EB2::makeShop(udf_if);

    amrex::Box dom_ls = geom.Domain();
    dom_ls.refine(ls_refinement_in);
    amrex::Geometry geom_ls(dom_ls);

    int required_coarsening_level = 0;
    if (ls_refinement_in > 1)
    {
        int tmp = ls_refinement_in;
        while (tmp >>= 1)
            ++required_coarsening_level;
    }

    amrex::EB2::Build(gshop, geom_ls, required_coarsening_level, 10);

    const amrex::EB2::IndexSpace &ebis = amrex::EB2::IndexSpace::top();
    const amrex::EB2::Level &eblev = ebis.getLevel(geom);

    ebfactory_out = new amrex::EBFArrayBoxFactory(
        eblev, geom, ba, dm, {nghost, nghost, nghost}, amrex::EBSupport::full);

    amrex::BoxArray ls_ba = amrex::convert(ba, amrex::IntVect::TheNodeVector());
    ls_ba.refine(ls_refinement_in);
    lsphi_out = new amrex::MultiFab;
    lsphi_out->define(ls_ba, dm, 1, nghost);

}
#endif
