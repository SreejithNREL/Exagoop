// clang-format off
void Initialise_Domain(MPMspecs &specs,
                       Geometry &geom,
					   Geometry &geom_leset,
                       BoxArray &ba,
                       DistributionMapping &dm,
                       int &ng_cells,
					   int &ng_cells_nodaldata,
                       MultiFab &nodaldata,
                       MultiFab &levset_data,
                       amrex::Vector<std::string> &nodaldata_names);

void Initialise_Material_Points(MPMspecs &specs, MPMParticleContainer &mpm_pc, int &steps, Real &time, int &output_it);
void Initialise_Internal_Forces(MPMspecs &specs, MPMParticleContainer &mpm_pc,amrex::MultiFab &nodaldata, amrex::MultiFab &levset_data, Geometry &geom,
		   Geometry &geom_levset);
// clang-format on
