import numpy as np
from sys import argv
from _locale import ABMON_10


def write_inputs_file(ncells_x: int,
                      buffery: int,
                      periodic: int,
                      np_per_cell_x: int,
                      order_scheme: int,
                      alpha_pic_flip: float,
                      stress_update_scheme: int,
                      CFL: float,                      
                      output_tag: str,
                      dx1: float,
                      out_filename: str = "inputs_axialbar.inp") -> None:
    """
    Write inputs_axialbar file using provided parameters. Creates Solution/<output_tag> directory if needed.
    """
    bufferx = 4
    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 1.0
    zmin = 0.0
    zmax = 0.05

    sol_dir = os.path.join(".", "Solution", output_tag)
    try:
        os.makedirs(sol_dir, exist_ok=True)
    except Exception as e:
        warn(f"Could not create Solution directory '{sol_dir}': {e}")

    try:
        with open(out_filename, "w", encoding="utf-8") as f:
            f.write("#geometry parameters\n")
            f.write(f"mpm.prob_lo = {xmin} {ymin} {zmin}\t\t\t#Lower corner of physical domain\n")
            f.write(f"mpm.prob_hi = {xmax} {ymax} {zmax}\t\t\t#Upper corner of physical domain\n")
            f.write(f"mpm.ncells  = {ncells_x + bufferx} {2 * buffery + 1} {2 * buffery + 1}\n")
            f.write(f"mpm.max_grid_size = {ncells_x + bufferx + 1}\n")
            f.write(f"mpm.is_it_periodic = 0  {periodic}  {periodic}\n")

            f.write("\n\n#AMR Parameters\n")
            f.write("#restart_checkfile = \"\"\n")

            f.write("\n\n#Input files\n")
            f.write("mpm.use_autogen=0\n")
            f.write("mpm.mincoords_autogen=0.0 0.0 0.0\n")
            f.write("mpm.maxcoords_autogen=1.0 1.0 1.0\n")
            f.write("mpm.vel_autogen=0.0 0.0 0.0\n")
            f.write("mpm.constmodel_autogen=0\n")
            f.write("mpm.dens_autogen=1.0\n")
            f.write("mpm.E_autogen=1e6\n")
            f.write("mpm.nu_autogen=0.3\n")
            f.write("mpm.bulkmod_autogen=2e6\n")
            f.write("mpm.Gama_pres_autogen=7\n")
            f.write("mpm.visc_autogen=0.001\n")
            f.write("mpm.multi_part_per_cell_autogen=1\n")
            f.write("mpm.particle_file=\"mpm_particles.dat\"\n")

            f.write("\n\n#File output parameters\n")
            f.write(f"#mpm.prefix_particlefilename=\"./Solution/{output_tag}/plt\"\n")
            f.write(f"#mpm.prefix_gridfilename=\"./Solution/{output_tag}/nplt\"\n")
            f.write(f"#mpm.prefix_densityfilename=\"./Solution/{output_tag}/dens\"\n")
            f.write(f"#mpm.prefix_checkpointfilename=\"./Solution/{output_tag}/chk\"\n")
            f.write("mpm.num_of_digits_in_filenames=6\n")

            f.write("\n\n#Simulation run parameters\n")
            f.write("mpm.final_time=50.0\n")
            f.write("mpm.max_steps=5000000\n")
            f.write("mpm.screen_output_time = 0.001\n")
            f.write("mpm.write_output_time=0.5\n")
            f.write("mpm.num_redist = 1\n")

            f.write("\n\n#Timestepping parameters\n")
            f.write("mpm.fixed_timestep = 0\n")
            f.write("mpm.timestep = 1.0e-5\n")
            f.write(f"mpm.CFL={CFL}\n")
            f.write("mpm.dt_min_limit=1e-12\n")
            f.write("mpm.dt_max_limit=1e+00\n")

            f.write("\n\n#Numerical schemes\n")
            f.write(f"mpm.order_scheme={order_scheme}\n")
            f.write(f"mpm.alpha_pic_flip = {alpha_pic_flip}\n")
            f.write(f"mpm.stress_update_scheme= {stress_update_scheme}\n")
            f.write("mpm.mass_tolerance = 1e-18\n")

            f.write("\n\n#Physics parameters\n")
            f.write("mpm.gravity = 0.0 0.0 0.0\n")
            f.write("mpm.applied_strainrate_time=0.0\n")
            f.write("mpm.applied_strainrate=0.0\n")
            f.write("mpm.external_loads=0\n")
            f.write("mpm.force_slab_lo= 0.0 0.0 0.0\n")
            f.write("mpm.force_slab_hi= 1.0 1.0 1.0\n")
            f.write("mpm.extforce = 0.0 0.0 0.0\n")

            f.write("\n\n#Diagnostics and Test\n")
            f.write("mpm.print_diagnostics= 1\n")            
            f.write("mpm.do_calculate_tke_tse= 1\n")
            f.write("mpm.write_diag_output_time= 0.01\n")

            f.write("\n\n#Boundary conditions\n")
            f.write("mpm.bc_lower=0 0 0\n")
            f.write("mpm.bc_upper=0 0 0\n")

    except Exception as e:
        die(f"Failed to write inputs file '{out_filename}': {e}")

    print(f"WROTE: {out_filename}")

L = 1.0
no_of_cells_in_z = 1

# Use the same parameters as in input file
ncell_x = 20
ncell_y = 20
ncell_z = no_of_cells_in_z
dx2 = L/ncell_x

blo    = np.array([float(0.0),float(0.0),float(0.0)])
bhi    = np.array([L,L,float(dx2*ncell_z)])
ncells = np.array([ncell_x,ncell_y,ncell_z])
npart  = 0 
dx = (bhi-blo)/ncells;
if(dx[0]!=dx[1] or dx[0]!=dx[2] or dx[1]!=dx[2]):
    print("Error! mesh sizes are not same in all directions",dx[0],dx[1],dx[2])
nparticle_per_cells_eachdir=4

dim = 3
xmin=0.0
xmax=L
ymin=0.0
ymax=L
zmin=0.0
zmax=dx2*ncell_z

print(range(nparticle_per_cells_eachdir))

print('Number of particles = ',npart)
outfile=open("mpm_particles.dat","w")

factor=[0.125,0.375,0.625,0.875]

dens=997.5
phase=0
rad=0.025
E = 1000
nu=0.3
#Volume in each cell
xc=0.2
yc=0.2
zc=0.7
vol_cell=dx[0]*dx[1]*dx[2]
vol_particle=vol_cell/(nparticle_per_cells_eachdir*nparticle_per_cells_eachdir*nparticle_per_cells_eachdir)
rad=(3.0/4.0*vol_particle/3.1416)**(1.0/3.0)
npart=0

for k in range(ncells[2]):
    for j in range(ncells[1]):
        for i in range(ncells[0]):
            c_cx=blo[0]+(i)*dx[0]
            c_cy=blo[1]+(j)*dx[1]
            c_cz=blo[2]+(k)*dx[2]
            if(c_cx>=xmin and c_cx<xmax and c_cy>=ymin and c_cy<ymax and c_cz>=zmin and c_cz<zmax):
                for ii in range(nparticle_per_cells_eachdir):
                    for jj in range(nparticle_per_cells_eachdir):
                        for kk in range(nparticle_per_cells_eachdir):
                            
                            #cell_cx=c_cx+(2*ii+1)*dx[0]/(2.0*nparticle_per_cells_eachdir)
                            cell_cx=c_cx+factor[ii]*dx[0]
                            #cell_cy=c_cy+(2*jj+1)*dx[1]/(2.0*nparticle_per_cells_eachdir)
                            cell_cy=c_cy+factor[jj]*dx[1]
                            #cell_cz=c_cz+(2*kk+1)*dx[2]/(2.0*nparticle_per_cells_eachdir)
                            cell_cz=c_cz+factor[kk]*dx[2]
                            
                            if(((cell_cx-xc)*(cell_cx-xc)+(cell_cy-yc)*(cell_cy-yc))**0.5<=0.2):
                                npart=npart+1                                
xc=0.8
yc=0.8
zc=0.7
for k in range(ncells[2]):
    for j in range(ncells[1]):
        for i in range(ncells[0]):
            c_cx=blo[0]+(i)*dx[0]
            c_cy=blo[1]+(j)*dx[1]
            c_cz=blo[2]+(k)*dx[2]
            if(c_cx>=xmin and c_cx<xmax and c_cy>=ymin and c_cy<ymax and c_cz>=zmin and c_cz<zmax):
                for ii in range(nparticle_per_cells_eachdir):
                    for jj in range(nparticle_per_cells_eachdir):
                        for kk in range(nparticle_per_cells_eachdir):
                            
                            #cell_cx=c_cx+(2*ii+1)*dx[0]/(2.0*nparticle_per_cells_eachdir)
                            cell_cx=c_cx+factor[ii]*dx[0]
                            #cell_cy=c_cy+(2*jj+1)*dx[1]/(2.0*nparticle_per_cells_eachdir)
                            cell_cy=c_cy+factor[jj]*dx[1]
                            #cell_cz=c_cz+(2*kk+1)*dx[2]/(2.0*nparticle_per_cells_eachdir)
                            cell_cz=c_cz+factor[kk]*dx[2]
                            
                            if(((cell_cx-xc)*(cell_cx-xc)+(cell_cy-yc)*(cell_cy-yc))**0.5<=0.2):
                                npart=npart+1
                                

outfile.write("%d\n"%(npart));

xc=0.2
yc=0.2
zc=0.7
vol_cell=dx[0]*dx[1]*dx[2]
vol_particle=vol_cell/(nparticle_per_cells_eachdir*nparticle_per_cells_eachdir*nparticle_per_cells_eachdir)
rad=(3.0/4.0*vol_particle/3.1416)**(1.0/3.0)
npart=0

for k in range(ncells[2]):
    for j in range(ncells[1]):
        for i in range(ncells[0]):
            c_cx=blo[0]+(i)*dx[0]
            c_cy=blo[1]+(j)*dx[1]
            c_cz=blo[2]+(k)*dx[2]
            if(c_cx>=xmin and c_cx<xmax and c_cy>=ymin and c_cy<ymax and c_cz>=zmin and c_cz<zmax):
                for ii in range(nparticle_per_cells_eachdir):
                    for jj in range(nparticle_per_cells_eachdir):
                        for kk in range(nparticle_per_cells_eachdir):
                            
                            #cell_cx=c_cx+(2*ii+1)*dx[0]/(2.0*nparticle_per_cells_eachdir)
                            cell_cx=c_cx+factor[ii]*dx[0]
                            #cell_cy=c_cy+(2*jj+1)*dx[1]/(2.0*nparticle_per_cells_eachdir)
                            cell_cy=c_cy+factor[jj]*dx[1]
                            #cell_cz=c_cz+(2*kk+1)*dx[2]/(2.0*nparticle_per_cells_eachdir)
                            cell_cz=c_cz+factor[kk]*dx[2]
                            
                            if(((cell_cx-xc)*(cell_cx-xc)+(cell_cy-yc)*(cell_cy-yc))**0.5<=0.2):
                                
                                velx=0.1;
                                vely=0.1;
                                velz=0.0;

                                outfile.write("%d\t%e\t%e\t"%(phase,cell_cx,cell_cy));
                                if(dim==3):
                                   outfile.write("%e\t"%(cell_cz))
                                outfile.write("%e\t%e\t"%(rad,dens));
                                outfile.write("%e\t%e\t"%(velx,vely));
                                if(dim==3):
                                   outfile.write("%e\t"%(velz))
                                outfile.write("%d\t%e\t%e\n"%(0,E,nu));                                

xc=0.8
yc=0.8
zc=0.7
for k in range(ncells[2]):
    for j in range(ncells[1]):
        for i in range(ncells[0]):
            c_cx=blo[0]+(i)*dx[0]
            c_cy=blo[1]+(j)*dx[1]
            c_cz=blo[2]+(k)*dx[2]
            if(c_cx>=xmin and c_cx<xmax and c_cy>=ymin and c_cy<ymax and c_cz>=zmin and c_cz<zmax):
                for ii in range(nparticle_per_cells_eachdir):
                    for jj in range(nparticle_per_cells_eachdir):
                        for kk in range(nparticle_per_cells_eachdir):
                            
                            #cell_cx=c_cx+(2*ii+1)*dx[0]/(2.0*nparticle_per_cells_eachdir)
                            cell_cx=c_cx+factor[ii]*dx[0]
                            #cell_cy=c_cy+(2*jj+1)*dx[1]/(2.0*nparticle_per_cells_eachdir)
                            cell_cy=c_cy+factor[jj]*dx[1]
                            #cell_cz=c_cz+(2*kk+1)*dx[2]/(2.0*nparticle_per_cells_eachdir)
                            cell_cz=c_cz+factor[kk]*dx[2]
                            
                            if(((cell_cx-xc)*(cell_cx-xc)+(cell_cy-yc)*(cell_cy-yc))**0.5<=0.2):
                                
                                velx=-0.1;
                                vely=-0.1;
                                velz=0.0;
                                
                                outfile.write("%d\t%e\t%e\t"%(phase,cell_cx,cell_cy));
                                if(dim==3):
                                   outfile.write("%e\t"%(cell_cz))
                                outfile.write("%e\t%e\t"%(rad,dens));
                                outfile.write("%e\t%e\t"%(velx,vely));                                
                                if(dim==3):
                                   outfile.write("%e\t"%(velz))
                                outfile.write("%d\t%e\t%e\n"%(0,E,nu));

print(npart)
outfile.close()
