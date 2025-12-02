import numpy as np
from sys import argv
from _locale import ABMON_10

L = 1.0

# Use the same parameters as in input file
ncell_x = 20
ncell_y = 20

dx2 = L/ncell_x

blo    = np.array([float(0.0),float(0.0),float(0.0)])
bhi    = np.array([L,L,float(1)])
ncells = np.array([ncell_x,ncell_y,1])
npart  = 0 
dx = (bhi-blo)/ncells;
if(dx[0]!=dx[1]):
    print("Error! mesh sizes are not same in all directions",dx[0],dx[1],dx[2])
nparticle_per_cells_eachdir=4

dim = 2
xmin=0.0
xmax=L
ymin=0.0
ymax=L

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

vol_cell=dx[0]*dx[1]
vol_particle=vol_cell/(nparticle_per_cells_eachdir*nparticle_per_cells_eachdir)
rad=(4.0*vol_particle/3.1416)**(1.0/2.0)
npart=0

for j in range(ncells[1]):
    for i in range(ncells[0]):
        c_cx=blo[0]+(i)*dx[0]
        c_cy=blo[1]+(j)*dx[1]        
        if(c_cx>=xmin and c_cx<xmax and c_cy>=ymin and c_cy<ymax):
            for ii in range(nparticle_per_cells_eachdir):
                for jj in range(nparticle_per_cells_eachdir):
                    cell_cx=c_cx+factor[ii]*dx[0]
                    cell_cy=c_cy+factor[jj]*dx[1]
                    
                    if(((cell_cx-xc)*(cell_cx-xc)+(cell_cy-yc)*(cell_cy-yc))**0.5<=0.2):
                        npart=npart+1                                
xc=0.8
yc=0.8

for j in range(ncells[1]):
    for i in range(ncells[0]):
        c_cx=blo[0]+(i)*dx[0]
        c_cy=blo[1]+(j)*dx[1]

        if(c_cx>=xmin and c_cx<xmax and c_cy>=ymin and c_cy<ymax):
            for ii in range(nparticle_per_cells_eachdir):
                for jj in range(nparticle_per_cells_eachdir):                    
                    cell_cx=c_cx+factor[ii]*dx[0]                    
                    cell_cy=c_cy+factor[jj]*dx[1]

                    if(((cell_cx-xc)*(cell_cx-xc)+(cell_cy-yc)*(cell_cy-yc))**0.5<=0.2):
                        npart=npart+1
                        
outfile.write("%d\n"%(npart))

xc=0.2
yc=0.2

for j in range(ncells[1]):
    for i in range(ncells[0]):
        c_cx=blo[0]+(i)*dx[0]
        c_cy=blo[1]+(j)*dx[1]
        
        if(c_cx>=xmin and c_cx<xmax and c_cy>=ymin and c_cy<ymax):
            for ii in range(nparticle_per_cells_eachdir):
                for jj in range(nparticle_per_cells_eachdir):
                    cell_cx=c_cx+factor[ii]*dx[0]
                    cell_cy=c_cy+factor[jj]*dx[1]
                    if(((cell_cx-xc)*(cell_cx-xc)+(cell_cy-yc)*(cell_cy-yc))**0.5<=0.2):
                        velx=0.1
                        vely=0.1;
                        
                        outfile.write("%d\t%e\t%e\t"%(phase,cell_cx,cell_cy));
                        outfile.write("%e\t%e\t"%(rad,dens));
                        outfile.write("%e\t%e\t"%(velx,vely));
                        outfile.write("%d\t%e\t%e\n"%(0,E,nu));                                
xc=0.8
yc=0.8

for j in range(ncells[1]):
    for i in range(ncells[0]):
        c_cx=blo[0]+(i)*dx[0]
        c_cy=blo[1]+(j)*dx[1]

        if(c_cx>=xmin and c_cx<xmax and c_cy>=ymin and c_cy<ymax ):
            for ii in range(nparticle_per_cells_eachdir):
                for jj in range(nparticle_per_cells_eachdir):                    
                    cell_cx=c_cx+factor[ii]*dx[0]                   
                    cell_cy=c_cy+factor[jj]*dx[1]

                    if(((cell_cx-xc)*(cell_cx-xc)+(cell_cy-yc)*(cell_cy-yc))**0.5<=0.2):
                        velx=-0.1
                        vely=-0.1

                        outfile.write("%d\t%e\t%e\t"%(phase,cell_cx,cell_cy));
                        outfile.write("%e\t%e\t"%(rad,dens));
                        outfile.write("%e\t%e\t"%(velx,vely));
                        outfile.write("%d\t%e\t%e\n"%(0,E,nu));

print(npart)
outfile.close()
