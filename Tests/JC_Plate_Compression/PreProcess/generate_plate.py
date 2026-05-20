#!/usr/bin/env python3
"""
PreProcessor for JC_Plate_Compression.

Generates the particle file and AMReX input file for a 10x10 mm square
plate under gravitational compression (Section 10.3.3, Nguyen et al. 2023).

Units: mm, ms, kg  =>  stress in GPa, density in kg/mm^3.

Material parameters (Table 10.3, copper-like):
  rho  = 8.94e-6  kg/mm^3
  E    = 115      GPa
  nu   = 0.31
  JC:  A=0.065 GPa, B=0.356 GPa, n=0.37, C=0, m=0
  EOS: c0=3.586 mm/ms, S_alpha=1.50, Gamma0=0
  Thermal: cp=3.84e-4 GPa.mm^3/(kg.K) = 384 J/(kg.K) in SI,
           k=3.86e-4 kW/(mm.K),  chi=0.9,  T0=0 degC
"""

import json
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")

with open(CONFIG_FILE) as f:
    cfg = json.load(f)

g = cfg["grid"]
gxmin, gxmax = g["xmin"], g["xmax"]
gymin, gymax = g["ymin"], g["ymax"]
gzmin, gzmax = g["zmin"], g["zmax"]
nx, ny, nz = g["nx"], g["ny"], g["nz"]
dx = (gxmax - gxmin) / nx
dy = (gymax - gymin) / ny
dz = (gzmax - gzmin) / nz

ppc = cfg["ppc"]
ppx, ppy, ppz = ppc[0], ppc[1], ppc[2]

body = cfg["bodies"][0]
sh = body["shape"]
xmin, xmax = sh["xmin"], sh["xmax"]
ymin, ymax = sh["ymin"], sh["ymax"]
zmin, zmax = sh["zmin"], sh["zmax"]
cm   = body["constitutive_model"]
temp = body["temperature"]

rho      = cm["density"]
E        = cm["E"]
nu       = cm["nu"]
JC_A     = cm["JC_A"]
JC_B     = cm["JC_B"]
JC_n     = cm["JC_n"]
JC_C     = cm["JC_C"]
JC_m     = cm["JC_m"]
JC_edot0 = cm["JC_eps_dot_0"]
JC_Tr    = cm["JC_Tr"]
JC_Tm    = cm["JC_Tm"]
JC_chi   = cm["JC_chi"]
JC_c0    = cm["JC_c0"]
JC_Sa    = cm["JC_Salpha"]
JC_G0    = cm["JC_Gamma0"]

T0       = temp["T"]
cp       = temp["spheat"]
ktherm   = temp["thermcond"]

vol_p = (dx * dy * dz) / (ppx * ppy * ppz)
mass_p = rho * vol_p
radius = (3.0 * vol_p / (4.0 * np.pi)) ** (1.0 / 3.0)

def ppc_offsets(n):
    return (2.0 * np.arange(1, n + 1) - 1.0) / (2.0 * n)

ox = ppc_offsets(ppx)
oy = ppc_offsets(ppy)
oz = ppc_offsets(ppz)

particles = []
for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            for fx in ox:
                for fy in oy:
                    for fz in oz:
                        x = gxmin + (ix + fx) * dx
                        y = gymin + (iy + fy) * dy
                        z = gzmin + (iz + fz) * dz
                        if (xmin <= x <= xmax and
                            ymin <= y <= ymax and
                            zmin <= z <= zmax):
                            particles.append((x, y, z))

npart = len(particles)
print(f"Generated {npart} particles")

out_dir = os.path.join(SCRIPT_DIR, "..")
part_file = os.path.join(out_dir, "mpm_particles.dat")

colnames = ("phase x y z radius density vx vy vz cm_id "
            "E nu "
            "JC_A JC_B JC_n JC_C JC_m JC_eps_dot_0 "
            "JC_Tr JC_Tm JC_chi JC_c0 JC_Salpha JC_Gamma0 "
            "T spheat thermcond heatsrc")

with open(part_file, "w") as f:
    f.write("dim: 3\n")
    f.write(f"number_of_material_points: {npart}\n")
    f.write(f"# {colnames}\n")
    for (x, y, z) in particles:
        f.write(
            f"0 "
            f"{x:.6e} {y:.6e} {z:.6e} "
            f"{radius:.6e} "
            f"{rho:.6e} "
            f"0.0 0.0 0.0 "
            f"2 "
            f"{E:.6e} {nu:.6e} "
            f"{JC_A:.6e} {JC_B:.6e} {JC_n:.6e} {JC_C:.6e} {JC_m:.6e} "
            f"{JC_edot0:.6e} "
            f"{JC_Tr:.6e} {JC_Tm:.6e} {JC_chi:.6e} "
            f"{JC_c0:.6e} {JC_Sa:.6e} {JC_G0:.6e} "
            f"{T0:.6e} {cp:.6e} {ktherm:.6e} 0.0\n"
        )

print(f"Particle file written to: {part_file}")

phys = cfg["physics"]
inp_file = os.path.join(out_dir, cfg["input_filename"])
with open(inp_file, "w") as f:
    f.write(f"""# ============================================================
# JC Plate under gravitational compression
# Section 10.3.3, Nguyen et al. 2023 (Table 10.3)
# Units: mm, ms, kg  (stress in GPa)
# ============================================================

mpm.prob_lo        = {gxmin} {gymin} {gzmin}
mpm.prob_hi        = {gxmax} {gymax} {gzmax}
mpm.ncells         = {nx} {ny} {nz}
mpm.is_it_periodic = 0 0 0
mpm.max_grid_size  = 8

mpm.max_steps      = 100000
mpm.final_time     = {phys["final_time"]}
mpm.write_output_time = {phys["write_output_time"]}
mpm.screen_output_time = {phys["screen_output_time"]}

mpm.CFL            = {cfg["CFL"]}
mpm.order_scheme   = {cfg["order_scheme"]}
mpm.stress_update_scheme = {cfg["stress_update_scheme"]}

mpm.gravity        = 0.0 {phys["gravity"][1]} 0.0

mpm.particle_file  = mpm_particles.dat

mpm.bc_xlo_mom     = slip
mpm.bc_xhi_mom     = slip
mpm.bc_ylo_mom     = noslip
mpm.bc_yhi_mom     = slip
mpm.bc_zlo_mom     = slip
mpm.bc_zhi_mom     = slip

mpm.bc_xlo_temp    = adiabatic
mpm.bc_xhi_temp    = adiabatic
mpm.bc_ylo_temp    = adiabatic
mpm.bc_yhi_temp    = adiabatic
mpm.bc_zlo_temp    = adiabatic
mpm.bc_zhi_temp    = adiabatic

mpm.prefix_particlefilename = {cfg["output_tag"]}
mpm.write_output_time       = {phys["write_output_time"]}
""")

print(f"Input file written to: {inp_file}")
