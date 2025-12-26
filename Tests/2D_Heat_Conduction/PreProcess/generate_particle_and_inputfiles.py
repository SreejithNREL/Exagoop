#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_particles.py
Generate MPM particle file (mpm_particles.dat) and inputs_axialbar for ExaGOOP.

Positional arguments (original ordering kept):
  1  dimension               (0<int<=3)
  2  no_of_cell_in_x         (int >0)  
  3  np_per_cell_x           (int >0)
  4  order_scheme            (int)
  5  alpha_pic_flip          (float)
  6  stress_update_scheme    (int)
  7  time step               (float)  
  8  output_tag              (string) used for Solution/<tag> prefixes

Example:
  python3 generate_particles.py 2 25 3 1 1 3 1 1 0.1 mytag
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import tempfile
import shutil
import os
from typing import List, Tuple


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def warn(msg: str) -> None:
    print(f"Warning: {msg}", file=sys.stderr)


def write_atomic_with_count(lines: List[str], out_filename: str) -> None:
    """
    Write lines to out_filename with the first line equal to the count.
    Uses a temporary file and atomic move.
    """
    out_dir = os.path.dirname(os.path.abspath(out_filename)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=out_dir, delete=False) as tmpf:
            tmp = tmpf.name
            tmpf.write(f"{len(lines)}\n")
            for ln in lines:
                tmpf.write(ln)
        shutil.move(tmp, out_filename)
    except Exception as e:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        die(f"Failed to write '{out_filename}': {e}")


def generate_particles_and_return(dimensions: int,
                                  ncells_x: int,                                  
                                  np_per_cell_x: int,
                                  order_scheme: int,                                  
                                  stress_update_scheme: int,                                  
                                  output_tag: str,
                                  out_particles: str = "mpm_particles.dat"
                                  ) -> Tuple[int, float]:
    """
    Generate particle lines and write mpm_particles.dat.
    Returns (npart, dx1).
    """
    
    if dimensions not in [2,3]:
        die("dimension should be 2 or 3")
    if ncells_x <= 0:
        die("no_of_cell_in_x must be > 0")    
    if np_per_cell_x <= 0:
        die("np_per_cell_x must be > 0")
    

    # domain geometry
    Lx = 1.0
    Ly = 1.0
    bufferx = 0         #no buffer cells in x for this test case    
    dx1 = Lx / float(ncells_x)

    blo = np.array([0.0, 0.0, -(0.5) * dx1], dtype=float)
    bhi = np.array([Lx,   Ly,  (0.5) * dx1], dtype=float)
    ncells = np.array([ncells_x,ncells_x, 1], dtype=int)
    dx = (bhi - blo) / ncells

    if not np.allclose(dx[0], dx):
        warn(f"mesh sizes differ: dx = {dx}")

    # sampling window for placing particles (same as original)
    xmin, xmax = 0.0, Lx    

    # particle geometry and physical properties (original defaults)
    vol_cell = float(dx[0] * dx[1] * dx[2])
    vol_particle = vol_cell / float(np_per_cell_x * 1 * 1)
    if vol_particle <= 0.0 or not np.isfinite(vol_particle):
        die("Computed particle volume is non-positive or invalid.")
    rad = (3.0 / 4.0 * vol_particle / np.pi) ** (1.0 / 3.0)

    dens = 1.0
    phase = 0
    E = 0.0
    nu = 0.0
    v0 = 0.1
    n = 1
    T = 0.0
    spheat= 1.0
    thermcond = 1.0
    heatsrc=0.0    

    particle_lines: List[str] = []   

    # generate particle lines; keep formatting stable so 'phase' prints as an integer token
    for i in range(int(ncells[0])):
        for j in range(int(ncells[1])):
            c_cx = blo[0] + i * dx[0]
            c_cy = blo[1] + j * dx[1]
            c_cz = 0.0
            if (xmin <= c_cx < xmax):
                for ii in range(int(np_per_cell_x)):
                    for jj in range(int(np_per_cell_x)):
                        cell_cx = c_cx + (2 * ii + 1) * dx[0] / (2.0 * np_per_cell_x)         
                        cell_cy = c_cy + (2 * jj + 1) * dx[1] / (2.0 * np_per_cell_x)                
                        velx = 0.0
                        vely = 0.0
                        velz = 0.0              
                
                        if(dimensions==2):
                            line = "{phase:d} {cx:.6e} {cy:.6e} {rad:.6e} {dens:.6e} {vx:.6e} {vy:.6e} {flag:d} {E:.6e} {nu:.6e} {T:.6e} {spheat:.6e} {thermcond:.6e} {heatsrc:.6e}\n".format(
                                phase=int(phase),
                                cx=cell_cx,
                                cy=cell_cy,                                
                                rad=rad,
                                dens=dens,
                                vx=velx,
                                vy=vely,                                
                                flag=0,
                                E=E,
                                nu=nu,
                                T = T,
                                spheat = spheat,
                                thermcond = thermcond,
                                heatsrc = heatsrc
                                )
                        else:
                            line = "{phase:d} {cx:.6e} {cy:.6e} {cz:.6e} {rad:.6e} {dens:.6e} {vx:.6e} {vy:.6e} {vz:.6e} {flag:d} {E:.6e} {nu:.6e} {T:.6e} {spheat:.6e} {thermcond:.6e} {heatsrc:.6e}\n".format(
                                phase=int(phase),
                                cx=cell_cx,
                                cy=c_cy,
                                cz=c_cz,
                                rad=rad,
                                dens=dens,
                                vx=velx,
                                vy=vely,
                                vz=velz,
                                flag=0,
                                E=E,
                                nu=nu,
                                T = T,
                                spheat = spheat,
                                thermcond = thermcond,
                                heatsrc = heatsrc
                                )                        
                        particle_lines.append(line)

    npart = len(particle_lines)
    if npart == 0:
        warn("Zero particles generated; check domain/parameters.")

    # write atomically (header + particle lines)
    write_atomic_with_count(particle_lines, out_particles)
    print(f"WROTE: {out_particles} with {npart} particles")
    return npart, dx1

def write_block(f, entries, comment=None):
    """
    Write a block of key=value lines with aligned '=' signs.
    Adds decorative separators before and after the block title.
    """
    if comment:
        dash_len = max(len(comment) + 10, 30)
        dashes = "-" * dash_len
        f.write(f"\n#{dashes}\n")
        f.write(f"# {comment}\n")
        f.write(f"#{dashes}\n")

    # Determine alignment column
    max_key_len = max(len(k) for k, _ in entries)
    align_col = max_key_len + 3  # space before and after '='

    for key, value in entries:
        padding = " " * (align_col - len(key))
        f.write(f"{key}{padding}= {value}\n")


def write_inputs_file(ncells_x: int,                      
                      np_per_cell_x: int,
                      order_scheme: int,                      
                      stress_update_scheme: int,                      
                      output_tag: str,
                      dx1: float,
                      out_filename: str = "inputs_axialbar.inp") -> None:

    bufferx = 0
    buffery = 0
    xmin = 0.0
    xmax = 1.0 
    ymin = 0.0
    ymax = 1.0
    zmin = -dx1 * (buffery + 0.5)
    zmax = dx1 * (buffery + 0.5)

    #sol_dir = os.path.join(".", "Solution", output_tag)
    #os.makedirs(sol_dir, exist_ok=True)

    with open(out_filename, "w", encoding="utf-8") as f:

        # ---------------------------------------------------------
        # Geometry
        # ---------------------------------------------------------
        write_block(f, [
            ("mpm.prob_lo", f"{xmin} {ymin} {zmin}    # Lower corner"),
            ("mpm.prob_hi", f"{xmax} {ymax} {zmax}    # Upper corner"),
            ("mpm.ncells", f"{ncells_x} {ncells_x} {1}"),
            ("mpm.max_grid_size", f"{ncells_x + 1}"),
            ("mpm.is_it_periodic", f"0 0 1")
        ], comment="Geometry Parameters")

        # AMR
        write_block(f, [
            ("#restart_checkfile", "\"\"")
        ], comment="AMR Parameters")

        # Input Material Points
        write_block(f, [
            ("mpm.use_autogen", "0"),
            ("mpm.mincoords_autogen", "0.0 0.0 0.0"),
            ("mpm.maxcoords_autogen", "1.0 1.0 1.0"),
            ("mpm.vel_autogen", "0.0 0.0 0.0"),
            ("mpm.constmodel_autogen", "0"),
            ("mpm.dens_autogen", "1.0"),
            ("mpm.E_autogen", "1e6"),
            ("mpm.nu_autogen", "0.3"),
            ("mpm.bulkmod_autogen", "2e6"),
            ("mpm.Gama_pres_autogen", "7"),
            ("mpm.visc_autogen", "0.001"),
            ("mpm.multi_part_per_cell_autogen", "1"),
            ("mpm.particle_file", "\"mpm_particles.dat\"")
        ], comment="Input Material Points")

        # Output Parameters
        write_block(f, [
            ("mpm.prefix_particlefilename", f"\"{output_tag}/plt\""),
            ("mpm.prefix_gridfilename", f"\"{output_tag}/nplt\""),
            ("mpm.prefix_densityfilename", f"\"{output_tag}/dens\""),
            ("mpm.prefix_checkpointfilename", f"\"{output_tag}/chk\""),
            ("mpm.prefix_asciifilename", f"\"{output_tag}/matpnt\""),
            ("mpm.diagnostic_output_folder", f"\"./Diagnostics/{output_tag}\""),
            ("mpm.num_of_digits_in_filenames", "6"),
            ("mpm.write_ascii", "1")
        ], comment="Output Parameters")

        # Simulation Run Parameters
        write_block(f, [
            ("mpm.final_time", "0.05"),
            ("mpm.max_steps", "5000000"),
            ("mpm.screen_output_time", "0.0001"),
            ("mpm.write_output_time", "0.001"),
            ("mpm.num_redist", "1")
        ], comment="Simulation Run Parameters")

        # Timestepping
        write_block(f, [
            ("mpm.fixed_timestep", "1"),
            ("mpm.timestep", "1.0e-5"),
            ("mpm.CFL", "0.1"),
            ("mpm.dt_min_limit", "1e-12"),
            ("mpm.dt_max_limit", "1e+00")
        ], comment="Timestepping Parameters")

        # Levelset
        write_block(f, [
            ("mpm.levset_output", "0"),
            ("mpm.levset_smoothfactor", "1.0"),
            ("mpm.levset_gridratio", "1")
        ], comment="Levelset Parameters")

        # Numerical Schemes
        write_block(f, [
            ("mpm.order_scheme", f"{order_scheme}"),
            ("mpm.alpha_pic_flip", "1.0"),
            ("mpm.stress_update_scheme", f"{stress_update_scheme}"),
            ("mpm.mass_tolerance", "1e-18")
        ], comment="Numerical Schemes")

        # Physics
        write_block(f, [
            ("mpm.gravity", "0.0 0.0 0.0"),
            ("mpm.applied_strainrate_time", "0.0"),
            ("mpm.applied_strainrate", "0.0"),
            ("mpm.calculate_strain_based_on_delta", "0"),
            ("mpm.external_loads", "0"),
            ("mpm.force_slab_lo", "0.0 0.0 0.0"),
            ("mpm.force_slab_hi", "1.0 1.0 1.0"),
            ("mpm.extforce", "0.0 0.0 0.0")
        ], comment="Physics Parameters")

        # Boundary Conditions
        write_block(f, [
            ("mpm.bc_lower", "1 0 0"),
            ("mpm.bc_upper", "1 0 0"),
            ("mpm.bc_lower_temp", "1 1 0"),
            ("mpm.bc_upper_temp", "1 1 0"),
            ("mpm.bc_lower_tempval", "1.0 1.0 0"),
            ("mpm.bc_upper_tempval", "1.0 1.0 0"),
            ("mpm.levelset_bc", "2 0 0"),
            ("mpm.levelset_wall_mu", "2 0 0"),
            ("mpm.wall_mu_lo", "2 0 0"),
            ("mpm.wall_mu_hi", "2 0 0"),
            ("mpm.wall_vel_lo", "0 0 0 0 0 0 0 0 0"),
            ("mpm.wall_vel_hi", "0 0 0 0 0 0 0 0 0")
        ], comment="Boundary Conditions")

        # Diagnostics
        write_block(f, [
            ("mpm.print_diagnostics", "0"),
            ("mpm.do_calculate_tke_tse", "0"),
            ("mpm.do_calculate_mwa_velcomp", "0"),
            ("mpm.do_calculate_mwa_velmag", "0"),
            ("mpm.do_calculate_minmaxpos", "0"),
            ("mpm.write_diag_output_time", "0.01")
        ], comment="Diagnostics Parameters")

    print(f"WROTE: {out_filename}")

import hashlib

def make_auto_tag(args) -> str:
    """
    Create a descriptive, deterministic, unique tag based on user inputs.
    """
    # Human-readable descriptive part
    desc = (
        f"2D_Heat_Conduction_"        
        f"nx{args.no_of_cell_in_x_y}_"
        f"ppc{args.np_per_cell_x_y}_"        
        f"ord{args.order_scheme}_"        
        f"sus{args.stress_update_scheme}"        
    )

    # Create a short hash for uniqueness
    key = desc
    short_hash = hashlib.md5(key.encode()).hexdigest()[:6]

    return f"{desc}_{short_hash}"


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MPM particle file and inputs_axialbar.in")    
    p.add_argument("--no_of_cell_in_x", type=int, required=True)   
    p.add_argument("--np_per_cell_x", type=int, required=True)
    p.add_argument("--order_scheme", type=int, required=True)    
    p.add_argument("--stress_update_scheme", type=int, required=True)    
    p.add_argument("--output_tag", type=str, default="", help="Leave empty to auto-generate")
    p.add_argument("--debug", action="store_true", help="Print parsed args and exit")

    args = p.parse_args()

    # Auto-generate tag if empty
    if args.output_tag.strip() == "":
        args.output_tag = make_auto_tag(args)
        print(f"[INFO] Auto-generated output_tag = {args.output_tag}")

    return args


# def write_inputs_file(dimensions: int,
#                       ncells_x: int,                      
#                       np_per_cell_x: int,
#                       order_scheme: int,
#                       alpha_pic_flip: float,
#                       stress_update_scheme: int,
#                       timestep: float,                      
#                       output_tag: str,
#                       dx1: float,
#                       out_filename: str = "inputs_axialbar") -> None:
#     """
#     Write inputs_axialbar file using provided parameters. Creates Solution/<output_tag> directory if needed.
#     """
#
#     xmin = 0.0
#     xmax = 1.0 
#     ymin = 0.0
#     ymax = 1.0
#     zmin = -dx1 * (0.5)       #Keep buffery in z-direction as well
#     zmax = dx1 * (0.5)        #Keep buffery in z-direction as well
#
#     sol_dir = os.path.join(".", "Solution", output_tag)
#     try:
#         os.makedirs(sol_dir, exist_ok=True)
#     except Exception as e:
#         warn(f"Could not create Solution directory '{sol_dir}': {e}")
#
#     try:
#         with open(out_filename, "w", encoding="utf-8") as f:
#             f.write("#geometry parameters\n")
#             f.write(f"mpm.prob_lo = {xmin} {ymin} {zmin}\t\t\t#Lower corner of physical domain\n")
#             f.write(f"mpm.prob_hi = {xmax} {ymax} {zmax}\t\t\t#Upper corner of physical domain\n")
#             f.write(f"mpm.ncells  = {ncells_x} {ncells_x } {1}\n")
#             f.write(f"mpm.max_grid_size = {ncells_x + 1}\n")
#             f.write(f"mpm.is_it_periodic = 0  0  1\n")
#
#             f.write("\n\n#AMR Parameters\n")
#             f.write("#restart_checkfile = \"\"\n")
#
#             f.write("\n\n#Input files\n")
#             f.write("mpm.use_autogen=0\n")
#             f.write("mpm.mincoords_autogen=0.0 0.0 0.0\n")
#             f.write("mpm.maxcoords_autogen=1.0 1.0 1.0\n")
#             f.write("mpm.vel_autogen=0.0 0.0 0.0\n")
#             f.write("mpm.constmodel_autogen=0\n")
#             f.write("mpm.dens_autogen=1.0\n")
#             f.write("mpm.E_autogen=1e6\n")
#             f.write("mpm.nu_autogen=0.3\n")
#             f.write("mpm.bulkmod_autogen=2e6\n")
#             f.write("mpm.Gama_pres_autogen=7\n")
#             f.write("mpm.visc_autogen=0.001\n")
#             f.write("mpm.multi_part_per_cell_autogen=1\n")
#             f.write("mpm.particle_file=\"mpm_particles.dat\"\n")
#
#             f.write("\n\n#File output parameters\n")
#             f.write(f"mpm.prefix_particlefilename=\"{output_tag}/plt\"\n")
#             f.write(f"mpm.prefix_gridfilename=\"{output_tag}/nplt\"\n")
#             f.write(f"mpm.prefix_densityfilename=\"{output_tag}/dens\"\n")
#             f.write(f"mpm.prefix_checkpointfilename=\"{output_tag}/chk\"\n")
#             f.write("mpm.write_ascii=1\n")
#             f.write("mpm.num_of_digits_in_filenames=6\n")
#
#             f.write("\n\n#Simulation run parameters\n")
#             f.write("mpm.final_time= 0.05\n")
#             f.write("mpm.max_steps=5000000\n")
#             f.write("mpm.screen_output_time = 0.0001\n")
#             f.write("mpm.write_output_time=0.001\n")
#             f.write("mpm.num_redist = 1\n")
#
#             f.write("\n\n#Timestepping parameters\n")
#             f.write("mpm.fixed_timestep = 1\n")
#             f.write(f"mpm.timestep = {timestep}\n")
#             f.write("mpm.CFL=0.1\n")
#             f.write("mpm.dt_min_limit=1e-12\n")
#             f.write("mpm.dt_max_limit=1e+00\n")
#
#             f.write("\n\n#Numerical schemes\n")
#             f.write(f"mpm.order_scheme={order_scheme}\n")
#             f.write(f"mpm.alpha_pic_flip = {alpha_pic_flip}\n")
#             f.write(f"mpm.stress_update_scheme= {stress_update_scheme}\n")
#             f.write("mpm.mass_tolerance = 1e-18\n")
#
#             f.write("\n\n#Physics parameters\n")
#             f.write("mpm.gravity = 0.0 0.0 0.0\n")
#             f.write("mpm.applied_strainrate_time=0.0\n")
#             f.write("mpm.applied_strainrate=0.0\n")
#             f.write("mpm.external_loads=0\n")
#             f.write("mpm.force_slab_lo= 0.0 0.0 0.0\n")
#             f.write("mpm.force_slab_hi= 1.0 1.0 1.0\n")
#             f.write("mpm.extforce = 0.0 0.0 0.0\n")
#
#             f.write("\n\n#Diagnostics and Test\n")
#             f.write("mpm.print_diagnostics= 0\n")
#
#             f.write("\n\n#Boundary conditions\n")
#             f.write("mpm.bc_lower=1 1 0\n")
#             f.write("mpm.bc_upper=1 1 0\n")            
#             f.write("mpm.bc_lower_temp=1 1 0\n")
#             f.write("mpm.bc_upper_temp=1 1 0\n")
#             f.write("mpm.bc_lower_tempval=1.0 1.0 0\n")
#             f.write("mpm.bc_upper_tempval=1.0 1.0 0\n")
#
#     except Exception as e:
#         die(f"Failed to write inputs file '{out_filename}': {e}")
#
#     print(f"WROTE: {out_filename}")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MPM particle file and inputs_axialbar.in")   
    p.add_argument("--no_of_cell_in_x_y", type=int, required=True)    
    p.add_argument("--np_per_cell_x_y", type=int, required=True)
    p.add_argument("--order_scheme", type=int, required=True)    
    p.add_argument("--stress_update_scheme", type=int, required=True)    
    p.add_argument("--output_tag", type=str, default="", help="Leave empty to auto-generate")
    p.add_argument("--debug", action="store_true", help="Print parsed args and exit")

    args = p.parse_args()

    # Auto-generate tag if empty
    if args.output_tag.strip() == "":
        args.output_tag = make_auto_tag(args)
        print(f"[INFO] Auto-generated output_tag = {args.output_tag}")

    return args


def main() -> None:
    args = parse_cli()
    if args.debug:
        print("DEBUG ARGS:", args)

    npart, dx1 = generate_particles_and_return(
        dimensions=2,
        ncells_x=args.no_of_cell_in_x_y,        
        np_per_cell_x=args.np_per_cell_x_y,
        order_scheme=args.order_scheme,        
        stress_update_scheme=args.stress_update_scheme,        
        output_tag=args.output_tag,
        out_particles="mpm_particles.dat"
    )

    write_inputs_file(        
        ncells_x=args.no_of_cell_in_x_y,        
        np_per_cell_x=args.np_per_cell_x_y,
        order_scheme=args.order_scheme,        
        stress_update_scheme=args.stress_update_scheme,         
        output_tag=args.output_tag,
        dx1=dx1,
        out_filename="Inputs_2DHeatConduction.inp"
    )

    print("All done.")


if __name__ == "__main__":
    main()

