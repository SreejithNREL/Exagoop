#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_particles.py
Generate MPM particle file (mpm_particles.dat) and inputs_axialbar for ExaGOOP.

Positional arguments (original ordering kept):
  1  dimension               (int>0)
  2  no_of_cell_in_x         (int >0)
  3  buffery                 (int >=0)
  4  periodic                (0 or 1)
  5  np_per_cell_x           (int >0)
  6  order_scheme            (int)
  7  alpha_pic_flip          (float)
  8  stress_update_scheme    (int)
  9  CFL                     (float)  
  10  output_tag              (string) used for Solution/<tag> prefixes

Example:
  python3 generate_particles.py 25 3 1 1 3 1 1 0.1 mytag
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


def generate_particles_and_return(dim: int,
                                  ncells_x: int,
                                  buffery: int,
                                  periodic: int,
                                  np_per_cell_x: int,
                                  order_scheme: int,
                                  alpha_pic_flip: float,
                                  stress_update_scheme: int,
                                  CFL: float,
                                  output_tag: str,
                                  out_particles: str = "mpm_particles.dat"
                                  ) -> Tuple[int, float]:
    """
    Generate particle lines and write mpm_particles.dat atomically.
    Returns (npart, dx1).
    """
    if ncells_x <= 0:
        die("no_of_cell_in_x must be > 0")
    if buffery < 0:
        die("buffery must be >= 0")
    if np_per_cell_x <= 0:
        die("np_per_cell_x must be > 0")
    if periodic not in (0, 1):
        warn("periodic should be 0 or 1; treating other values as 0")
        periodic = 0

    # domain geometry (kept from original script)
    L = 25.0
    bufferx = 4
    bufferz = buffery
    dx1 = L / float(ncells_x)

    blo = np.array([0.0, -(buffery + 0.5) * dx1, -(bufferz + 0.5) * dx1], dtype=float)
    bhi = np.array([L + bufferx * dx1, (buffery + 0.5) * dx1, (bufferz + 0.5) * dx1], dtype=float)
    ncells = np.array([ncells_x + bufferx, 2 * buffery + 1, 2 * bufferz + 1], dtype=int)
    dx = (bhi - blo) / ncells

    if not np.allclose(dx, dx[0]):
        warn(f"Non-uniform mesh spacing detected: dx = {dx}")


    # sampling window for placing particles (same as original)
    xmin, xmax = 0.0, L
    ymin, ymax = -dx1 / 2.0, dx1 / 2.0
    zmin, zmax = -dx1 / 2.0, dx1 / 2.0

    # particle geometry and physical properties (original defaults)
    vol_cell = float(dx[0] * dx[1] * dx[2])
    vol_particle = vol_cell / float(np_per_cell_x * 1 * 1)
    if vol_particle <= 0.0 or not np.isfinite(vol_particle):
        die("Computed particle volume is non-positive or invalid.")
    rad = (3.0 / 4.0 * vol_particle / np.pi) ** (1.0 / 3.0)

    dens = 1.0
    phase = 0
    E = 100.0
    nu = 0.0
    v0 = 0.1
    n = 1
    beta_n = (2*n - 1) * np.pi / (2*L)


    particle_lines: List[str] = []

    # generate particle lines; keep formatting stable so 'phase' prints as an integer token
    if(dim==3):
        for i in range(int(ncells[0])):
            c_cx = blo[0] + i * dx[0]            
            if (xmin <= c_cx < xmax):
                for ii in range(int(np_per_cell_x)):
                    cell_cx = c_cx + (2 * ii + 1) * dx[0] / (2.0 * np_per_cell_x)
                    velx = v0 * np.sin(beta_n * cell_cx)
                    vely = 0.0
                    velz = 0.0
                    # explicit formatting keeps tokens separated and phase as integer
                    line = "{phase:d} {cx:.6e} {cy:.6e} {cz:.6e} {rad:.6e} {dens:.6e} {vx:.6e} {vy:.6e} {vz:.6e} {flag:d} {E:.6e} {nu:.6e}\n".format(
                        phase=int(phase),
                        cx=cell_cx,
                        cy=0.0,
                        cz=0.0,
                        rad=rad,
                        dens=dens,
                        vx=velx,
                        vy=vely,
                        vz=velz,
                        flag=0,
                        E=E,
                        nu=nu
                        )
                    particle_lines.append(line)
    elif(dim==2):
        for i in range(int(ncells[0])):
            c_cx = blo[0] + i * dx[0]            
            if (xmin <= c_cx < xmax):
                for ii in range(int(np_per_cell_x)):
                    cell_cx = c_cx + (2 * ii + 1) * dx[0] / (2.0 * np_per_cell_x)
                    velx = v0 * np.sin(beta_n * cell_cx)
                    vely = 0.0
                    velz = 0.0
                    # explicit formatting keeps tokens separated and phase as integer
                    line = "{phase:d} {cx:.6e} {cy:.6e} {rad:.6e} {dens:.6e} {vx:.6e} {vy:.6e} {flag:d} {E:.6e} {nu:.6e}\n".format(
                        phase=int(phase),
                        cx=cell_cx,
                        cy=0.0,                        
                        rad=rad,
                        dens=dens,
                        vx=velx,
                        vy=vely,                        
                        flag=0,
                        E=E,
                        nu=nu
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
                      buffery: int,
                      periodic: int,
                      np_per_cell_x: int,
                      order_scheme: int,
                      alpha_pic_flip: float,
                      stress_update_scheme: int,
                      CFL: float,
                      output_tag: str,
                      dx1: float,
                      out_filename: str = "Inputs_1DAxialBarVibration.inp") -> None:

    bufferx = 4
    xmin = 0.0
    xmax = 25.0 + bufferx * dx1
    ymin = -dx1 * (buffery + 0.5)
    ymax = dx1 * (buffery + 0.5)
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
            ("mpm.ncells", f"{ncells_x + bufferx} {2*buffery+1} {2*buffery+1}"),
            ("mpm.max_grid_size", f"{ncells_x + bufferx + 1}"),
            ("mpm.is_it_periodic", f"0 {periodic} {periodic}")
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
            ("mpm.final_time", "50.0"),
            ("mpm.max_steps", "5000000"),
            ("mpm.screen_output_time", "0.001"),
            ("mpm.write_output_time", "0.5"),
            ("mpm.num_redist", "1")
        ], comment="Simulation Run Parameters")

        # Timestepping
        write_block(f, [
            ("mpm.fixed_timestep", "0"),
            ("mpm.timestep", "1.0e-5"),
            ("mpm.CFL", f"{CFL}"),
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
            ("mpm.alpha_pic_flip", f"{alpha_pic_flip}"),
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
            ("mpm.bc_upper", "2 0 0"),
            ("mpm.bc_lower_temp", "2 0 0"),
            ("mpm.bc_upper_temp", "2 0 0"),
            ("mpm.bc_lower_tempval", "1 0 0"),
            ("mpm.bc_upper_tempval", "1 0 0"),
            ("mpm.levelset_bc", "2 0 0"),
            ("mpm.levelset_wall_mu", "2 0 0"),
            ("mpm.wall_mu_lo", "2 0 0"),
            ("mpm.wall_mu_hi", "2 0 0"),
            ("mpm.wall_vel_lo", "0 0 0 0 0 0 0 0 0"),
            ("mpm.wall_vel_hi", "0 0 0 0 0 0 0 0 0")
        ], comment="Boundary Conditions")

        # Diagnostics
        write_block(f, [
            ("mpm.print_diagnostics", "1"),
            ("mpm.do_calculate_tke_tse", "1"),
            ("mpm.do_calculate_mwa_velcomp", "1"),
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
        f"AVB_"
        f"dim{args.dimension}_"
        f"nx{args.no_of_cell_in_x}_"
        f"ppc{args.np_per_cell_x}_"
        f"buff{args.buffery}_"
        f"ord{args.order_scheme}_"
        f"alpha{args.alpha_pic_flip}_"
        f"sus{args.stress_update_scheme}_"
        f"CFL{args.CFL}"
    )

    # Create a short hash for uniqueness
    key = desc
    short_hash = hashlib.md5(key.encode()).hexdigest()[:6]

    return f"{desc}_{short_hash}"


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MPM particle file and inputs_axialbar.in")

    p.add_argument("--dimension", type=int, required=True)
    p.add_argument("--no_of_cell_in_x", type=int, required=True)
    p.add_argument("--buffery", type=int, required=True)
    p.add_argument("--periodic", type=int, choices=[0, 1], required=True)
    p.add_argument("--np_per_cell_x", type=int, required=True)
    p.add_argument("--order_scheme", type=int, required=True)
    p.add_argument("--alpha_pic_flip", type=float, required=True)
    p.add_argument("--stress_update_scheme", type=int, required=True)
    p.add_argument("--CFL", type=float, required=True)
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
        dim=args.dimension,
        ncells_x=args.no_of_cell_in_x,
        buffery=args.buffery,
        periodic=args.periodic,
        np_per_cell_x=args.np_per_cell_x,
        order_scheme=args.order_scheme,
        alpha_pic_flip=args.alpha_pic_flip,
        stress_update_scheme=args.stress_update_scheme,
        CFL=args.CFL,
        output_tag=args.output_tag,
        out_particles="mpm_particles.dat"
    )

    write_inputs_file(
        ncells_x=args.no_of_cell_in_x,
        buffery=args.buffery,
        periodic=args.periodic,
        np_per_cell_x=args.np_per_cell_x,
        order_scheme=args.order_scheme,
        alpha_pic_flip=args.alpha_pic_flip,
        stress_update_scheme=args.stress_update_scheme,
        CFL=args.CFL,        
        output_tag=args.output_tag,
        dx1=dx1,
        out_filename="Inputs_1DAxialBarVibration.inp"
    )

    print("All done.")


if __name__ == "__main__":
    main()

