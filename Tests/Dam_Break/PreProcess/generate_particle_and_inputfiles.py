#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dambreak particle and input generator for ExaGOOP.

Based on your preliminary dambreak script:

  blo    = [0.0, 0.0, 0.0]
  bhi    = [0.4, 0.4, 0.004]
  ncells = [100, 100, 1]

Water region:
  x ∈ [0.0, 0.1]
  y ∈ [0.0, 0.2]
  z ∈ [0.0, 0.02]   (z extent only used for 3D)

Material properties:
  dens          = 997.5
  K_BM          = 2e4
  Gama_Pressure = 7.0
  Dyn_visc      = 0.001

Z‑direction rules:

  dim == 2:
    - background blo[2], bhi[2] don't matter for the solver
    - all particle z = 0.0

  dim == 3:
    - order_scheme == 1:
        3 grids in z
        blo[2] = -1.5 * dz
        bhi[2] = +1.5 * dz
        mpm.is_it_periodic = 0 0 1
    - order_scheme == 2 or 3:
        5 grids in z
        blo[2] = -2.5 * dz
        bhi[2] = +2.5 * dz
        mpm.is_it_periodic = 0 0 1
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
import tempfile
import shutil
import os
import hashlib
from typing import List, Tuple


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

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

    max_key_len = max(len(k) for k, _ in entries)
    align_col = max_key_len + 3

    for key, value in entries:
        padding = " " * (align_col - len(key))
        f.write(f"{key}{padding}= {value}\n")


# -------------------------------------------------------------------
# Input file writer (Dambreak)
# -------------------------------------------------------------------

def write_inputs_file(dim: int,
                      nx: int,
                      ny: int,
                      nz: int,                      
                      periodic_z: int,
                      order_scheme: int,
                      alpha_pic_flip: float,
                      stress_update_scheme: int,
                      CFL: float,
                      output_tag: str,
                      dx: float,
                      out_filename: str = "Inputs_DamBreak.inp") -> None:
    """
    Write Inputs_DamBreak.inp using the dambreak geometry
    and your preferred ExaGOOP input format.
    """

    # Domain extents implied by grid and dx
    Lx = nx * dx
    Ly = ny * dx
    Lz = nz * dx if dim == 3 else dx  # for dim=2, z extent is irrelevant
    
    xmin, ymin, zmin = 0.0, 0.0, -Lz*0.5
    xmax, ymax, zmax = Lx, Ly, Lz*0.5
    

    with open(out_filename, "w", encoding="utf-8") as f:

        # Geometry
        write_block(f, [
            ("mpm.prob_lo", f"{xmin} {ymin} {zmin}    # Lower corner"),
            ("mpm.prob_hi", f"{xmax} {ymax} {zmax}    # Upper corner"),
            ("mpm.ncells", f"{nx} {ny} {nz}"),
            ("mpm.max_grid_size", f"{max(nx, ny, nz)}"),
            ("mpm.is_it_periodic", f"0 0 {periodic_z}")
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
            ("mpm.final_time", "2.5"),
            ("mpm.max_steps", "5000000"),
            ("mpm.screen_output_time", "0.001"),
            ("mpm.write_output_time", "0.01"),
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
            ("mpm.gravity", "0.0 -9.81 0.0"),
            ("mpm.applied_strainrate_time", "0.0"),
            ("mpm.applied_strainrate", "0.0"),
            ("mpm.calculate_strain_based_on_delta", "0"),
            ("mpm.external_loads", "0"),
            ("mpm.force_slab_lo", "0.0 0.0 0.0"),
            ("mpm.force_slab_hi", "1.0 1.0 1.0"),
            ("mpm.extforce", "0.0 0.0 0.0")
        ], comment="Physics Parameters")

        # Boundary Conditions (you will probably tune these)
        write_block(f, [
            ("mpm.bc_lower", "2 2 0"),
            ("mpm.bc_upper", "2 2 0"),
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
            ("mpm.do_calculate_minmaxpos", "1"),
            ("mpm.write_diag_output_time", "0.01")
        ], comment="Diagnostics Parameters")

    print(f"WROTE: {out_filename}")


# -------------------------------------------------------------------
# Particle generator (Dambreak)
# -------------------------------------------------------------------

def generate_particles_and_return(dim: int,
                                  ncells_x: int,                                  
                                  np_per_cell_x: int,
                                  order_scheme: int,
                                  alpha_pic_flip: float,
                                  stress_update_scheme: int,
                                  CFL: float,
                                  output_tag: str,
                                  out_particles: str = "mpm_particles.dat"
                                  ) -> Tuple[int, float, int, int]:
    """
    Generate dambreak particles based on your preliminary script,
    with updated z‑rules for 2D/3D and order_scheme.
    Returns (npart, dx, nz, periodic_z).
    """

    # Base background grid (from your original script)
    blo = np.array([0.0, 0.0, 0.0], dtype=float)
    bhi = np.array([0.4, 0.4, 0.004], dtype=float)
    ncells = np.array([100, 100, 1], dtype=int)

    # Base cell sizes
    dx = (bhi - blo) / ncells

    # z‑direction rules
    if dim == 2:
        nz = 1
        periodic_z = 0
        # background blo[2], bhi[2] don't matter for solver; keep as is
    elif dim == 3:
        dz = dx[2]

        if order_scheme == 1:
            nz = 3
            blo[2] = -1.5 * dz
            bhi[2] = +1.5 * dz
        elif order_scheme in (2, 3):
            nz = 5
            blo[2] = -2.5 * dz
            bhi[2] = +2.5 * dz
        else:
            die("For dim=3, order_scheme must be 1, 2, or 3 for this dambreak setup")

        ncells[2] = nz
        dx = (bhi - blo) / ncells
        periodic_z = 1
    else:
        die("dimension must be 2 or 3 for dambreak")

    # Recompute dx and check uniformity
    if not np.allclose(dx[0], dx):
        warn(f"mesh sizes differ: dx = {dx}")

    dx1 = dx[0]

    # Dambreak water region (from your original script)
    xmin, xmax = 0.0, 0.1
    ymin, ymax = 0.0, 0.2
    zmin, zmax = blo[2], bhi[2]  # only relevant for 3D

    nparticle_per_dir = np_per_cell_x  # allow using CLI value here

    # Material properties (from your script)
    dens = 997.5
    phase = 0
    K_BM = 2e4
    Gama_Pressure = 7.0
    Dyn_visc = 0.001

    # Volume and radius
    if dim == 3:
        vol_cell = dx[0] * dx[1] * dx[2]
        vol_particle = vol_cell / (nparticle_per_dir ** 3)
        rad = (3.0 / 4.0 * vol_particle / np.pi) ** (1.0 / 3.0)
    else:
        vol_cell = dx[0] * dx[1]
        vol_particle = vol_cell / (nparticle_per_dir ** 2)
        rad = (4.0 * vol_particle / np.pi) ** 0.5

    lines: List[str] = []

    for k in range(ncells[2]):
        c_cz = blo[2] + k * dx[2]
        for j in range(ncells[1]):
            c_cy = blo[1] + j * dx[1]
            for i in range(ncells[0]):
                c_cx = blo[0] + i * dx[0]

                # Cull outside water region
                if not (xmin <= c_cx < xmax and ymin <= c_cy < ymax):
                    continue
                if dim == 3 and not (zmin <= c_cz < zmax):
                    continue

                for ii in range(nparticle_per_dir):
                    for jj in range(nparticle_per_dir):
                        # Base particle position in x and y
                        cell_cx = c_cx + (2 * ii + 1) * dx[0] / (2.0 * nparticle_per_dir)
                        cell_cy = c_cy + (2 * jj + 1) * dx[1] / (2.0 * nparticle_per_dir)

                        if dim == 3:
                            for kk in range(nparticle_per_dir):
                                cell_cz = c_cz + (2 * kk + 1) * dx[2] / (2.0 * nparticle_per_dir)
                                velx = vely = velz = 0.0
                                line = (
                                    f"{phase:d}\t"
                                    f"{cell_cx:.6e}\t{cell_cy:.6e}\t{cell_cz:.6e}\t"
                                    f"{rad:.6e}\t{dens:.6e}\t"
                                    f"{velx:.6e}\t{vely:.6e}\t{velz:.6e}\t"
                                    f"{1:d}\t{K_BM:.6e}\t{Gama_Pressure:.6e}\t{Dyn_visc:.6e}\n"
                                )
                                lines.append(line)
                        else:
                            # 2D: z is always 0.0, not written to file (2D format)
                            velx = vely = 0.0
                            line = (
                                f"{phase:d}\t"
                                f"{cell_cx:.6e}\t{cell_cy:.6e}\t"
                                f"{rad:.6e}\t{dens:.6e}\t"
                                f"{velx:.6e}\t{vely:.6e}\t"
                                f"{1:d}\t{K_BM:.6e}\t{Gama_Pressure:.6e}\t{Dyn_visc:.6e}\n"
                            )
                            lines.append(line)

    if not lines:
        warn("Zero particles generated; check dambreak region vs domain.")

    write_atomic_with_count(lines, out_particles)
    print(f"WROTE: {out_particles} with {len(lines)} particles")

    return len(lines), dx1, ncells[2], periodic_z


# -------------------------------------------------------------------
# Tag + CLI
# -------------------------------------------------------------------

def make_auto_tag(args) -> str:
    desc = (
        f"DamBreak_"
        f"dim{args.dimension}_"
        f"nx{args.no_of_cell_in_x}_"
        f"ppc{args.np_per_cell_x}_"
        f"buff{args.buffery}_"
        f"ord{args.order_scheme}_"
        f"alpha{args.alpha_pic_flip}_"
        f"sus{args.stress_update_scheme}_"
        f"CFL{args.CFL}"
    )

    short_hash = hashlib.md5(desc.encode()).hexdigest()[:6]
    return f"{desc}_{short_hash}"


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate dambreak mpm_particles.dat and Inputs_DamBreak.inp")
    p.add_argument("--dimension", type=int, required=True)
    p.add_argument("--no_of_cell_in_x", type=int, required=True)    
    p.add_argument("--np_per_cell_x", type=int, required=True)
    p.add_argument("--order_scheme", type=int, required=True)
    p.add_argument("--alpha_pic_flip", type=float, required=True)
    p.add_argument("--stress_update_scheme", type=int, required=True)
    p.add_argument("--CFL", type=float, required=True)
    p.add_argument("--output_tag", type=str, default="", help="Leave empty to auto-generate")

    p.add_argument("--debug", action="store_true", help="Print parsed args and exit")

    args = p.parse_args()

    if args.output_tag.strip() == "":
        args.output_tag = make_auto_tag(args)
        print(f"[INFO] Auto-generated output_tag = {args.output_tag}")

    return args


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

def main() -> None:
    args = parse_cli()
    if args.debug:
        print("DEBUG ARGS:", args)

    npart, dx1, nz, periodic_z = generate_particles_and_return(
        dim=args.dimension,
        ncells_x=args.no_of_cell_in_x,        
        np_per_cell_x=args.np_per_cell_x,
        order_scheme=args.order_scheme,
        alpha_pic_flip=args.alpha_pic_flip,
        stress_update_scheme=args.stress_update_scheme,
        CFL=args.CFL,
        output_tag=args.output_tag,
        out_particles="mpm_particles.dat"
    )
    
    write_inputs_file(
        dim=args.dimension,
        nx=100,                 # from base dambreak grid
        ny=100,                 # from base dambreak grid
        nz=nz,        
        periodic_z=periodic_z,
        order_scheme=args.order_scheme,
        alpha_pic_flip=args.alpha_pic_flip,
        stress_update_scheme=args.stress_update_scheme,
        CFL=args.CFL,
        output_tag=args.output_tag,
        dx=dx1,
        out_filename="Inputs_DamBreak.inp"
    )

    print("All done.")


if __name__ == "__main__":
    main()
