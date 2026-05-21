#!/usr/bin/env python3
import argparse
import json
import hashlib
import platform
import numpy as np
import os

LIB_EXT = "dylib" if platform.system() == "Darwin" else "so"


def die(msg: str):
    raise SystemExit(f"[ERROR] {msg}")


def parse_cli():
    p = argparse.ArgumentParser(
        description="PreProcessor for 3D_Twisted_Column: generates particles + input file"
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to JSON configuration file")
    return p.parse_args()


class Block:
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax

    def contains(self, p):
        x, y, z = p
        return (self.xmin <= x <= self.xmax and
                self.ymin <= y <= self.ymax and
                self.zmin <= z <= self.zmax)


def ppc_offsets(N):
    i = np.arange(1, N + 1)
    return (2 * i - 1) / (2 * N)


def generate_particle_chunks(grid, ppc, shape, cm_cfg, density,
                              vx0, vy0, vz0, chunk_size=200_000):
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    xmin, xmax = grid["xmin"], grid["xmax"]
    ymin, ymax = grid["ymin"], grid["ymax"]
    zmin, zmax = grid["zmin"], grid["zmax"]
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz

    ox = ppc_offsets(ppc[0])
    oy = ppc_offsets(ppc[1])
    oz = ppc_offsets(ppc[2])

    E   = cm_cfg["E"]
    nu  = cm_cfg["nu"]
    vol = dx * dy * dz / (ppc[0] * ppc[1] * ppc[2])
    radius = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)

    buf = {k: [] for k in ["phase", "x", "y", "z", "radius",
                             "density", "vx", "vy", "vz", "cm_id",
                             "E", "nu"]}

    def flush():
        out = {k: np.array(v) for k, v in buf.items()}
        for k in buf:
            buf[k].clear()
        return out

    count = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                cx = xmin + ix * dx
                cy = ymin + iy * dy
                cz = zmin + iz * dz
                for fx in ox:
                    for fy in oy:
                        for fz in oz:
                            px = cx + fx * dx
                            py = cy + fy * dy
                            pz = cz + fz * dz
                            if not shape.contains((px, py, pz)):
                                continue
                            buf["phase"].append(0)
                            buf["x"].append(px)
                            buf["y"].append(py)
                            buf["z"].append(pz)
                            buf["radius"].append(radius)
                            buf["density"].append(density)
                            buf["vx"].append(vx0)
                            buf["vy"].append(vy0)
                            buf["vz"].append(vz0)
                            buf["cm_id"].append(0)
                            buf["E"].append(E)
                            buf["nu"].append(nu)
                            count += 1
                            if count >= chunk_size:
                                yield flush()
                                count = 0

    if count > 0:
        yield flush()


def write_particles_ascii(filename, chunk_iter):
    colnames = ["phase", "x", "y", "z", "radius",
                "density", "vx", "vy", "vz", "cm_id", "E", "nu"]
    f = open(filename, "w")
    f.write("dim: 3\n")
    f.write("number_of_material_points: 0\n")
    f.write("# " + " ".join(colnames) + "\n")

    total = 0
    for chunk in chunk_iter:
        n = len(chunk["x"])
        for i in range(n):
            row = []
            for k in colnames:
                v = chunk[k][i]
                if isinstance(v, (float, np.floating)):
                    row.append(f"{v:.6e}")
                else:
                    row.append(str(int(v)))
            f.write("\t".join(row) + "\n")
        total += n

    f.close()

    with open(filename, "r+") as f2:
        lines = f2.readlines()
        lines[1] = f"number_of_material_points: {total}\n"
        f2.seek(0)
        f2.writelines(lines)

    return total


def write_block(f, entries, comment=None):
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


def write_inputs_file(cfg, output_tag, particle_filename, out_filename):
    grid    = cfg["grid"]
    physics = cfg["physics"]
    udf     = cfg["udf"]
    B       = physics["B"]
    W       = physics["W"]
    L       = physics["L"]
    ft      = physics["final_time"]
    wt      = physics["write_output_time"]
    st      = physics["screen_output_time"]
    CFL     = cfg["CFL"]
    order   = cfg["order_scheme"]
    sus     = cfg["stress_update_scheme"]

    with open(out_filename, "w") as f:
        f.write("# Auto-generated MPM input file\n")

        write_block(f, [
            ("mpm.prob_lo",        f"{-B} {-B} 0.0"),
            ("mpm.prob_hi",        f"{B} {B} {L}"),
            ("mpm.ncells",         f"{grid['nx']} {grid['ny']} {grid['nz']}"),
            ("mpm.max_grid_size",  f"{max(grid['nx'], grid['ny'], grid['nz'])}"),
            ("mpm.is_it_periodic", "0 0 0"),
        ], comment="Geometry Parameters")

        write_block(f, [
            ("#restart_checkfile", "\"\"")
        ], comment="AMR Parameters")

        write_block(f, [
            ("mpm.use_autogen",                 "0"),
            ("mpm.mincoords_autogen",           "0.0 0.0 0.0"),
            ("mpm.maxcoords_autogen",           "1.0 1.0 1.0"),
            ("mpm.vel_autogen",                 "0.0 0.0 0.0"),
            ("mpm.constmodel_autogen",          "0"),
            ("mpm.dens_autogen",                "1.0"),
            ("mpm.E_autogen",                   "1e7"),
            ("mpm.nu_autogen",                  "0.3"),
            ("mpm.bulkmod_autogen",             "2e6"),
            ("mpm.Gama_pres_autogen",           "7"),
            ("mpm.visc_autogen",                "0.001"),
            ("mpm.multi_part_per_cell_autogen", "1"),
            ("mpm.particle_file",               f"\"{particle_filename}\""),
        ], comment="Input Material Points")

        write_block(f, [
            ("mpm.prefix_particlefilename",   f"\"{output_tag}/plt\""),
            ("mpm.prefix_gridfilename",       f"\"{output_tag}/nplt\""),
            ("mpm.prefix_densityfilename",    f"\"{output_tag}/dens\""),
            ("mpm.prefix_checkpointfilename", f"\"{output_tag}/chk\""),
            ("mpm.prefix_asciifilename",      f"\"{output_tag}/matpnt\""),
            ("mpm.diagnostic_output_folder",  f"\"./Diagnostics/{output_tag}\""),
            ("mpm.num_of_digits_in_filenames", "6"),
            ("mpm.write_ascii",               "1"),
        ], comment="Output Parameters")

        write_block(f, [
            ("mpm.final_time",         f"{ft}"),
            ("mpm.max_steps",          "10000000"),
            ("mpm.screen_output_time", f"{st}"),
            ("mpm.write_output_time",  f"{wt}"),
            ("mpm.num_redist",         "1"),
        ], comment="Simulation Run Parameters")

        write_block(f, [
            ("mpm.fixed_timestep", "0"),
            ("mpm.timestep",       "1.0e-5"),
            ("mpm.CFL",            f"{CFL}"),
            ("mpm.dt_min_limit",   "1e-12"),
            ("mpm.dt_max_limit",   "1e+00"),
        ], comment="Timestepping Parameters")

        write_block(f, [
            ("mpm.levset_output",       "0"),
            ("mpm.levset_smoothfactor", "1.0"),
            ("mpm.levset_gridratio",    "1"),
        ], comment="Levelset Parameters")

        write_block(f, [
            ("mpm.order_scheme",         f"{order}"),
            ("mpm.alpha_pic_flip",       "1.0"),
            ("mpm.stress_update_scheme", f"{sus}"),
            ("mpm.mass_tolerance",       "1e-18"),
        ], comment="Numerical Schemes")

        write_block(f, [
            ("mpm.gravity",                        "0.0 0.0 0.0"),
            ("mpm.applied_strainrate_time",        "0.0"),
            ("mpm.applied_strainrate",             "0.0"),
            ("mpm.calculate_strain_based_on_delta","0"),
            ("mpm.external_loads",                 "0"),
            ("mpm.force_slab_lo",                  "0.0 0.0 0.0"),
            ("mpm.force_slab_hi",                  "1.0 1.0 1.0"),
            ("mpm.extforce",                       "0.0 0.0 0.0"),
        ], comment="Physics Parameters")

        write_block(f, [
            ("mpm.bc_xlo_mom",        "slip"),
            ("mpm.bc_xhi_mom",        "slip"),
            ("mpm.bc_ylo_mom",        "slip"),
            ("mpm.bc_yhi_mom",        "slip"),
            ("mpm.bc_zlo_mom",        "noslip"),
            ("mpm.bc_zhi_mom",        "noslip"),
            ("mpm.bc_zhi_mom.udf_lib",  f"\"{udf['zhi_lib']}\""),
            ("mpm.bc_zhi_mom.udf_func", f"\"{udf['zhi_func']}\""),
            ("mpm.levelset_bc",       "1"),
            ("mpm.levelset_wall_mu",  "0.0"),
        ], comment="Boundary Conditions")

        write_block(f, [
            ("mpm.print_diagnostics",        "1"),
            ("mpm.do_calculate_tke_tse",     "1"),
            ("mpm.do_calculate_mwa_velcomp", "1"),
            ("mpm.do_calculate_mwa_velmag",  "0"),
            ("mpm.do_calculate_minmaxpos",   "1"),
            ("mpm.write_diag_output_time",   f"{wt}"),
        ], comment="Diagnostics Parameters")

    print(f"WROTE: {out_filename}")


def main():
    args = parse_cli()
    with open(args.config, "r") as fh:
        cfg = json.load(fh)

    grid    = cfg["grid"]
    ppc     = tuple(cfg["ppc"])
    bodies  = cfg["bodies"]
    physics = cfg["physics"]
    density = bodies[0]["constitutive_model"].get("density", 1000.0)

    shape_cfg = bodies[0]["shape"]
    shape = Block(shape_cfg["xmin"], shape_cfg["xmax"],
                  shape_cfg["ymin"], shape_cfg["ymax"],
                  shape_cfg["zmin"], shape_cfg["zmax"])

    vel_cfg = bodies[0]["initial_velocity"]
    vx0 = vel_cfg.get("vx", 0.0)
    vy0 = vel_cfg.get("vy", 0.0)
    vz0 = vel_cfg.get("vz", 0.0)

    matpt_filename = cfg["materialpoint_filename"]

    def chunk_iter():
        yield from generate_particle_chunks(
            grid=grid,
            ppc=ppc,
            shape=shape,
            cm_cfg=bodies[0]["constitutive_model"],
            density=density,
            vx0=vx0, vy0=vy0, vz0=vz0,
        )

    total = write_particles_ascii(matpt_filename, chunk_iter())
    print(f"Generated {total} material points → {matpt_filename}")

    output_tag    = cfg.get("output_tag", "3D_Twisted_Column")
    input_filename = cfg["input_filename"]
    write_inputs_file(cfg, output_tag, matpt_filename, input_filename)

    print("""
    ┌───────────────────────────────────────────────────────────────────────────────┐
    │  IMPORTANT: Review the input file and build the UDF before running.           │
    │                                                                               │
    │    cd UDF && make                                                             │
    │                                                                               │
    │  The rotating-top UDF (wall_vel_twist) applies:                              │
    │    vx = -omega * y,   vy = omega * x,   vz = 0                              │
    │  at the z=L face, giving a solid-body angular velocity omega about z.        │
    └───────────────────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
