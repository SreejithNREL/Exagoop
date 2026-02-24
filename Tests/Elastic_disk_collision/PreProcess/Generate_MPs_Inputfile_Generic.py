#!/usr/bin/env python3
import argparse
import json
import hashlib
import importlib.util
from typing import Tuple, Callable, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import os

def die(msg: str):
    raise SystemExit(f"[ERROR] {msg}")

def parse_cli():
    p = argparse.ArgumentParser(
        description="General MPM preprocessor: particles + input file from JSON config"
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file",
    )
    return p.parse_args()


# ------------------------------------------------------------
# Shape generation routines
# ------------------------------------------------------------
class ShapeBase:
    def contains(self, p):
        raise NotImplementedError


class Rectangle(ShapeBase):
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

    def contains(self, p):
        x, y = p[:2]
        return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)


class Circle(ShapeBase):
    def __init__(self, center, radius):
        self.cx, self.cy = center
        self.r = radius

    def contains(self, p):
        x, y = p[:2]
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r ** 2


class AnnularCircle(ShapeBase):
    def __init__(self, center, r1, r2):
        self.cx, self.cy = center
        self.r1, self.r2 = r1, r2

    def contains(self, p):
        x, y = p[:2]
        r2 = (x - self.cx) ** 2 + (y - self.cy) ** 2
        return self.r1 ** 2 <= r2 <= self.r2 ** 2


class AnnularRectangle(ShapeBase):
    def __init__(self, outer, inner):
        self.outer = Rectangle(*outer)
        self.inner = Rectangle(*inner)

    def contains(self, p):
        return self.outer.contains(p) and not self.inner.contains(p)


class Sphere(ShapeBase):
    def __init__(self, center, radius):
        self.cx, self.cy, self.cz = center
        self.r = radius

    def contains(self, p):
        x, y, z = p
        return (
            (x - self.cx) ** 2
            + (y - self.cy) ** 2
            + (z - self.cz) ** 2
            <= self.r ** 2
        )

class AnnularSphere(ShapeBase):
    def __init__(self, center, r1, r2):
        self.cx, self.cy, self.cz = center
        self.r1, self.r2 = r1, r2

    def contains(self, p):
        x, y, z = p
        d2 = (x - self.cx) ** 2 + (y - self.cy) ** 2 + (z - self.cz) ** 2
        return self.r1 ** 2 <= d2 <= self.r2 ** 2


class Block(ShapeBase):
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax

    def contains(self, p):
        x, y, z = p
        return (
            self.xmin <= x <= self.xmax
            and self.ymin <= y <= self.ymax
            and self.zmin <= z <= self.zmax
        )

def make_shape(shape_cfg: Optional[dict], dimensions: int) -> Optional[ShapeBase]:
    if shape_cfg is None:
        return None

    t = shape_cfg["type"]  

    # ------------------------------------------------------------
    # 1D interval shape
    # ------------------------------------------------------------
    if dimensions == 1 and t == "interval":
        x0 = shape_cfg["x_start"]
        x1 = shape_cfg["x_end"]

        class IntervalShape:
            def contains(self, pt):
                # pt is (x,) or x
                x = pt[0] if isinstance(pt, (tuple, list)) else pt
                return (x >= x0) and (x <= x1)

        return IntervalShape()


    # 2D shapes
    if t == "rectangle":
        return Rectangle(
            shape_cfg["xmin"],
            shape_cfg["xmax"],
            shape_cfg["ymin"],
            shape_cfg["ymax"],
        )
    if t == "circle":
        return Circle(shape_cfg["center"], shape_cfg["radius"])
    if t == "annular_circle":
        return AnnularCircle(
            shape_cfg["center"],
            shape_cfg["r_inner"],
            shape_cfg["r_outer"],
        )
    if t == "annular_rectangle":
        return AnnularRectangle(shape_cfg["outer"], shape_cfg["inner"])

    # 3D shapes
    if t == "sphere":
        return Sphere(shape_cfg["center"], shape_cfg["radius"])
    if t == "annular_sphere":
        return AnnularSphere(
            shape_cfg["center"],
            shape_cfg["r_inner"],
            shape_cfg["r_outer"],
        )
    if t == "block":
        return Block(
            shape_cfg["xmin"],
            shape_cfg["xmax"],
            shape_cfg["ymin"],
            shape_cfg["ymax"],
            shape_cfg["zmin"],
            shape_cfg["zmax"],
        )
    die(f"Unsupported shape type: {t}")
    return None


def ppc_offsets(N: int) -> np.ndarray:
    i = np.arange(1, N + 1)
    return (2 * i - 1) / (2 * N)


def generate_particles_and_return(
    dimensions: int,
    grid: dict,
    ppc: Tuple[int, ...],
    constitutive_model: dict,
    enable_temperature: bool,
    shape_cfg: Optional[dict],
    velocity_function: Callable[[float, float, float], Tuple[float, float, float]],
    temperature_function: Optional[
        Callable[[float, float, float], Tuple[float, float, float, float]]
    ],
    out_particles: str = "mpm_particles.dat",
) -> Tuple[int, float]:
    if dimensions not in [1, 2, 3]:
        die("dimensions must be 1, 2, or 3")
    if len(ppc) != dimensions:
        die("ppc tuple length must match dimensions")

    xmin, xmax, nx = grid["xmin"], grid["xmax"], grid["nx"]
    dx = (xmax - xmin) / nx
    dx1 = dx

    if dimensions >= 2:
        ymin, ymax, ny = grid["ymin"], grid["ymax"], grid["ny"]
        dy = (ymax - ymin) / ny
    else:
        ymin, ymax, ny = 0.0, 1.0, 1
        dy = 1.0

    if dimensions == 3:
        zmin, zmax, nz = grid["zmin"], grid["zmax"], grid["nz"]
        dz = (zmax - zmin) / nz
    else:
        zmin = -0.5 * dx
        zmax = 0.5 * dx
        nz = 1
        dz = (zmax - zmin) / nz

    offsets = [ppc_offsets(ppc[d]) for d in range(dimensions)]

    #shape_obj = None if dimensions == 1 else make_shape(shape_cfg, dimensions)
    
    if shape_cfg is None:
        shape_obj = None
    else:
        shape_obj = make_shape(shape_cfg, dimensions)


    if dimensions == 1:
        vol_cell = dx
        vol_particle = dx / ppc[0]
    elif dimensions == 2:
        vol_cell = dx * dy
        vol_particle = vol_cell / (ppc[0] * ppc[1])
    else:
        vol_cell = dx * dy * dz
        vol_particle = vol_cell / np.prod(ppc)

    rad = (3.0 / 4.0 * vol_particle / np.pi) ** (1.0 / 3.0)
    phase = 0
    dens = 1.0

    cm_type = constitutive_model["type"]

    if cm_type == "elastic":
        cm_extra = {
            "E": constitutive_model["E"],
            "nu": constitutive_model["nu"],
        }
        cm_id = 0
    elif cm_type == "fluid":
        cm_extra = {
            "Bulk_modulus": constitutive_model["Bulk_modulus"],
            "Gama_pressure": constitutive_model["Gama_pressure"],
            "Dynamic_viscosity": constitutive_model["Dynamic_viscosity"],
        }
        cm_id = 1
    else:
        cm_extra = {k: v for k, v in constitutive_model.items() if k != "type"}
        cm_id = -1

    def column_names():
        cols = ["phase", "x"]
        if dimensions >= 2:
            cols.append("y")
        if dimensions == 3:
            cols.append("z")

        # velocities
        if dimensions == 1:
            cols += ["vx"]
        elif dimensions == 2:
            cols += ["vx", "vy"]
        else:
            cols += ["vx", "vy", "vz"]

        cols += ["radius", "density"]
        cols += list(cm_extra.keys())

        if enable_temperature:
            cols += ["T", "spheat", "thermcond", "heatsrc"]

        return cols

    npart = 0

    for i in range(nx):
        cx = xmin + i * dx
        for ox in offsets[0]:
            px = cx + ox * dx

            if dimensions == 1:
                npart += 1
                continue

            for j in range(ny):
                cy = ymin + j * dy
                for oy in offsets[1]:
                    py = cy + oy * dy

                    if dimensions == 2:
                        if shape_obj is None or shape_obj.contains((px, py)):
                            npart += 1
                        continue

                    for k in range(nz):
                        cz = zmin + k * dz
                        for oz in offsets[2]:
                            pz = cz + oz * dz
                            if shape_obj is None or shape_obj.contains((px, py, pz)):
                                npart += 1

    with open(out_particles, "w") as f:
        f.write(f"dim: {dimensions}\n")
        f.write(f"number_of_material_points: {npart}\n")
        f.write("# " + " ".join(column_names()) + "\n")

        for i in range(nx):
            cx = xmin + i * dx
            for ox in offsets[0]:
                px = cx + ox * dx

                if dimensions == 1:
                    vx, vy, vz = velocity_function(px, 0.0, 0.0)
                    cols = [
                        f"{phase:d}",
                        f"{px:.6e}",
                        f"{rad:.6e}",
                        f"{dens:.6e}",
                        f"{vx:.6e}",
                        f"{cm_id:d}"      

                    ]
                    for v in cm_extra.values():
                        cols.append(f"{v:.6e}")
                    if enable_temperature:
                        T, spheat, thermcond, heatsrc = temperature_function(
                            px, 0.0, 0.0
                        )
                        cols += [
                            f"{T:.6e}",
                            f"{spheat:.6e}",
                            f"{thermcond:.6e}",
                            f"{heatsrc:.6e}",
                        ]
                    f.write(" ".join(cols) + "\n")
                    continue

                for j in range(ny):
                    cy = ymin + j * dy
                    for oy in offsets[1]:
                        py = cy + oy * dy

                        if dimensions == 2:
                            if shape_obj is not None and not shape_obj.contains((px, py)):
                                continue

                            vx, vy, vz = velocity_function(px, py, 0.0)
                            cols = [
                                f"{phase:d}",
                                f"{px:.6e}",
                                f"{py:.6e}",
                                f"{rad:.6e}",
                                f"{dens:.6e}",
                                f"{vx:.6e}",
                                f"{vy:.6e}",
                                f"{cm_id:d}"                                
                            ]
                            for v in cm_extra.values():
                                cols.append(f"{v:.6e}")
                            if enable_temperature:
                                T, spheat, thermcond, heatsrc = temperature_function(
                                    px, py, 0.0
                                )
                                cols += [
                                    f"{T:.6e}",
                                    f"{spheat:.6e}",
                                    f"{thermcond:.6e}",
                                    f"{heatsrc:.6e}",
                                ]
                            f.write(" ".join(cols) + "\n")
                            continue

                        for k in range(nz):
                            cz = zmin + k * dz
                            for oz in offsets[2]:
                                pz = cz + oz * dz
                                if shape_obj is not None and not shape_obj.contains(
                                    (px, py, pz)
                                ):
                                    continue

                                vx, vy, vz = velocity_function(px, py, pz)
                                cols = [
                                    f"{phase:d}",
                                    f"{px:.6e}",
                                    f"{py:.6e}",
                                    f"{pz:.6e}",
                                    f"{rad:.6e}",
                                    f"{dens:.6e}",
                                    f"{vx:.6e}",
                                    f"{vy:.6e}",
                                    f"{vz:.6e}",
                                    f"{cm_id:d}"                                    
                                ]
                                for v in cm_extra.values():
                                    cols.append(f"{v:.6e}")
                                if enable_temperature:
                                    T, spheat, thermcond, heatsrc = (
                                        temperature_function(px, py, pz)
                                    )
                                    cols += [
                                        f"{T:.6e}",
                                        f"{spheat:.6e}",
                                        f"{thermcond:.6e}",
                                        f"{heatsrc:.6e}",
                                    ]
                                f.write(" ".join(cols) + "\n")

    print(f"WROTE: {out_particles} with {npart} particles (cm_type={cm_type}, id={cm_id})")
    return npart, dx1

def generate_particle_chunks(
    dimensions,
    grid,
    ppc,
    constitutive_model,
    enable_temperature,
    shape_cfg,
    velocity_function,
    temperature_function,
    cell_block=(32, 32, 8),
    chunk_size=200_000,   # number of particles per yielded chunk
):
    """
    Streaming particle generator for 1D, 2D, or 3D.
    Yields chunks of particles as dictionaries of NumPy arrays.
    Never stores all particles in memory at once.
    """

    import numpy as np

    # ------------------------------------------------------------
    # Grid setup
    # ------------------------------------------------------------
    xmin, xmax, nx = grid["xmin"], grid["xmax"], grid["nx"]
    dx = (xmax - xmin) / nx

    if dimensions >= 2:
        ymin, ymax, ny = grid["ymin"], grid["ymax"], grid["ny"]
        dy = (ymax - ymin) / ny
    else:
        ymin, ymax, ny = 0.0, 1.0, 1
        dy = 1.0

    if dimensions == 3:
        zmin, zmax, nz = grid["zmin"], grid["zmax"], grid["nz"]
        dz = (zmax - zmin) / nz
    else:
        zmin, zmax, nz = -0.5 * dx, 0.5 * dx, 1
        dz = (zmax - zmin) / nz

    offsets = [ppc_offsets(ppc[d]) for d in range(dimensions)]
    shape_obj = None if shape_cfg is None else make_shape(shape_cfg, dimensions)

    # ------------------------------------------------------------
    # Volume, radius, density
    # ------------------------------------------------------------
    if dimensions == 1:
        vol_cell = dx
        vol_particle = dx / ppc[0]
    elif dimensions == 2:
        vol_cell = dx * dy
        vol_particle = vol_cell / (ppc[0] * ppc[1])
    else:
        vol_cell = dx * dy * dz
        vol_particle = vol_cell / np.prod(ppc)

    rad = 0.025
    dens = 997.0
    phase = 0

    # ------------------------------------------------------------
    # Constitutive model
    # ------------------------------------------------------------
    cm_type = constitutive_model["type"]

    if cm_type == "elastic":
        cm_extra = {
            "E": constitutive_model["E"],
            "nu": constitutive_model["nu"],
        }
        cm_id = 0
    
    elif cm_type == "fluid":
        cm_extra = {
            "Bulk_modulus": constitutive_model["Bulk_modulus"],
            "Gamma_pressure": constitutive_model["Gamma_pressure"],
            "Dynamic_viscosity": constitutive_model["Dynamic_viscosity"],
        }
        cm_id = 1    
    else:
        # Generic fallback for custom models
        cm_extra = {k: v for k, v in constitutive_model.items() if k != "type"}
        cm_id = -1

    # ------------------------------------------------------------
    # Block generators
    # ------------------------------------------------------------
    def block_1d(ix0, ix1):
        ix = np.arange(ix0, ix1)
        cx = xmin + ix * dx
    
        # ppc offsets in 1D
        PX = cx[:, None] + offsets[0][None, :] * dx
        PX = PX.ravel()
    
        # ------------------------------------------------------------
        # 1D shape handling
        # ------------------------------------------------------------
        if shape_cfg is None:
            # No shape → full domain
            return PX
    
        # If shape_cfg defines x_start/x_end, use them
        x_start = shape_cfg.get("x_start", None)
        x_end   = shape_cfg.get("x_end", None)
    
        if x_start is not None and x_end is not None:
            mask = (PX >= x_start) & (PX <= x_end)
            PX = PX[mask]
            return PX
    
        # Otherwise fall back to shape_obj.contains
        if shape_obj is not None:
            mask = np.array([shape_obj.contains((x,)) for x in PX])
            PX = PX[mask]
    
        return PX


    def block_2d(ix0, ix1, iy0, iy1):
        ix = np.arange(ix0, ix1)
        iy = np.arange(iy0, iy1)
        cx = xmin + ix * dx
        cy = ymin + iy * dy
    
        CX, CY = np.meshgrid(cx, cy, indexing="ij")
    
        ox = offsets[0]
        oy = offsets[1]
    
        OX, OY = np.meshgrid(ox, oy, indexing="ij")
    
        PX = CX[:, :, None, None] + OX[None, None, :, :] * dx
        PY = CY[:, :, None, None] + OY[None, None, :, :] * dy
    
        PX = PX.reshape(-1)
        PY = PY.reshape(-1)
    
        if shape_obj is not None:
            mask = np.array([shape_obj.contains((x, y)) for x, y in zip(PX, PY)])
            PX = PX[mask]
            PY = PY[mask]
    
        return PX, PY


    def block_3d(ix0, ix1, iy0, iy1, iz0, iz1):
        ix = np.arange(ix0, ix1)
        iy = np.arange(iy0, iy1)
        iz = np.arange(iz0, iz1)
        cx = xmin + ix * dx
        cy = ymin + iy * dy
        cz = zmin + iz * dz
        CX, CY, CZ = np.meshgrid(cx, cy, cz, indexing="ij")
        PX = CX[:, :, :, None] + offsets[0][None, None, None, :] * dx
        PY = CY[:, :, :, None] + offsets[1][None, None, None, :] * dy
        PZ = CZ[:, :, :, None] + offsets[2][None, None, None, :] * dz
        PX = PX.ravel()
        PY = PY.ravel()
        PZ = PZ.ravel()
        if shape_obj is not None:
            mask = np.array([shape_obj.contains((x, y, z)) for x, y, z in zip(PX, PY, PZ)])
            PX = PX[mask]
            PY = PY[mask]
            PZ = PZ[mask]
        return PX, PY, PZ

    # ------------------------------------------------------------
    # Chunk buffers
    # ------------------------------------------------------------
    buf = {
        "phase": [],
        "x": [],
        "vx": [],
        "radius": [],
        "density": [],
        "cm_id": [],
    }

    if dimensions >= 2:
        buf["y"] = []
        buf["vy"] = []
    if dimensions == 3:
        buf["z"] = []
        buf["vz"] = []

    for k in cm_extra.keys():
        buf[k] = []

    if enable_temperature:
        for k in ["T", "spheat", "thermcond", "heatsrc"]:
            buf[k] = []

    # ------------------------------------------------------------
    # Helper: flush chunk
    # ------------------------------------------------------------
    def flush_chunk():
        n = len(buf["x"])
        if n == 0:
            return None
    
        # Convert lists → arrays
        chunk = {k: np.asarray(v) for k, v in buf.items()}
    
        # ------------------------------------------------------------
        # Spatial sort for MPI performance
        # ------------------------------------------------------------
        if dimensions == 1:
            order = np.argsort(chunk["x"])
        elif dimensions == 2:
            order = np.lexsort((chunk["y"], chunk["x"]))
        else:  # 3D
            order = np.lexsort((chunk["z"], chunk["y"], chunk["x"]))
    
        for k in chunk:
            chunk[k] = chunk[k][order]
    
        # Clear buffers
        for k in buf.keys():
            buf[k].clear()
    
        return chunk


    # ------------------------------------------------------------
    # Main loop over blocks
    # ------------------------------------------------------------
    bx, by, bz = cell_block

    if dimensions == 1:
        for ix0 in range(0, nx, bx):
            ix1 = min(ix0 + bx, nx)
            PX = block_1d(ix0, ix1)
            # y,z are zero in 1D
            PY = np.zeros_like(PX)
            PZ = np.zeros_like(PX)

            for px, py, pz in zip(PX, PY, PZ):
                vx, vy, vz = velocity_function(px, py, pz)

                if enable_temperature:
                    T0, SP0, K0, Q0 = temperature_function(px, py, pz)

                buf["phase"].append(phase)
                buf["x"].append(px)
                buf["vx"].append(vx)
                buf["radius"].append(rad)
                buf["density"].append(dens)
                buf["cm_id"].append(cm_id)

                for k, v in cm_extra.items():
                    buf[k].append(v)

                if enable_temperature:
                    buf["T"].append(T0)
                    buf["spheat"].append(SP0)
                    buf["thermcond"].append(K0)
                    buf["heatsrc"].append(Q0)

                if len(buf["x"]) >= chunk_size:
                    chunk = flush_chunk()
                    if chunk is not None:
                        yield chunk

    elif dimensions == 2:
        for ix0 in range(0, nx, bx):
            ix1 = min(ix0 + bx, nx)
            for iy0 in range(0, ny, by):
                iy1 = min(iy0 + by, ny)

                PX, PY = block_2d(ix0, ix1, iy0, iy1)
                PZ = np.zeros_like(PX)

                # ------------------------------------------------------------
                # Sort the block BEFORE appending to buffers
                # ------------------------------------------------------------
                order = np.lexsort((PY, PX))
                PX = PX[order]
                PY = PY[order]
                
                for px, py, pz in zip(PX, PY, PZ):
                    vx, vy, vz = velocity_function(px, py, pz)
                
                    if enable_temperature:
                        T0, SP0, K0, Q0 = temperature_function(px, py, pz)
                
                    buf["phase"].append(phase)
                    buf["x"].append(px)
                    buf["y"].append(py)
                    buf["vx"].append(vx)
                    buf["vy"].append(vy)
                    buf["radius"].append(rad)
                    buf["density"].append(dens)
                    buf["cm_id"].append(cm_id)
                
                    for k, v in cm_extra.items():
                        buf[k].append(v)
                
                    if enable_temperature:
                        buf["T"].append(T0)
                        buf["spheat"].append(SP0)
                        buf["thermcond"].append(K0)
                        buf["heatsrc"].append(Q0)
                
                # ------------------------------------------------------------
                # Flush AFTER each block — one chunk per block
                # ------------------------------------------------------------
                chunk = flush_chunk()
                if chunk is not None:
                    yield chunk


    else:  # dimensions == 3
        for ix0 in range(0, nx, bx):
            ix1 = min(ix0 + bx, nx)
            for iy0 in range(0, ny, by):
                iy1 = min(iy0 + by, ny)
                for iz0 in range(0, nz, bz):
                    iz1 = min(iz0 + bz, nz)

                    PX, PY, PZ = block_3d(ix0, ix1, iy0, iy1, iz0, iz1)

                    for px, py, pz in zip(PX, PY, PZ):
                        vx, vy, vz = velocity_function(px, py, pz)

                        if enable_temperature:
                            T0, SP0, K0, Q0 = temperature_function(px, py, pz)

                        buf["phase"].append(phase)
                        buf["x"].append(px)
                        buf["y"].append(py)
                        buf["z"].append(pz)
                        buf["vx"].append(vx)
                        buf["vy"].append(vy)
                        buf["vz"].append(vz)
                        buf["radius"].append(rad)
                        buf["density"].append(dens)
                        buf["cm_id"].append(cm_id)

                        for k, v in cm_extra.items():
                            buf[k].append(v)

                        if enable_temperature:
                            buf["T"].append(T0)
                            buf["spheat"].append(SP0)
                            buf["thermcond"].append(K0)
                            buf["heatsrc"].append(Q0)

                        if len(buf["x"]) >= chunk_size:
                            chunk = flush_chunk()
                            if chunk is not None:
                                yield chunk

    # ------------------------------------------------------------
    # Final flush
    # ------------------------------------------------------------
    chunk = flush_chunk()
    if chunk is not None:
        yield chunk



def generate_particles_vectorized(
    dimensions,
    grid,
    ppc,
    constitutive_model,
    enable_temperature,
    shape_cfg,
    velocity_function,
    temperature_function,
    out_particles=None,
    output_format="ascii",   # "ascii", "hdf5", "memory"
    cell_block=(32, 32, 8),
):
    import numpy as np

    # Optional import for HDF5 mode
    if output_format == "hdf5":
        import h5py

    # ------------------------------------------------------------
    # Validate mode
    # ------------------------------------------------------------
    if output_format not in ("ascii", "hdf5", "memory"):
        raise ValueError("output_format must be 'ascii', 'hdf5', or 'memory'")

    is_ascii  = (output_format == "ascii")
    is_hdf5   = (output_format == "hdf5")
    is_memory = (output_format == "memory")

    if (is_ascii or is_hdf5) and out_particles is None:
        raise ValueError("out_particles must be provided for ascii or hdf5 output")

    # ------------------------------------------------------------
    # Grid setup
    # ------------------------------------------------------------
    xmin, xmax, nx = grid["xmin"], grid["xmax"], grid["nx"]
    dx = (xmax - xmin) / nx

    if dimensions >= 2:
        ymin, ymax, ny = grid["ymin"], grid["ymax"], grid["ny"]
        dy = (ymax - ymin) / ny
    else:
        ymin, ymax, ny = 0.0, 1.0, 1
        dy = 1.0

    if dimensions == 3:
        zmin, zmax, nz = grid["zmin"], grid["zmax"], grid["nz"]
        dz = (zmax - zmin) / nz
    else:
        zmin, zmax, nz = -0.5 * dx, 0.5 * dx, 1
        dz = (zmax - zmin) / nz

    offsets = [ppc_offsets(ppc[d]) for d in range(dimensions)]
    shape_obj = None if shape_cfg is None else make_shape(shape_cfg, dimensions)

    # ------------------------------------------------------------
    # Volume, radius, density
    # ------------------------------------------------------------
    if dimensions == 1:
        vol_cell = dx
        vol_particle = dx / ppc[0]
    elif dimensions == 2:
        vol_cell = dx * dy
        vol_particle = vol_cell / (ppc[0] * ppc[1])
    else:
        vol_cell = dx * dy * dz
        vol_particle = vol_cell / np.prod(ppc)

    rad = (3.0 / 4.0 * vol_particle / np.pi) ** (1.0 / 3.0)
    dens = 1.0
    phase = 0

    # ------------------------------------------------------------
    # Constitutive model
    # ------------------------------------------------------------
    cm_type = constitutive_model["type"]
    if cm_type == "elastic":
        cm_extra = {"E": constitutive_model["E"], "nu": constitutive_model["nu"]}
        cm_id = 0
    else:
        cm_extra = {k: v for k, v in constitutive_model.items() if k != "type"}
        cm_id = -1

    # ------------------------------------------------------------
    # ASCII header helper
    # ------------------------------------------------------------
    def ascii_column_names():
        cols = ["phase", "x"]
        if dimensions >= 2:
            cols.append("y")
        if dimensions == 3:
            cols.append("z")

        if dimensions == 1:
            cols += ["vx"]
        elif dimensions == 2:
            cols += ["vx", "vy"]
        else:
            cols += ["vx", "vy", "vz"]

        cols += ["radius", "density"]
        cols += list(cm_extra.keys())

        if enable_temperature:
            cols += ["T", "spheat", "thermcond", "heatsrc"]

        return cols

    # ------------------------------------------------------------
    # Output mode setup
    # ------------------------------------------------------------
    if is_memory:
        mem = {k: [] for k in ["x","y","z","vx","vy","vz","radius","density","cm_id"]}
        mem["phase"] = []
        for k in cm_extra.keys():
            mem[k] = []
        if enable_temperature:
            for k in ["T","spheat","thermcond","heatsrc"]:
                mem[k] = []

    elif is_ascii:
        f = open(out_particles, "w")
        # We do NOT know npart yet → write placeholder, fix later
        f.write(f"dim: {dimensions}\n")
        f.write(f"number_of_material_points: 0\n")
        f.write("# " + " ".join(ascii_column_names()) + "\n")
        ascii_count = 0

    elif is_hdf5:
        h5 = h5py.File(out_particles, "w")
        h5["dim"] = dimensions
        h5["number_of_material_points"] = 0

        def create_dset(name):
            return h5.create_dataset(name, shape=(0,), maxshape=(None,), dtype="f8")

        dsets = {}
        for name in ["phase","x","radius","density","vx","cm_id"]:
            dsets[name] = create_dset(name)
        if dimensions >= 2:
            dsets["y"] = create_dset("y")
            dsets["vy"] = create_dset("vy")
        if dimensions == 3:
            dsets["z"] = create_dset("z")
            dsets["vz"] = create_dset("vz")

        for k in cm_extra.keys():
            dsets[k] = create_dset(k)

        if enable_temperature:
            for k in ["T","spheat","thermcond","heatsrc"]:
                dsets[k] = create_dset(k)

        buf = {k: [] for k in dsets.keys()}
        total_npart = 0

        def flush():
            nonlocal total_npart
            n = len(buf["x"])
            if n == 0:
                return
            old = total_npart
            new = old + n
            for name, dset in dsets.items():
                dset.resize((new,))
                dset[old:new] = np.asarray(buf[name])
                buf[name].clear()
            total_npart = new

    # ------------------------------------------------------------
    # Vectorized block generator (2D only)
    # ------------------------------------------------------------
    def block_2d(ix0, ix1, iy0, iy1):
        ix = np.arange(ix0, ix1)
        iy = np.arange(iy0, iy1)
        cx = xmin + ix * dx
        cy = ymin + iy * dy
        CX, CY = np.meshgrid(cx, cy, indexing="ij")
        PX = CX[:, :, None] + offsets[0][None, None, :] * dx
        PY = CY[:, :, None] + offsets[1][None, None, :] * dy
        PX = PX.ravel()
        PY = PY.ravel()
        if shape_obj is not None:
            mask = np.array([shape_obj.contains((x, y)) for x, y in zip(PX, PY)])
            PX = PX[mask]
            PY = PY[mask]
        return PX, PY

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    bx, by, bz = cell_block

    for ix0 in range(0, nx, bx):
        ix1 = min(ix0 + bx, nx)

        for iy0 in range(0, ny, by):
            iy1 = min(iy0 + by, ny)

            PX, PY = block_2d(ix0, ix1, iy0, iy1)
            PZ = np.zeros_like(PX)

            for px, py, pz in zip(PX, PY, PZ):
                vx, vy, vz = velocity_function(px, py, pz)

                if enable_temperature:
                    T0, SP0, K0, Q0 = temperature_function(px, py, pz)

                # -------------------------
                # MEMORY MODE
                # -------------------------
                if is_memory:
                    mem["phase"].append(phase)
                    mem["x"].append(px)
                    mem["y"].append(py)
                    mem["z"].append(pz)
                    mem["vx"].append(vx)
                    mem["vy"].append(vy)
                    mem["vz"].append(vz)
                    mem["radius"].append(rad)
                    mem["density"].append(dens)
                    mem["cm_id"].append(cm_id)
                    for k, v in cm_extra.items():
                        mem[k].append(v)
                    if enable_temperature:
                        mem["T"].append(T0)
                        mem["spheat"].append(SP0)
                        mem["thermcond"].append(K0)
                        mem["heatsrc"].append(Q0)

                # -------------------------
                # ASCII MODE
                # -------------------------
                elif is_ascii:
                    cols = [f"{phase:d}", f"{px:.6e}"]
                    if dimensions >= 2:
                        cols.append(f"{py:.6e}")
                    if dimensions == 3:
                        cols.append(f"{pz:.6e}")

                    if dimensions == 1:
                        cols.append(f"{vx:.6e}")
                    elif dimensions == 2:
                        cols += [f"{vx:.6e}", f"{vy:.6e}"]
                    else:
                        cols += [f"{vx:.6e}", f"{vy:.6e}", f"{vz:.6e}"]

                    cols += [f"{rad:.6e}", f"{dens:.6e}"]

                    for v in cm_extra.values():
                        cols.append(f"{v:.6e}")

                    if enable_temperature:
                        cols += [
                            f"{T0:.6e}",
                            f"{SP0:.6e}",
                            f"{K0:.6e}",
                            f"{Q0:.6e}",
                        ]

                    f.write(" ".join(cols) + "\n")
                    ascii_count += 1

                # -------------------------
                # HDF5 MODE
                # -------------------------
                elif is_hdf5:
                    buf["phase"].append(phase)
                    buf["x"].append(px)
                    if dimensions >= 2:
                        buf["y"].append(py)
                    if dimensions == 3:
                        buf["z"].append(pz)
                    buf["vx"].append(vx)
                    if dimensions >= 2:
                        buf["vy"].append(vy)
                    if dimensions == 3:
                        buf["vz"].append(vz)
                    buf["radius"].append(rad)
                    buf["density"].append(dens)
                    buf["cm_id"].append(cm_id)
                    for k, v in cm_extra.items():
                        buf[k].append(v)
                    if enable_temperature:
                        buf["T"].append(T0)
                        buf["spheat"].append(SP0)
                        buf["thermcond"].append(K0)
                        buf["heatsrc"].append(Q0)

            if is_hdf5:
                flush()

    # ------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------
    if is_memory:
        return {k: np.asarray(v) for k, v in mem.items()}

    elif is_ascii:
        f.close()
        # Fix header count
        with open(out_particles, "r+") as f2:
            lines = f2.readlines()
            lines[1] = f"number_of_material_points: {ascii_count}\n"
            f2.seek(0)
            f2.writelines(lines)
        return ascii_count, dx

    elif is_hdf5:
        h5["number_of_material_points"][...] = total_npart
        h5.close()
        return total_npart

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_material_points(
    points: np.ndarray,
    grid: dict,
    dimensions: int,
    output_tag: str,
    *,
    slice_axis: Optional[str] = None,
    slice_value: Optional[float] = None,
    figsize=(8, 6),
):
    fig, ax = plt.subplots(figsize=figsize)

    if dimensions == 1:
        x = points[:, 0]
        xmin, xmax, nx = grid["xmin"], grid["xmax"], grid["nx"]
        dx = (xmax - xmin) / nx

        for i in range(nx + 1):
            ax.axvline(xmin + i * dx, color="lightgray", linewidth=0.8)

        ax.axvline(xmin, color="black", linewidth=2.5)
        ax.axvline(xmax, color="black", linewidth=2.5)

        ax.plot(x, np.zeros_like(x), "o", markersize=4)
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlabel("x")
        ax.set_title("1D Material Points with Grid + Boundary")
        plt.savefig(output_tag)
        return

    if dimensions == 2:
        x = points[:, 0]
        y = points[:, 1]

        xmin, xmax, nx = grid["xmin"], grid["xmax"], grid["nx"]
        ymin, ymax, ny = grid["ymin"], grid["ymax"], grid["ny"]

        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny

        for i in range(nx + 1):
            ax.axvline(xmin + i * dx, color="lightgray", linewidth=0.8)
        for j in range(ny + 1):
            ax.axhline(ymin + j * dy, color="lightgray", linewidth=0.8)

        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2.5,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.plot(x, y, "o", markersize=3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_title("2D Material Points with Grid + Boundary")
        plt.savefig(output_tag)
        return

    if dimensions == 3:
        if slice_axis not in ["x", "y", "z"]:
            die("For 3D visualization, slice_axis must be 'x', 'y', or 'z'")
        if slice_value is None:
            die("For 3D visualization, slice_value must be provided")

        xmin, xmax, nx = grid["xmin"], grid["xmax"], grid["nx"]
        ymin, ymax, ny = grid["ymin"], grid["ymax"], grid["ny"]
        zmin, zmax, nz = grid["zmin"], grid["zmax"], grid["nz"]

        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        dz = (zmax - zmin) / nz

        if slice_axis == "x":
            lo = slice_value - dx
            hi = slice_value + dx
            mask = (points[:, 0] >= lo) & (points[:, 0] <= hi)
            pts = points[mask]
            x2 = pts[:, 1]
            y2 = pts[:, 2]
            xlabel, ylabel = "y", "z"
            xmin2, xmax2 = ymin, ymax
            ymin2, ymax2 = zmin, zmax
            dx2, dy2 = dy, dz
            nx2, ny2 = ny, nz

        elif slice_axis == "y":
            lo = slice_value - dy
            hi = slice_value + dy
            mask = (points[:, 1] >= lo) & (points[:, 1] <= hi)
            pts = points[mask]
            x2 = pts[:, 0]
            y2 = pts[:, 2]
            xlabel, ylabel = "x", "z"
            xmin2, xmax2 = xmin, xmax
            ymin2, ymax2 = zmin, zmax
            dx2, dy2 = dx, dz
            nx2, ny2 = nx, nz

        else:
            lo = slice_value - dz
            hi = slice_value + dz
            mask = (points[:, 2] >= lo) & (points[:, 2] <= hi)
            pts = points[mask]
            x2 = pts[:, 0]
            y2 = pts[:, 1]
            xlabel, ylabel = "x", "y"
            xmin2, xmax2 = xmin, xmax
            ymin2, ymax2 = ymin, ymax
            dx2, dy2 = dx, dy
            nx2, ny2 = nx, ny

        for i in range(nx2 + 1):
            ax.axvline(xmin2 + i * dx2, color="lightgray", linewidth=0.8)
        for j in range(ny2 + 1):
            ax.axhline(ymin2 + j * dy2, color="lightgray", linewidth=0.8)

        rect = patches.Rectangle(
            (xmin2, ymin2),
            xmax2 - xmin2,
            ymax2 - ymin2,
            linewidth=2.5,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.plot(x2, y2, "o", markersize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.set_title(f"3D Slice at {slice_axis}={slice_value} (±1 cell) with Boundary")
        plt.savefig(output_tag)
        return

def read_grid_from_input(filename):
    grid = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            # Normalize separators
            line = line.replace(":", "=")

            if line.startswith("mpm.prob_lo"):
                _, rhs = line.split("=")
                vals = rhs.split()
                grid["xmin"] = float(vals[0])
                grid["ymin"] = float(vals[1])
                grid["zmin"] = float(vals[2])

            elif line.startswith("mpm.prob_hi"):
                _, rhs = line.split("=")
                vals = rhs.split()
                grid["xmax"] = float(vals[0])
                grid["ymax"] = float(vals[1])
                grid["zmax"] = float(vals[2])

            elif line.startswith("mpm.ncells"):
                _, rhs = line.split("=")
                vals = rhs.split()
                grid["nx"] = int(vals[0])
                grid["ny"] = int(vals[1])
                grid["nz"] = int(vals[2])

    # Determine dimensionality
    if grid["nz"] == 0:
        grid["dim"] = 2
    else:
        grid["dim"] = 3

    return grid


def read_particles_ascii(filename):
    """
    Read ASCII particle file written by write_particles_ascii_streaming().
    Returns:
        dim, data_dict
    where data_dict maps column names -> NumPy arrays.
    """

    import numpy as np

    with open(filename, "r") as f:
        # ------------------------------------------------------------
        # Read header
        # ------------------------------------------------------------
        line = f.readline().strip()
        assert line.startswith("dim:")
        dim = int(line.split(":")[1])

        line = f.readline().strip()
        assert line.startswith("number_of_material_points:")
        npart = int(line.split(":")[1])

        # ------------------------------------------------------------
        # Read column names
        # ------------------------------------------------------------
        line = f.readline().strip()
        assert line.startswith("#")
        cols = line[1:].strip().split()

        # Prepare storage
        data = {c: [] for c in cols}

        # ------------------------------------------------------------
        # Read particle rows
        # ------------------------------------------------------------
        for line in f:
            if not line.strip():
                continue
            vals = line.split()
            for c, v in zip(cols, vals):
                # float or int?
                if v.replace(".", "", 1).replace("e", "", 1).replace("-", "", 1).isdigit():
                    data[c].append(float(v))
                else:
                    data[c].append(v)

    # Convert lists → NumPy arrays
    for c in data:
        data[c] = np.asarray(data[c])

    return dim, data


# ------------------------------------------------------------
# Read particle data from HDF5
# ------------------------------------------------------------
def read_particles_h5(filename):
    with h5py.File(filename, "r") as h5:
        dim = int(h5["dim"][()])
        x = np.array(h5["x"])
        y = np.array(h5["y"]) if dim >= 2 else None
        z = np.array(h5["z"]) if dim == 3 else None
    return dim, x, y, z


def plot_1d(x, grid):
    import matplotlib.pyplot as plt
    import numpy as np

    xmin, xmax, nx = grid["xmin"], grid["xmax"], grid["nx"]
    dx = (xmax - xmin) / nx

    fig, ax = plt.subplots(figsize=(12, 3))

    # Plot material points
    ax.scatter(x, np.zeros_like(x), s=20, c="blue", alpha=0.7, label="Material Points")

    # Plot grid cell boundaries
    for i in range(nx + 1):
        xc = xmin + i * dx
        ax.axvline(x=xc, color="gray", lw=0.5)

    # Formatting
    ax.set_title("1D Material Points Overlaid on Grid")
    ax.set_xlabel("x")
    ax.set_yticks([])  # no y-axis in 1D
    ax.set_xlim(xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin))
    ax.grid(False)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Plot 2D particles + grid
# ------------------------------------------------------------
def plot_2d(x, y, grid):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(x, y, s=2, c="blue", alpha=0.6)

    # Draw grid lines
    xmin, xmax, nx = grid["xmin"], grid["xmax"], grid["nx"]
    ymin, ymax, ny = grid["ymin"], grid["ymax"], grid["ny"]

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    # Vertical lines
    for i in range(nx + 1):
        ax.plot([xmin + i * dx, xmin + i * dx], [ymin, ymax], color="gray", lw=0.3)

    # Horizontal lines
    for j in range(ny + 1):
        ax.plot([xmin, xmax], [ymin + j * dy, ymin + j * dy], color="gray", lw=0.3)

    ax.set_title("2D Material Points Overlaid on Grid")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.show()


# ------------------------------------------------------------
# Plot 3D slice
# ------------------------------------------------------------
def plot_3d_slice(x, y, z, grid, slice_axis="z", slice_value=None):
    if slice_value is None:
        # Default slice: mid-plane
        if slice_axis == "x":
            slice_value = 0.5 * (grid["xmin"] + grid["xmax"])
        elif slice_axis == "y":
            slice_value = 0.5 * (grid["ymin"] + grid["ymax"])
        else:
            slice_value = 0.5 * (grid["zmin"] + grid["zmax"])

    # Thickness = one grid cell
    dx = (grid["xmax"] - grid["xmin"]) / grid["nx"]
    dy = (grid["ymax"] - grid["ymin"]) / grid["ny"]
    dz = (grid["zmax"] - grid["zmin"]) / grid["nz"]

    if slice_axis == "x":
        mask = np.abs(x - slice_value) <= dx
        xs, ys = y[mask], z[mask]
        xlabel, ylabel = "y", "z"
    elif slice_axis == "y":
        mask = np.abs(y - slice_value) <= dy
        xs, ys = x[mask], z[mask]
        xlabel, ylabel = "x", "z"
    else:
        mask = np.abs(z - slice_value) <= dz
        xs, ys = x[mask], y[mask]
        xlabel, ylabel = "x", "y"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, s=2, c="red", alpha=0.6)
    ax.set_title(f"3D Slice ({slice_axis} = {slice_value:.3f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    plt.show()

# ------------------------------------------------------------
# Helpers: load particles, write inputs, auto-tag
# ------------------------------------------------------------
def load_particle_positions(filename: str, dimensions: int) -> np.ndarray:
    if dimensions == 1:
        pts = np.loadtxt(filename, comments="#", skiprows=3, usecols=[1])
        return pts.reshape(-1, 1)
    if dimensions == 2:
        return np.loadtxt(filename, comments="#", skiprows=3, usecols=[1, 2])
    return np.loadtxt(filename, comments="#", skiprows=3, usecols=[1, 2, 3])


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

def write_inputs_file(
    grid: dict,
    dimensions: int,
    order_scheme: int,
    CFL: float,
    stress_update_scheme: int,
    output_tag: str,
    constitutive_model: dict,
    enable_temperature: bool,
    particle_filename: str,
    out_filename: str = "Inputs_MPM.inp",
):
    with open(out_filename, "w") as f:
        f.write("# Auto-generated MPM input file\n")       
        
        # ---------------------------------------------------------
        # Geometry
        # ---------------------------------------------------------
        if(dimensions==1):
            write_block(f, [
                ("mpm.prob_lo", f"{grid['xmin']} 0.0 0.0    # Lower corner"),
                ("mpm.prob_hi", f"{grid['xmax']} 0.0 0.0    # Upper corner"),
                ("mpm.ncells", f"{grid['nx']} 0 0"),
                ("mpm.max_grid_size", f"16"),
                ("mpm.is_it_periodic", f"0")
            ], comment="Geometry Parameters")
        elif(dimensions==2):
            write_block(f, [
                ("mpm.prob_lo", f"{grid['xmin']} {grid['ymin']} 0.0    # Lower corner"),
                ("mpm.prob_hi", f"{grid['xmax']} {grid['ymax']} 0.0    # Upper corner"),
                ("mpm.ncells", f"{grid['nx']} {grid['ny']} 0"),
                ("mpm.max_grid_size", f"16"),
                ("mpm.is_it_periodic", f"0 0")
            ], comment="Geometry Parameters")
        else:
            write_block(f, [
                ("mpm.prob_lo", f"{grid['xmin']} {grid['ymin']} {grid['zmin']}    # Lower corner"),
                ("mpm.prob_hi", f"{grid['xmax']} {grid['ymax']} {grid['zmax']}    # Upper corner"),
                ("mpm.ncells", f"{grid['nx']} {grid['ny']} {grid['nz']}"),
                ("mpm.max_grid_size", f"{grid['nx'] + 1}"),
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
            ("mpm.particle_file", f"\"{particle_filename}\"")
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
            ("mpm.final_time", "3.5"),
            ("mpm.max_steps", "5000000"),
            ("mpm.screen_output_time", "0.0001"),
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
            ("mpm.bc_lower", "0 0 0"),
            ("mpm.bc_upper", "0 0 0"),
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
            ("mpm.print_diagnostics", "1"),
            ("mpm.do_calculate_tke_tse", "1"),
            ("mpm.do_calculate_mwa_velcomp", "1"),
            ("mpm.do_calculate_mwa_velmag", "0"),
            ("mpm.do_calculate_minmaxpos", "1"),
            ("mpm.write_diag_output_time", "0.01")
        ], comment="Diagnostics Parameters")
    print(f"WROTE: {out_filename}")


def make_auto_tag_from_cfg(cfg: dict) -> str:
    dims = cfg["dimensions"]
    grid = cfg["grid"]
    ppc = cfg["ppc"]
    cm_type = cfg["constitutive_model"]["type"]
    temp_enabled = cfg["temperature"]["enabled"]
    ord_scheme = cfg["order_scheme"]
    sus_scheme = cfg["stress_update_scheme"]

    desc = (
        f"{dims}D_"
        f"nx{grid['nx']}"
        + (f"_ny{grid['ny']}" if dims >= 2 else "")
        + (f"_nz{grid['nz']}" if dims == 3 else "")
        + f"_ppc{'x'.join(str(x) for x in ppc)}_"
        f"cm{cm_type}_"
        f"T{int(temp_enabled)}_"
        f"ord{ord_scheme}_"
        f"sus{sus_scheme}"
    )
    short_hash = hashlib.md5(desc.encode()).hexdigest()[:6]
    return f"{desc}_{short_hash}"

def write_particles_ascii_streaming(filename, chunk_iter, dimensions):
    """
    Streaming ASCII writer with:
      - header (dim, number_of_material_points)
      - fixed core fields
      - model-specific extra fields (elastic, fluid, etc.)
      - optional temperature fields
      - nothing written for unused blocks (not even in header)
    """

    import numpy as np

    # -------------------------------
    # Open file + placeholder header
    # -------------------------------
    f = open(filename, "w")
    f.write(f"dim: {dimensions}\n")
    f.write("number_of_material_points: 0\n")   # patched later

    # -------------------------------
    # First chunk
    # -------------------------------
    try:
        first_chunk = next(chunk_iter)
    except StopIteration:
        raise RuntimeError("No particle chunks were generated")

    # -------------------------------
    # Core fields (by dimension)
    # -------------------------------
    if dimensions == 1:
        core_fields = ["phase", "x", "radius", "density", "vx", "cm_id"]
    elif dimensions == 2:
        core_fields = ["phase", "x", "y", "radius", "density", "vx", "vy", "cm_id"]
    else:  # 3D
        core_fields = ["phase", "x", "y", "z", "radius", "density", "vx", "vy", "vz", "cm_id"]

    # -------------------------------
    # Constitutive model fields
    # -------------------------------
    # Convention:
    #   cm_id = 0 -> elastic: E, nu
    #   cm_id = 1 -> fluid: Bulk_modulus, Gamma_pressure, Dynamic_viscosity
    # Future models: just extend this dict.
    CM_FIELDS = {
        0: ["E", "nu"],
        1: ["Bulk_modulus", "Gamma_pressure", "Dynamic_viscosity"],
    }

    # Assume single cm_id in this file (typical for a run)
    cm_ids_in_first = np.unique(first_chunk["cm_id"])
    if len(cm_ids_in_first) == 1:
        cm_id0 = int(cm_ids_in_first[0])
    else:
        # Mixed models: you can refine this later if needed
        cm_id0 = int(cm_ids_in_first[0])

    model_fields = CM_FIELDS.get(cm_id0, [])

    # Only keep fields that actually exist in the chunk
    model_fields = [k for k in model_fields if k in first_chunk]

    # -------------------------------
    # Temperature fields (optional)
    # -------------------------------
    temp_fields = []
    if "T" in first_chunk:
        temp_fields = ["T", "spheat", "thermcond", "heatsrc"]
        temp_fields = [k for k in temp_fields if k in first_chunk]

    # -------------------------------
    # Final column order
    # -------------------------------
    colnames = core_fields + model_fields + temp_fields

    # Write column header
    f.write("# " + " ".join(colnames) + "\n")

    # -------------------------------
    # Helper to write one chunk
    # -------------------------------
    def write_chunk(chunk):
        n = len(chunk["x"])
        for i in range(n):
            row = []
            for k in colnames:
                v = chunk[k][i]
                if isinstance(v, (float, np.floating)):
                    row.append(f"{v:.6e}")
                else:
                    row.append(str(v))
            f.write("\t".join(row) + "\n")
        return n

    # First + remaining chunks
    total = write_chunk(first_chunk)
    for chunk in chunk_iter:
        total += write_chunk(chunk)

    f.close()

    # -------------------------------
    # Patch particle count in header
    # -------------------------------
    with open(filename, "r+") as f2:
        lines = f2.readlines()
        lines[1] = f"number_of_material_points: {total}\n"
        f2.seek(0)
        f2.writelines(lines)

    return total




def write_particles_hdf5_streaming(filename, chunk_iter, dimensions):
    """
    filename: final HDF5 file to write
    chunk_iter: an iterator yielding particle chunks, each chunk is a dict of arrays
    dimensions: 1, 2, or 3
    """

    # Open file
    h5 = h5py.File(filename, "w")

    # Metadata
    h5["dim"] = dimensions
    h5["number_of_material_points"] = 0  # will update later

    # Determine dataset names from the first chunk
    first_chunk = next(chunk_iter)
    keys = list(first_chunk.keys())

    # Create extendable datasets
    dsets = {}
    for k in keys:
        dsets[k] = h5.create_dataset(
            k,
            shape=(0,),
            maxshape=(None,),
            dtype=first_chunk[k].dtype,
            chunks=True
        )

    # Write first chunk
    total = 0
    n = len(first_chunk[keys[0]])
    for k in keys:
        dsets[k].resize((total + n,))
        dsets[k][total:total+n] = first_chunk[k]
    total += n

    # Write remaining chunks
    for chunk in chunk_iter:
        n = len(chunk[keys[0]])
        for k in keys:
            dsets[k].resize((total + n,))
            dsets[k][total:total+n] = chunk[k]
        total += n

    # Update metadata
    h5["number_of_material_points"][...] = total
    h5.close()

    return total


def write_particles_hdf5(filename, particles):
    import h5py
    import numpy as np

    with h5py.File(filename, "w") as h5:
        h5["dim"] = 2
        h5["number_of_material_points"] = len(particles["x"])

        for key, arr in particles.items():
            h5.create_dataset(key, data=np.asarray(arr))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_cli()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    dimensions = cfg["dimensions"]
    grid = cfg["grid"]
    ppc = tuple(cfg["ppc"])

    order_scheme = cfg["order_scheme"]
    stress_update_scheme = cfg["stress_update_scheme"]
    output_tag = cfg.get("output_tag", "").strip()
    input_filename = cfg["input_filename"]
    matpt_filename = cfg["materialpoint_filename"]
    plot_to_check = cfg["plot_to_check"]
    CFL = cfg["CFL"]

    # user choice: "ascii" or "hdf5"
    output_format = cfg.get("output_format", "hdf5").lower()

    bodies = cfg.get("bodies")
    if bodies is None:
        bodies = [{
            "shape": cfg["shape"],
            "constitutive_model": cfg["constitutive_model"],
            "initial_velocity": cfg["initial_velocity"],
            "temperature": cfg["temperature"],
        }]

    # ---------------------------------------------------------
    # Build a merged chunk iterator over ALL bodies
    # ---------------------------------------------------------
    def merged_chunk_iter():
        for body in bodies:
            shape_cfg = body["shape"]
            cm_cfg = body["constitutive_model"]
            vel_cfg = body["initial_velocity"]
            temp_cfg = body["temperature"]
            enable_temperature = temp_cfg.get("enabled", False)

            # Velocity function
            if vel_cfg["type"] == "uniform":
                vx0 = vel_cfg.get("vx", 0.0)
                vy0 = vel_cfg.get("vy", 0.0)
                vz0 = vel_cfg.get("vz", 0.0)

                def velocity_function(x, y, z):
                    return vx0, vy0, vz0
            else:
                spec = importlib.util.spec_from_file_location("user_vel", vel_cfg["module"])
                user_vel = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(user_vel)
                velocity_function = getattr(user_vel, vel_cfg["function"])

            # Temperature function
            if not enable_temperature:
                def temperature_function(x, y, z):
                    return 0.0, 0.0, 0.0, 0.0
            elif temp_cfg["type"] == "uniform":
                T0 = temp_cfg["T"]
                sp0 = temp_cfg["spheat"]
                k0 = temp_cfg["thermcond"]
                q0 = temp_cfg["heatsrc"]

                def temperature_function(x, y, z):
                    return T0, sp0, k0, q0
            else:
                spec = importlib.util.spec_from_file_location("user_temp", temp_cfg["module"])
                user_temp = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(user_temp)
                temperature_function = getattr(user_temp, temp_cfg["function"])

            # Stream chunks for this body
            chunk_gen = generate_particle_chunks(
                dimensions=dimensions,
                grid=grid,
                ppc=ppc,
                constitutive_model=cm_cfg,
                enable_temperature=enable_temperature,
                shape_cfg=shape_cfg,
                velocity_function=velocity_function,
                temperature_function=temperature_function,
                # you can tune these if needed:
                cell_block=(32, 32, 8),
                chunk_size=200_000,
            )

            for chunk in chunk_gen:
                yield chunk

    # ---------------------------------------------------------
    # Write final particle file (single merged file)
    # ---------------------------------------------------------
    if output_format == "hdf5":
        particle_file = matpt_filename
        write_particles_hdf5_streaming(
            particle_file,
            merged_chunk_iter(),
            dimensions,
        )
    elif output_format == "ascii":
        particle_file = matpt_filename
        write_particles_ascii_streaming(
            particle_file,
            merged_chunk_iter(),
            dimensions,
        )
    else:
        raise ValueError(f"Unknown output_format: {output_format}")

    # ---------------------------------------------------------
    # Write input file
    # ---------------------------------------------------------
    write_inputs_file(
        grid=grid,
        dimensions=dimensions,
        order_scheme=order_scheme,
        CFL = CFL,
        stress_update_scheme=stress_update_scheme,
        output_tag=output_tag,
        constitutive_model=bodies[0]["constitutive_model"],
        enable_temperature=bodies[0]["temperature"]["enabled"],
        particle_filename=particle_file,
        out_filename=input_filename,
    )

    print("Multi-body preprocessing complete.")

    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ ⚠️  IMPORTANT: Review the input file before proceeding                   │
    │                                                                         │
    │   Default values may have been applied for multiple input parameters    │
    │                                                                         │
    │   Make sure the configuration matches your test case.                   │
    └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    

    if(plot_to_check):
        if not os.path.exists(matpt_filename):
            print("Material point file not found.")
            return
        if not os.path.exists(input_filename):
            print("Input file not found.")
            return
    
        grid = read_grid_from_input(input_filename)
        if(output_format=="hdf5"):
            dim, x, y, z = read_particles_h5(matpt_filename)
        else:
            dim, data = read_particles_ascii(matpt_filename)
            x = data["x"]
            y = data.get("y")   # None in 1D
            z = data.get("z")   # None in 1D/2D
    
            
    
        if dim == 1:
            plot_1d(x, grid)
        elif dim == 2:
            plot_2d(x, y, grid)
        else:
            # 3D: ask user for slice axis
            axis = input("Slice axis (x/y/z)? ").strip().lower()
            if axis not in ["x", "y", "z"]:
                axis = "z"
            plot_3d_slice(x, y, z, grid, slice_axis=axis)

if __name__ == "__main__":
    main()
