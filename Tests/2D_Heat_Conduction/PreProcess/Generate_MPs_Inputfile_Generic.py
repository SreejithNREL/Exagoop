#!/usr/bin/env python3
import argparse
import json
import hashlib
import importlib.util
from typing import Tuple, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import h5py
import numpy as np
from typing import Tuple, Callable, Optional, Dict

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def die(msg: str):
    raise SystemExit(f"[ERROR] {msg}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
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
# Shape system (2D + 3D)
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


# ------------------------------------------------------------
# Particle generator
# ------------------------------------------------------------
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



def generate_particles_vectorized(
    dimensions,
    grid,
    ppc,
    constitutive_model,
    enable_temperature,
    shape_cfg,
    velocity_function,
    temperature_function,
    out_particles=None,          # None = multi-body mode
    cell_block=(32, 32, 8),
):
    import numpy as np
    import h5py

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
    # Multi-body mode: collect arrays
    # ------------------------------------------------------------
    if out_particles is None:
        X = []; Y = []; Z = []
        VX = []; VY = []; VZ = []
        R = []; D = []; CMID = []
        CMEXTRA = {k: [] for k in cm_extra.keys()}
        if enable_temperature:
            T = []; SP = []; K = []; Q = []

    # ------------------------------------------------------------
    # HDF5 mode
    # ------------------------------------------------------------
    else:
        h5 = h5py.File(out_particles, "w")
        h5["dim"] = dimensions
        h5["number_of_material_points"] = 0

        def create_dset(name):
            return h5.create_dataset(name, shape=(0,), maxshape=(None,), dtype="f8")

        dsets = {}
        dsets["phase"] = create_dset("phase")
        dsets["x"] = create_dset("x")
        if dimensions >= 2:
            dsets["y"] = create_dset("y")
        if dimensions == 3:
            dsets["z"] = create_dset("z")

        dsets["radius"] = create_dset("radius")
        dsets["density"] = create_dset("density")

        dsets["vx"] = create_dset("vx")
        if dimensions >= 2:
            dsets["vy"] = create_dset("vy")
        if dimensions == 3:
            dsets["vz"] = create_dset("vz")

        dsets["cm_id"] = create_dset("cm_id")
        for k in cm_extra.keys():
            dsets[k] = create_dset(k)

        if enable_temperature:
            for k in ["T", "spheat", "thermcond", "heatsrc"]:
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
    # Vectorized block generator (2D only for now)
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

                if out_particles is None:
                    X.append(px); Y.append(py); Z.append(pz)
                    VX.append(vx); VY.append(vy); VZ.append(vz)
                    R.append(rad); D.append(dens); CMID.append(cm_id)
                    for k, v in cm_extra.items():
                        CMEXTRA[k].append(v)
                    if enable_temperature:
                        T0, SP0, K0, Q0 = temperature_function(px, py, pz)
                        T.append(T0); SP.append(SP0); K.append(K0); Q.append(Q0)

                else:
                    buf["phase"].append(phase)
                    buf["x"].append(px)
                    buf["y"].append(py)
                    buf["radius"].append(rad)
                    buf["density"].append(dens)
                    buf["vx"].append(vx)
                    buf["vy"].append(vy)
                    buf["cm_id"].append(cm_id)
                    for k, v in cm_extra.items():
                        buf[k].append(v)
                    if enable_temperature:
                        T0, SP0, K0, Q0 = temperature_function(px, py, pz)
                        buf["T"].append(T0)
                        buf["spheat"].append(SP0)
                        buf["thermcond"].append(K0)
                        buf["heatsrc"].append(Q0)

    # ------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------
    if out_particles is None:
        result = {
            "x": np.array(X),
            "y": np.array(Y),
            "z": np.array(Z),
            "vx": np.array(VX),
            "vy": np.array(VY),
            "vz": np.array(VZ),
            "radius": np.array(R),
            "density": np.array(D),
            "cm_id": np.array(CMID),
        }
        for k in cm_extra.keys():
            result[k] = np.array(CMEXTRA[k])
        if enable_temperature:
            result["T"] = np.array(T)
            result["spheat"] = np.array(SP)
            result["thermcond"] = np.array(K)
            result["heatsrc"] = np.array(Q)
        return result

    else:
        flush()
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
                ("mpm.max_grid_size", f"{grid['nx'] + 1}"),
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
            ("mpm.particle_file", "\"mpm_particles.h5\"")
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
            ("mpm.write_output_time", "0.01"),
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

    bodies = cfg.get("bodies")
    if bodies is None:
        bodies = [{
            "shape": cfg["shape"],
            "constitutive_model": cfg["constitutive_model"],
            "initial_velocity": cfg["initial_velocity"],
            "temperature": cfg["temperature"],
        }]

    all_particles = []

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
                return 0, 0, 0, 0
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

        # Generate particles for this body
        pts = generate_particles_vectorized(
            dimensions=dimensions,
            grid=grid,
            ppc=ppc,
            constitutive_model=cm_cfg,
            enable_temperature=enable_temperature,
            shape_cfg=shape_cfg,
            velocity_function=velocity_function,
            temperature_function=temperature_function,
            out_particles=None,
        )

        all_particles.append(pts)

    # ---------------------------------------------------------
    # Merge all bodies
    # ---------------------------------------------------------
    merged = {}
    for key in all_particles[0].keys():
        merged[key] = np.concatenate([p[key] for p in all_particles], axis=0)

    # ---------------------------------------------------------
    # Write final particle file
    # ---------------------------------------------------------
    write_particles_hdf5("mpm_particles.h5", merged)

    # ---------------------------------------------------------
    # Write input file
    # ---------------------------------------------------------
    write_inputs_file(
        grid=grid,
        dimensions=dimensions,
        order_scheme=order_scheme,
        stress_update_scheme=stress_update_scheme,
        output_tag=output_tag,
        constitutive_model=bodies[0]["constitutive_model"],
        enable_temperature=bodies[0]["temperature"]["enabled"],
        particle_filename="mpm_particles.h5",
        out_filename=input_filename,
    )

    print("Multi-body preprocessing complete.")


    # pts = load_particle_positions(particle_file, dimensions)
    #
    # if dimensions in [1, 2]:
    #     plot_material_points(pts, grid, dimensions,output_tag)
    # else:
    #     # default slice for 3D: x = mid-plane
    #     slice_axis = "x"
    #     slice_value = 0.5 * (grid["xmin"] + grid["xmax"])
    #     plot_material_points(
    #         pts,
    #         grid,
    #         dimensions,
    #         output_tag,
    #         slice_axis=slice_axis,
    #         slice_value=slice_value,
    #     )

    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ ⚠️  IMPORTANT: Review the input file before proceeding                   │
    │                                                                         │
    │   Default values may have been applied for multiple input parameters    │
    │                                                                         │
    │   Make sure the configuration matches your test case.                   │
    └─────────────────────────────────────────────────────────────────────────┘
    """)



if __name__ == "__main__":
    main()
