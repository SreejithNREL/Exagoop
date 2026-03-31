#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Read grid from input file
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Plot 2D particles + grid
# ------------------------------------------------------------
def plot_2d(x, y, grid):
    fig, ax = plt.subplots(figsize=(6, 6))
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
# Main
# ------------------------------------------------------------
def main():
    import os

    h5file = input("Enter HDF5 particle file: ").strip()
    inpfile = input("Enter input file: ").strip()

    if not os.path.exists(h5file):
        print("HDF5 file not found.")
        return
    if not os.path.exists(inpfile):
        print("Input file not found.")
        return

    grid = read_grid_from_input(inpfile)
    dim, x, y, z = read_particles_h5(h5file)

    if dim == 1:
        print("1D plotting not implemented.")
        return

    if dim == 2:
        plot_2d(x, y, grid)
    else:
        # 3D: ask user for slice axis
        axis = input("Slice axis (x/y/z)? ").strip().lower()
        if axis not in ["x", "y", "z"]:
            axis = "z"
        plot_3d_slice(x, y, z, grid, slice_axis=axis)


if __name__ == "__main__":
    main()

