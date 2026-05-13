#!/usr/bin/env python3
"""
PostProcess/visualise_twist.py
Reads ExaGOOP ASCII particle output files and produces:
  1. A 3D scatter plot of the column at a chosen time snapshot.
  2. A twist-angle vs z plot comparing the MPM result to the
     analytical solid-body rotation angle  theta(t) = omega * t.
  3. A time-series of max displacement at the top face.

Usage:
    python3 visualise_twist.py --solution_dir ../Solution/particle_files \
                               --output_tag   3D_Twisted_Column \
                               --omega 0.5 \
                               [--snapshot 000040]
"""
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------
def read_ascii_particles(path):
    data = {}
    header_done = False
    colnames = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("dim:") or line.startswith("number_of"):
                continue
            if line.startswith("# "):
                colnames = line[2:].split()
                header_done = True
                for c in colnames:
                    data[c] = []
                continue
            if not header_done:
                continue
            vals = line.split()
            for c, v in zip(colnames, vals):
                data[c].append(float(v))
    return {k: np.array(v) for k, v in data.items()}


def list_snapshots(solution_dir, output_tag):
    pattern = os.path.join(solution_dir,
                           f"../{output_tag}/matpnt_??????")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(solution_dir, "matpnt_??????")
        files = sorted(glob.glob(pattern))
    return files


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------
def compute_twist_angle(x, y, x0, y0):
    """Mean rotation angle of material column cross-section relative to origin."""
    dx = x - x0
    dy = y - y0
    angles = np.arctan2(dy, dx)
    return float(np.mean(angles))


def parse_time_from_filename(fname):
    base = os.path.basename(fname)
    idx  = int(base.replace("matpnt_", ""))
    return idx


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_column_3d(d, title, omega, t):
    fig = plt.figure(figsize=(7, 10))
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(d["x"], d["y"], d["z"],
                     c=d["z"], cmap="plasma", s=2, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="z (m)", shrink=0.6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"{title}\nt = {t:.3f} s   (omega={omega} rad/s)")
    plt.tight_layout()
    return fig


def plot_twist_profile(d, omega, t, nz_bins=20):
    """
    Bin particles by z-layer, compute mean rotation angle in each layer,
    compare to analytical  theta(z, t) = omega * t * (z / L).
    """
    z   = d["z"]
    x   = d["x"]
    y   = d["y"]
    L   = z.max()
    z_edges = np.linspace(z.min(), L, nz_bins + 1)
    z_mid   = 0.5 * (z_edges[:-1] + z_edges[1:])

    angles_mpm  = []
    angles_anal = []
    for i in range(nz_bins):
        mask = (z >= z_edges[i]) & (z < z_edges[i + 1])
        if mask.sum() < 3:
            angles_mpm.append(np.nan)
        else:
            xi = x[mask] - np.mean(x[mask])
            yi = y[mask] - np.mean(y[mask])
            th = np.mean(np.arctan2(yi, xi))
            angles_mpm.append(float(th))
        angles_anal.append(omega * t * z_mid[i] / L)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.degrees(angles_anal), z_mid, "k--", label="Analytical")
    ax.plot(np.degrees(angles_mpm),  z_mid, "ro-",  label="MPM",
            markersize=4)
    ax.set_xlabel("Twist angle (degrees)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"Twist profile at t = {t:.3f} s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_top_displacement(files, omega, dt_per_step):
    """
    For each snapshot compute mean displacement magnitude at z > 0.95*L.
    """
    times = []
    disps = []
    for fpath in files:
        d   = read_ascii_particles(fpath)
        L   = d["z"].max()
        top = d["z"] > 0.95 * L
        if top.sum() == 0:
            continue
        x_top = d["x"][top]
        y_top = d["y"][top]
        disp  = np.sqrt(x_top**2 + y_top**2).mean()
        idx   = parse_time_from_filename(fpath)
        times.append(idx * dt_per_step)
        disps.append(disp)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, disps, "b-o", markersize=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean top-face displacement (m)")
    ax.set_title("Top-face displacement vs time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution_dir", default="../Solution/particle_files",
                    help="Folder containing matpnt_* files")
    ap.add_argument("--output_tag",   default="3D_Twisted_Column")
    ap.add_argument("--omega",        type=float, default=0.5,
                    help="Angular velocity (rad/s) applied at z-hi wall")
    ap.add_argument("--snapshot",     default=None,
                    help="Snapshot index to plot (e.g. 000040). Default: last.")
    ap.add_argument("--write_output_time", type=float, default=0.1,
                    help="Time between output snapshots (s)")
    ap.add_argument("--outdir",       default="./Figures",
                    help="Directory to save figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    base = os.path.join(args.solution_dir,
                        f"../{args.output_tag}")
    pattern = os.path.join(base, "matpnt_??????")
    files   = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(args.solution_dir, "matpnt_??????")
        files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No matpnt_* files found under {args.solution_dir}")

    print(f"Found {len(files)} snapshots.")

    if args.snapshot is None:
        snap_file = files[-1]
    else:
        matches = [f for f in files if args.snapshot in f]
        if not matches:
            raise FileNotFoundError(f"Snapshot {args.snapshot} not found")
        snap_file = matches[0]

    snap_idx = parse_time_from_filename(snap_file)
    t_snap   = snap_idx * args.write_output_time

    print(f"Loading snapshot: {snap_file}  (t = {t_snap:.3f} s)")
    d = read_ascii_particles(snap_file)

    fig1 = plot_column_3d(d, "3D Twisted Column", args.omega, t_snap)
    fig1.savefig(os.path.join(args.outdir, f"column_3d_{snap_idx:06d}.png"),
                 dpi=150)
    print("Saved: column_3d_*.png")

    fig2 = plot_twist_profile(d, args.omega, t_snap)
    fig2.savefig(os.path.join(args.outdir, f"twist_profile_{snap_idx:06d}.png"),
                 dpi=150)
    print("Saved: twist_profile_*.png")

    fig3 = plot_top_displacement(files, args.omega, args.write_output_time)
    fig3.savefig(os.path.join(args.outdir, "top_displacement_vs_time.png"),
                 dpi=150)
    print("Saved: top_displacement_vs_time.png")

    plt.show()


if __name__ == "__main__":
    main()
