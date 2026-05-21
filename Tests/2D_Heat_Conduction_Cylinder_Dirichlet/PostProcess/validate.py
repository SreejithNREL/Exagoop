import sys
import glob
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

USE_TEMP = True
AMREX_SPACEDIM = 2

CXX_ENUM = {
    "radius": 0,
    "xvel": 1,
    "yvel": 2,
    "zvel": 3,
    "xvel_prime": 4,
    "yvel_prime": 5,
    "zvel_prime": 6,
    "strainrate": 7,
    "strain": 13,
    "stress": 19,
    "deformation_gradient": 25,
    "volume": 34,
    "mass": 35,
    "density": 36,
    "jacobian": 37,
    "pressure": 38,
    "vol_init": 39,
    "E": 40,
    "nu": 41,
    "Bulk_modulus": 42,
    "Gama_pressure": 43,
    "Dynamic_viscosity": 44,
    "yacceleration": 45,
    "temperature": 46,
    "specific_heat": 47,
    "thermal_conductivity": 48,
    "heat_flux": 49,
    "heat_source": 52,
}

def build_field_dict(AMREX_SPACEDIM: int, USE_TEMP: bool):
    if AMREX_SPACEDIM not in (1, 2, 3):
        raise ValueError("AMREX_SPACEDIM must be 1, 2, or 3")
    fields = {}
    for d in range(AMREX_SPACEDIM):
        fields[f"pos{'xyz'[d]}"] = d
    for name, enum_idx in CXX_ENUM.items():
        if not USE_TEMP and name in (
            "temperature", "specific_heat",
            "thermal_conductivity", "heat_flux", "heat_source"
        ):
            continue
        fields[name] = enum_idx + AMREX_SPACEDIM
    return fields


def solve_laplace_fd(N, R_inner, cx, cy, T_inner, T_outer):
    """
    Solve ∇²T = 0 on [0,1]² with:
      T = T_inner  where r <= R_inner  (cylinder, Dirichlet)
      T = T_outer  on the four edges   (square boundary)
    """
    n_nodes = (N + 1) ** 2

    def idx(i, j):
        return j * (N + 1) + i

    A = lil_matrix((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    xs = np.linspace(0, 1, N + 1)
    ys = np.linspace(0, 1, N + 1)

    for j in range(N + 1):
        for i in range(N + 1):
            k = idx(i, j)
            x, y = xs[i], ys[j]
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            if r <= R_inner:
                A[k, k] = 1.0
                b[k] = T_inner
            elif i == 0 or i == N or j == 0 or j == N:
                A[k, k] = 1.0
                b[k] = T_outer
            else:
                A[k, k] = -4.0
                A[k, idx(i + 1, j)] = 1.0
                A[k, idx(i - 1, j)] = 1.0
                A[k, idx(i, j + 1)] = 1.0
                A[k, idx(i, j - 1)] = 1.0

    T_flat = spsolve(A.tocsr(), b)

    x_all = np.tile(xs, N + 1)
    y_all = np.repeat(ys, N + 1)
    r_all = np.sqrt((x_all - cx) ** 2 + (y_all - cy) ** 2)
    fluid_mask = (r_all > R_inner) & (x_all > 0) & (x_all < 1) & (y_all > 0) & (y_all < 1)

    return x_all[fluid_mask], y_all[fluid_mask], T_flat[fluid_mask]


def interpolate_fd_to_points(x_fd, y_fd, T_fd, x_pts, y_pts):
    """Interpolate scattered FD solution onto arbitrary points."""
    triang = mtri.Triangulation(x_fd, y_fd)
    interp = mtri.LinearTriInterpolator(triang, T_fd)
    return interp(x_pts, y_pts)


# ── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="2D heat conduction: numeric vs exact vs error.")
parser.add_argument("--time",      type=float, required=True,
                    help="Time value (e.g., 0.050000)")
parser.add_argument("--folder",    type=str,   default="Solution/ascii_files",
                    help="Folder containing matpnt files")
parser.add_argument("--outputpic", type=str,   default="Solution/ascii_files/Pics",
                    help="Output folder for saved plots")
parser.add_argument("--skiprows",  type=int,   default=5,
                    help="Number of metadata rows to skip")
parser.add_argument("--T0",        type=float, default=0.0,
                    help="Outer boundary temperature")
parser.add_argument("--T1",        type=float, default=1.0,
                    help="Cylinder surface temperature")
parser.add_argument("--L",         type=float, default=1.0,
                    help="Domain width")
parser.add_argument("--H",         type=float, default=1.0,
                    help="Domain height")
parser.add_argument("--N_terms",   type=int,   default=25,
                    help="Number of Fourier terms (unused for steady-state FD)")
args = parser.parse_args()

# ── Parameters ───────────────────────────────────────────────────────────────
T_INNER  = args.T1
T_OUTER  = args.T0
CX, CY   = 0.5, 0.5
R_INNER  = 0.15
R_MIN    = R_INNER
R_MAX    = 0.5
FD_N     = 400
RMS_TOL  = 1e-1

# Build the time-stamped filename from --time
time_str  = f"{args.time:f}"          # e.g. "0.050000"
filename  = f"matpnt_t{time_str}"     # e.g. "matpnt_t0.050000"

# ── Locate MPM output file ───────────────────────────────────────────────────
matches = glob.glob(os.path.join(args.folder, "*", filename))
if not matches:
    matches = glob.glob(os.path.join(args.folder, filename))
if not matches:
    matches = glob.glob(os.path.join("*", filename))
if not matches:
    print(f"FAIL: {filename} not found under {args.folder}")
    sys.exit(1)

FILEPATH = matches[0]
print(f"Found output file: {FILEPATH}")

# ── Load MPM output ──────────────────────────────────────────────────────────
data = np.loadtxt(FILEPATH, skiprows=args.skiprows)
assert data.shape[1] >= 49, f"Expected at least 49 columns, got {data.shape[1]}"

x_all = data[:, 0]
y_all = data[:, 1]
r_all = np.sqrt((x_all - CX) ** 2 + (y_all - CY) ** 2)
mask  = (r_all >= R_MIN) & (r_all <= R_MAX)

x_pts = x_all[mask]
y_pts = y_all[mask]

# ── Solve FD reference ───────────────────────────────────────────────────────
print(f"Solving FD reference on {FD_N}x{FD_N} grid...")
x_fd, y_fd, T_fd = solve_laplace_fd(FD_N, R_INNER, CX, CY, T_INNER, T_OUTER)
print("FD solve complete.")

# ── Diagnostics ──────────────────────────────────────────────────────────────
print(f"\n--- Diagnostics ---")
print(f"Time value:                   {args.time}")
print(f"Total particles loaded:       {len(x_all)}")
print(f"r range (all particles):      [{r_all.min():.4f}, {r_all.max():.4f}]")
print(f"x range (all particles):      [{x_all.min():.4f}, {x_all.max():.4f}]")
print(f"y range (all particles):      [{y_all.min():.4f}, {y_all.max():.4f}]")
print(f"Particles after radial mask:  {mask.sum()}")
if mask.sum() > 0:
    print(f"x range (masked):             [{x_pts.min():.4f}, {x_pts.max():.4f}]")
    print(f"y range (masked):             [{y_pts.min():.4f}, {y_pts.max():.4f}]")
print(f"FD fluid nodes:               {len(x_fd)}")
print(f"FD x range:                   [{x_fd.min():.4f}, {x_fd.max():.4f}]")
print(f"FD y range:                   [{y_fd.min():.4f}, {y_fd.max():.4f}]")

if mask.sum() < 3:
    print(f"\nFAIL: only {mask.sum()} particles in r∈[{R_MIN},{R_MAX}] from ({CX},{CY}).")
    sys.exit(1)

# ── Interpolate FD onto MPM points ───────────────────────────────────────────
fields   = build_field_dict(AMREX_SPACEDIM, True)
T_idx    = fields["temperature"]
T_num    = data[mask, T_idx]

T_ex_raw = interpolate_fd_to_points(x_fd, y_fd, T_fd, x_pts, y_pts)

T_ex_arr = np.ma.getdata(T_ex_raw)
mask_bad = np.ma.getmaskarray(T_ex_raw)
valid    = ~mask_bad & np.isfinite(T_ex_arr)

print(f"Valid after FD interpolation: {valid.sum()} / {len(valid)}")
print(f"Masked/NaN count:             {(~valid).sum()}")
print(f"-------------------\n")

if valid.sum() < 3:
    print("FAIL: fewer than 3 valid particles after interpolation.")
    sys.exit(1)

T_num = T_num[valid]
T_ex  = T_ex_arr[valid]
x_pts = x_pts[valid]
y_pts = y_pts[valid]

error = T_num - T_ex
rms   = np.sqrt(np.mean(error ** 2))

# ── Contour plots ─────────────────────────────────────────────────────────────
triang_mpm = mtri.Triangulation(x_pts, y_pts)
fig, axes  = plt.subplots(1, 3, figsize=(15, 4.5))
theta      = np.linspace(0, 2 * np.pi, 300)

# Shared colormap limits for numerical and exact panels
#T_vmin = min(T_num.min(), T_ex.min())
#T_vmax = max(T_num.max(), T_ex.max())

T_vmin = 0.0
T_vmax = 1.0

# Symmetric limits for error panel
err_vmax = np.abs(error).max()

datasets = [
    (T_num,  "Numerical Temperature", "RdBu_r",  T_vmin,    T_vmax),
    (T_ex,   "FD Reference (Exact)",  "RdBu_r",  T_vmin,    T_vmax),
    (error,  "Error (Num − FD Ref)",  "coolwarm", -err_vmax, err_vmax),
]

for ax, (values, title, cmap, vmin, vmax) in zip(axes, datasets):
    cf = ax.tricontourf(triang_mpm, values, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.tricontour(triang_mpm, values, levels=20, colors="k", linewidths=0.3, alpha=0.4)
    plt.colorbar(cf, ax=ax, pad=0.02, fraction=0.046)

    for R, ls in [(R_INNER, "-"), (R_MAX, "--")]:
        ax.plot(CX + R * np.cos(theta), CY + R * np.sin(theta),
                "k", ls=ls, lw=0.9, alpha=0.7)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

fig.suptitle(
    f"Steady-state heat conduction  |  t={args.time}  |  RMS error = {rms:.2e}  "
    f"|  r∈[{R_MIN:.2f},{R_MAX:.2f}] from ({CX},{CY})",
    fontsize=11, y=1.02
)
fig.tight_layout()

# ── Save plot ─────────────────────────────────────────────────────────────────
os.makedirs(args.outputpic, exist_ok=True)
outpath = os.path.join(args.outputpic, f"temperature_contours_t{time_str}.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Plot saved to {outpath}")
#plt.show()

print(f"RMS error          : {rms:.6e}")

# ── Pass/Fail ────────────────────────────────────────────────────────────────
if rms < RMS_TOL:
    print(f"PASS: RMS={rms:.2e}")
    sys.exit(0)
else:
    print(f"FAIL: RMS={rms:.2e} exceeds {RMS_TOL:.0e}")
    sys.exit(1)