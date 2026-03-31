import argparse
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Extract time from filename "matpnt_t0.030000"
# ---------------------------------------------------------
def extract_time(fname):
    match = re.search(r"matpnt_t([0-9.]+)", fname)
    return float(match.group(1)) if match else None

# ---------------------------------------------------------
# Exact velocity for axial vibration
# v(x,t) = V0 * sin(pi x / (2L)) * cos(omega1 t)
# ---------------------------------------------------------
def exact_velocity(x, t, V0=0.1, L=25.0, E=100, rho=1):
    omega1 = (np.pi / (2 * L)) * np.sqrt(E / rho)
    return V0 * np.sin(np.pi * x / (2 * L)) * np.cos(omega1 * t)

# ---------------------------------------------------------
# Plot numerical vs exact velocity
# ---------------------------------------------------------
def plot_overlay(x, vx_num, vx_exact, t):
    plt.figure(figsize=(7, 4))
    plt.scatter(x, vx_num, s=20, color="blue", label="Numerical (MPM)")
    plt.plot(x, vx_exact, "r-", linewidth=2, label="Exact")

    plt.xlabel("x")
    plt.ylabel("Velocity")
    plt.title(f"Velocity Comparison at t = {t:.4f} s")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute min/max/RMS error for axial vibration velocity and plot overlay.")
    parser.add_argument("--time", type=float, help="Time value (e.g., 0.030000)")
    parser.add_argument("--folder", type=str, default="Solution/ascii_files",
                        help="Folder containing matpnt_t files")
    parser.add_argument("--skiprows", type=int, default=5,
                        help="Number of metadata rows to skip")
    parser.add_argument("--dim", type=int, default=1,
                        help="Dimension")
    parser.add_argument("--showplot", type=bool, default=False,
                        help=" show plot?")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Find the correct file
    # ---------------------------------------------------------
    time_str = f"{args.time:.6f}"
    pattern = os.path.join(args.folder, f"matpnt_t{time_str}")
    matches = glob.glob(pattern)

    if not matches:
        print(f"No file found matching: {pattern}")
        return

    filename = matches[0]
    print(f"Reading file: {filename}")

    # ---------------------------------------------------------
    # Load numerical data
    # ---------------------------------------------------------
    data = np.loadtxt(filename, skiprows=args.skiprows)
    x = data[:, 0]
    vx_col = 2
    if(args.dim==1):
        vx_col = 3
    elif(args.dim==2):
        vx_col = 3
    elif(args.dim==3):
        vx_col = 4
        
    print(args.dim, vx_col)
        
        
        
    vx_num = data[:, vx_col]

    # ---------------------------------------------------------
    # Compute exact velocity
    # ---------------------------------------------------------
    vx_exact = exact_velocity(x, args.time)

    # ---------------------------------------------------------
    # Compute errors
    # ---------------------------------------------------------
    abs_err = np.abs(vx_num - vx_exact)
    min_err = np.min(abs_err)
    max_err = np.max(abs_err)
    rms_err = np.sqrt(np.mean(abs_err**2))

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------
    print("\n=== Velocity Error Metrics ===")
    print(f"Time: {args.time:.6f}")
    print(f"Min absolute error : {min_err:.6e}")
    print(f"Max absolute error : {max_err:.6e}")
    print(f"RMS error          : {rms_err:.6e}")

    # ---------------------------------------------------------
    # Plot overlay
    # ---------------------------------------------------------
    if(args.showplot):
        plot_overlay(x, vx_num, vx_exact, args.time)


if __name__ == "__main__":
    main()
