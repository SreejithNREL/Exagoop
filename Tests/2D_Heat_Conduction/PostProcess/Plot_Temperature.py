import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Exact 2D solution (your function, unchanged)
# ---------------------------------------------------------
def Texact_2D(x, y, T0, T1, L, H, t, N_terms):
    x = np.asarray(x)
    y = np.asarray(y)
    T_exact = np.full_like(x, T1, dtype=np.float64)
    factor = 16 * (T0 - T1) / (np.pi**2)

    for i in range(1, 2*N_terms, 2):  # odd i
        for j in range(1, 2*N_terms, 2):  # odd j
            decay = np.exp(-np.pi**2 * (i**2 / L**2 + j**2 / H**2) * t)
            term = decay / (i * j) * np.sin(i * np.pi * x / L) * np.sin(j * np.pi * y / H)
            T_exact += factor * term

    return T_exact


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="2D heat conduction: numeric vs exact vs error.")
    parser.add_argument("time", type=float, help="Time value (e.g., 0.050000)")
    parser.add_argument("--folder", type=str, default="Solution/ascii_files",
                        help="Folder containing matpnt files")
    parser.add_argument("--skiprows", type=int, default=5,
                        help="Number of metadata rows to skip")
    parser.add_argument("--T0", type=float, default=0.0)
    parser.add_argument("--T1", type=float, default=1.0)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--N_terms", type=int, default=25)
    parser.add_argument("--Tcol", type=int, default=48,
                        help="Column index for temperature")
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
    print(f"Reading: {filename}")

    # ---------------------------------------------------------
    # Load particle data
    # ---------------------------------------------------------
    data = np.loadtxt(filename, skiprows=args.skiprows)
    x = data[:, 0]
    y = data[:, 1]
    T_num = data[:, args.Tcol]

    # ---------------------------------------------------------
    # Compute exact solution
    # ---------------------------------------------------------
    T_ex = Texact_2D(x, y, args.T0, args.T1, args.L, args.H, args.time, args.N_terms)

    # ---------------------------------------------------------
    # Compute absolute error
    # ---------------------------------------------------------
    abs_err = np.abs(T_num - T_ex)

    min_err = np.min(abs_err)
    max_err = np.max(abs_err)
    rms_err = np.sqrt(np.mean(abs_err**2))

    print("\n=== Error Metrics ===")
    print(f"Min absolute error : {min_err:.6e}")
    print(f"Max absolute error : {max_err:.6e}")
    print(f"RMS error          : {rms_err:.6e}")

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Numerical
    c1 = axes[0].tricontourf(x, y, T_num, levels=100, cmap='coolwarm')
    axes[0].set_title('Numerical Temperature')
    fig.colorbar(c1, ax=axes[0])

    # Exact
    c2 = axes[1].tricontourf(x, y, T_ex, levels=100, cmap='coolwarm')
    axes[1].set_title('Exact Temperature')
    fig.colorbar(c2, ax=axes[1])

    # Absolute Error
    c3 = axes[2].tricontourf(x, y, abs_err, levels=100, cmap='inferno')
    axes[2].set_title('Absolute Error |T_num - T_exact|')
    fig.colorbar(c3, ax=axes[2])

    plt.suptitle(f"2D Heat Conduction at t = {args.time:.6f}", fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()

