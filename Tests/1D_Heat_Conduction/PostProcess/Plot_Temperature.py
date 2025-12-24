import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Exact solution function (section 10.3.3 Ngyuen's book)
# ---------------------------------------------------------
def Texact(x_array, T0, T1, t, N=100):
    x_array = np.asarray(x_array)
    base = T0 + (T1 - T0) * x_array
    series = np.zeros_like(x_array)

    for n in range(1, N + 1):
        coeff = ((-1)**n) / (n * np.pi)
        decay = np.exp(-(n * np.pi)**2 * t)
        series += coeff * decay * np.sin(n * np.pi * x_array)

    T_exact = base + 2 * (T1 - T0) * series
    return T_exact

def main():
    parser = argparse.ArgumentParser(description="Plot temperature vs x and compute errors.")
    parser.add_argument("time", type=float, help="Time value to search for (e.g., 0.020000)")
    parser.add_argument("--T0", type=float, default=0.0, help="Boundary temperature at x=0")
    parser.add_argument("--T1", type=float, default=1.0, help="Boundary temperature at x=1")
    args = parser.parse_args()

    time_str = f"{args.time:.6f}"
    folder = "Solution/ascii_files"
    pattern = os.path.join(folder, f"matpnt_t{time_str}")

    matches = glob.glob(pattern)
    if not matches:
        print(f"No file found for time {time_str} in {folder}")
        return

    filename = matches[0]
    print(f"Reading file: {filename}")

    # Load data
    data = np.loadtxt(filename,skiprows=5)
    x = data[:, 0]
    T_num = data[:, 49]

    # Compute exact solution
    T_ex = Texact(x, args.T0, args.T1, args.time)

    # Compute errors
    abs_err = np.abs(T_num - T_ex)
    min_err = np.min(abs_err)
    max_err = np.max(abs_err)
    rms_err = np.sqrt(np.mean(abs_err**2))

    print("\n=== Error Metrics ===")
    print(f"Min absolute error : {min_err:.6e}")
    print(f"Max absolute error : {max_err:.6e}")
    print(f"RMS error          : {rms_err:.6e}")

    # Plot numerical vs exact
    plt.figure(figsize=(8, 6))
    plt.scatter(x, T_num, label="ExaGOOP", linewidth=2,color='black')
    plt.plot(x, T_ex, "--", label="Exact", linewidth=2,color='red')
    plt.xlabel("x",fontsize=15)
    plt.ylabel("Temperature",fontsize=15)
    plt.title(f"Temperature vs X at t = {time_str}")
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
