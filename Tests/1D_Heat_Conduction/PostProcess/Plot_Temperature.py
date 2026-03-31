import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import string


USE_TEMP = True
AMREX_SPACEDIM=1

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

    # Prepend pos fields
    fields = {}
    for d in range(AMREX_SPACEDIM):
        fields[f"pos{'xyz'[d]}"] = d

    # Shift all enum indices by AMREX_SPACEDIM
    for name, enum_idx in CXX_ENUM.items():
        if not USE_TEMP and name in (
            "temperature", "specific_heat",
            "thermal_conductivity", "heat_flux", "heat_source"
        ):
            continue
        fields[name] = enum_idx + AMREX_SPACEDIM

    return fields

def logical_index(name, fields):
    if name not in fields:
        raise KeyError(f"{name} not in dictionary")
    return fields[name]


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
    parser.add_argument("--time", type=float, default=0.0, help="Time value to search for (e.g., 0.020000)")
    parser.add_argument("--T0", type=float, default=0.0, help="Boundary temperature at x=0")
    parser.add_argument("--dim", type=int, default=3, help="Temperature col index in the ascii file")
    parser.add_argument("--T1", type=float, default=1.0, help="Boundary temperature at x=1")
    parser.add_argument("--fileloc", type=str, default=1.0, help="Boundary temperature at x=1")
    parser.add_argument("--outputpic", type=str, default=1.0, help="Boundary temperature at x=1")
    args = parser.parse_args()

    time_str = f"{args.time:.6f}"
    folder = args.fileloc
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
    fields = build_field_dict(AMREX_SPACEDIM, True)
    print(fields["temperature"])
    temperature_idx = fields["temperature"]
    
    T_num = data[:, temperature_idx]   
    
    order = np.argsort(x)
    x_sorted = x[order]
    T_sorted = T_num[order]


    # Compute exact solution
    T_ex = Texact(x_sorted, args.T0, args.T1, args.time)

    # Compute errors
    abs_err = np.abs(T_sorted - T_ex)
    min_err = np.min(abs_err)
    max_err = np.max(abs_err)
    rms_err = np.sqrt(np.mean(abs_err**2))

    print("\n=== Error Metrics ===")
    print(f"Min absolute error : {min_err:.6e}")
    print(f"Max absolute error : {max_err:.6e}")
    print(f"RMS error          : {rms_err:.6e}")

    # Plot numerical vs exact
    plt.figure(figsize=(8, 6))
    plt.scatter(x_sorted, T_sorted, label="ExaGOOP", linewidth=2,color='black')
    plt.plot(x_sorted, T_ex, "--", label="Exact", linewidth=2,color='red')
    plt.xlabel("x",fontsize=15)
    plt.ylabel("Temperature",fontsize=15)
    plt.title(f"Temperature vs X at t = {time_str}")
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(args.outputpic)

if __name__ == "__main__":
    main()
