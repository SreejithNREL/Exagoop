"""
Plot_Temperature.py — 2D Convective Heat Transfer test case
"""
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

AMREX_SPACEDIM = 2
USE_TEMP       = True

CXX_ENUM = {
    "radius": 0, "xvel": 1, "yvel": 2, "zvel": 3,
    "xvel_prime": 4, "yvel_prime": 5, "zvel_prime": 6,
    "strainrate": 7, "strain": 13, "stress": 19,
    "deformation_gradient": 25, "volume": 34, "mass": 35,
    "density": 36, "jacobian": 37, "pressure": 38,
    "vol_init": 39, "E": 40, "nu": 41, "Bulk_modulus": 42,
    "Gama_pressure": 43, "Dynamic_viscosity": 44, "yacceleration": 45,
    "temperature": 46, "specific_heat": 47, "thermal_conductivity": 48,
    "heat_flux": 49, "heat_source": 52,
}

def build_field_dict(dim, use_temp):
    fields = {f"pos{'xyz'[d]}": d for d in range(dim)}
    for name, idx in CXX_ENUM.items():
        if not use_temp and name in (
            "temperature","specific_heat","thermal_conductivity",
            "heat_flux","heat_source"):
            continue
        fields[name] = idx + dim
    return fields


def T_ss_conv(x, T_wall=1.0, h=2.0, k=1.0, L=1.0, T_inf=0.0):
    """Steady-state solution: T_ss satisfies T(0)=T_wall, -k*T'(L)=h*(T(L)-T_inf)."""
    return T_wall + (T_inf - T_wall) * (h/k) * x / (1.0 + h*L/k)


def find_eigenvalues(h_over_k, L=1.0, N=50):
    """First N roots of lam*cos(lam*L) + (h/k)*sin(lam*L) = 0."""
    def f(lam):
        return lam * np.cos(lam * L) + h_over_k * np.sin(lam * L)
    lam_scan = np.linspace(1e-6, (N + 2) * np.pi / L, 100 * (N + 2))
    f_scan   = f(lam_scan)
    lambdas  = []
    for i in range(len(f_scan) - 1):
        if f_scan[i] * f_scan[i + 1] < 0:
            try:
                lambdas.append(brentq(f, lam_scan[i], lam_scan[i + 1], xtol=1e-12))
            except ValueError:
                pass
        if len(lambdas) == N:
            break
    return np.array(lambdas)


def T_exact_convective(x, t, T_wall=1.0, h=2.0, k=1.0,
                        T_inf=0.0, L=1.0, N=50):
    """
    Exact solution for:
      T(0,t) = T_wall  (Dirichlet)
      -k*dT/dx(L,t) = h*(T(L,t) - T_inf)  (Robin)
      T(x,0) = 0

    Decompose: T = T_ss(x) + w(x,t)
      T_ss = T_wall + (T_inf - T_wall)*(h/k)*x / (1 + h*L/k)
      w satisfies heat eq with homogeneous BCs and w(x,0) = -T_ss(x)
    """
    from scipy.integrate import quad
    x    = np.asarray(x, dtype=float)
    Tss  = T_ss_conv(x, T_wall=T_wall, h=h, k=k, L=L, T_inf=T_inf)
    lams = find_eigenvalues(h / k, L=L, N=N)
    T    = Tss.copy()
    for lam in lams:
        norm2 = L / 2.0 - np.sin(2.0 * lam * L) / (4.0 * lam)
        # w(x,0) = -T_ss(x)
        num, _ = quad(lambda xi: -T_ss_conv(xi, T_wall, h, k, L, T_inf)
                      * np.sin(lam * xi), 0, L)
        C_n = num / norm2
        T  += C_n * np.sin(lam * x) * np.exp(-lam**2 * t)
    return T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time",      type=float, default=0.5)
    parser.add_argument("--T_wall",    type=float, default=1.0)
    parser.add_argument("--h",         type=float, default=2.0)
    parser.add_argument("--k",         type=float, default=1.0)
    parser.add_argument("--T_inf",     type=float, default=0.0)
    parser.add_argument("--L",         type=float, default=1.0)
    parser.add_argument("--fileloc",   type=str,   default=".")
    parser.add_argument("--outputpic", type=str,
                        default="Temperature_Convective.png")
    args = parser.parse_args()

    time_str = f"{args.time:.6f}"
    pattern  = os.path.join(args.fileloc, f"matpnt_t{time_str}")
    matches  = glob.glob(pattern)
    if not matches:
        print(f"No file found: {pattern}")
        return

    data   = np.loadtxt(matches[0], skiprows=5)
    fields = build_field_dict(AMREX_SPACEDIM, USE_TEMP)
    T_idx  = fields["temperature"]
    x      = data[:, 0]
    T_num  = data[:, T_idx]

    order    = np.argsort(x)
    x_s, T_s = x[order], T_num[order]
    T_ex     = T_exact_convective(x_s, args.time, T_wall=args.T_wall,
                                   h=args.h, k=args.k, T_inf=args.T_inf,
                                   L=args.L)

    if np.any(np.isnan(T_s)):
        print(f"WARNING: {np.sum(np.isnan(T_s))} NaN values in ExaGOOP output")
        T_s = np.nan_to_num(T_s, nan=0.0)
    rms = np.sqrt(np.mean((T_s - T_ex)**2))
    print(f"RMS error: {rms:.4e}")

    plt.figure(figsize=(8, 6))
    plt.scatter(x_s, T_s,  label="ExaGOOP", color="black", s=10)
    plt.plot(x_s,    T_ex, label=f"Exact (h={args.h}, T_inf={args.T_inf})",
             color="red", linestyle="--", linewidth=2)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("Temperature", fontsize=14)
    plt.title(f"2D Convective HT — T vs x at t={args.time:.4f}"
              f"\nRMS={rms:.2e}")
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.outputpic)
    print(f"Saved: {args.outputpic}")


if __name__ == "__main__":
    main()
