"""
Plot_Temperature.py — 2D Heat Flux test case
"""
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

AMREX_SPACEDIM = 2
USE_TEMP = True

# Column map for ExaGOOP ASCII particle dumps: shared helper, kept in sync
# with Source/mpm_specs.H (was a hard-coded per-script copy before).
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "..", "..", "..", "Tools", "PostProcess"))
from exagoop_columns import build_field_dict


def T_exact_flux(x, t, q=1.0, k=1.0, L=1.0, N=300):
    """
    Exact solution for 1D slab:
      T(0,t)        = 0      (Dirichlet)
      dT/dx(L,t)    = q/k    (Neumann / prescribed flux)
      T(x,0)        = 0      (IC)

    Derivation:
      Steady state: T_s(x) = (q/k)*x
      Transient v = T - T_s satisfies:
        dv/dt = alpha*d^2v/dx^2,  v(0,t)=0,  dv/dx(L,t)=0,  v(x,0)=-T_s
      Eigenfunctions: sin(lambda_n*x)
        lambda_n = (2n-1)*pi/(2*L)   [Dirichlet at 0, Neumann at L]
      Coefficients (analytic, using cos(lambda_n*L)=0, sin(lambda_n*L)=(-1)^(n+1)):
        C_n = 2*(-1)^n / (L * lambda_n^2)
    """
    x = np.asarray(x, dtype=float)
    T = (q / k) * x
    for n in range(1, N + 1):
        lam = (2 * n - 1) * np.pi / (2 * L)
        Cn  = 2.0 * (-1)**n / (L * lam**2)
        T  += Cn * np.sin(lam * x) * np.exp(-lam**2 * t)
    return T



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time",      type=float, default=0.05)
    parser.add_argument("--q",         type=float, default=1.0,  help="Heat flux")
    parser.add_argument("--k",         type=float, default=1.0,  help="Thermal conductivity")
    parser.add_argument("--L",         type=float, default=1.0,  help="Domain length")
    parser.add_argument("--fileloc",   type=str,   default=".")
    parser.add_argument("--outputpic", type=str,   default="Temperature_HeatFlux.png")
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
    T_ex     = T_exact_flux(x_s, args.time, q=args.q, k=args.k, L=args.L)

    rms = np.sqrt(np.mean((T_s - T_ex)**2))
    print(f"RMS error: {rms:.4e}")

    plt.figure(figsize=(8, 6))
    plt.scatter(x_s, T_s,  label="ExaGOOP", color="black", s=10)
    plt.plot(x_s,    T_ex, label=f"Exact (q={args.q})", color="red",
             linestyle="--", linewidth=2)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("Temperature", fontsize=14)
    plt.title(f"2D Heat Flux — T vs x at t={args.time:.4f}\nRMS={rms:.2e}")
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.outputpic)
    print(f"Saved: {args.outputpic}")

if __name__ == "__main__":
    main()
