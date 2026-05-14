"""
validate.py — 2D Heat Flux test case
=====================================
Domain  : x in [0,1], y in [0,0.1]
BCs     : T(0,t) = 0  (Dirichlet)
          dT/dx|_{x=1} = q/k = 1.0  (prescribed flux, k=1)
          y faces: adiabatic
ICs     : T(x,0) = 0

Exact solution (1D in x, Fourier series):
  T(x,t) = q*x/k
          + 2*q*L/(k*pi^2) * sum_{n=1}^{N}
                (-1)^n / n^2 * exp(-(n*pi/L)^2 * t) * cos(n*pi*x/L)
          - 2*q*L/(k*pi^2) * sum_{n=1}^{N} (-1)^n / n^2

  Simplified with k=1, L=1, q=1:
  T(x,t) = x + 2/pi^2 * sum_{n=1}^{N}
                (-1)^(n+1) / n^2 * exp(-(n*pi)^2 * t) * cos(n*pi*x)

  (derived from separation of variables with Neumann at x=L,
   Dirichlet at x=0)

Reference: Carslaw & Jaeger, "Conduction of Heat in Solids", eq. 3.4.4
"""

import sys
import os
import glob
import numpy as np

AMREX_SPACEDIM = 2
USE_TEMP = True

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



# ---------------------------------------------------------------
# Test parameters — must match Inputs_2DHeatFlux.inp
# ---------------------------------------------------------------
T_TIME  = 0.5
Q_FLUX  = 1.0   # bc_upper_tempval for x_hi (flux)
K_COND  = 1.0   # thermal conductivity set in particle file
L_DOMAIN = 1.0
RMS_TOL = 1e-2

# ---------------------------------------------------------------
# Find output file
# ---------------------------------------------------------------
patterns = [
    "Solution/ascii_files/*/matpnt_t0.500000",
    "2D_Heat_Flux/matpnt_t0.500000",
    "*/matpnt_t0.500000",
]
matches = []
for pat in patterns:
    matches = glob.glob(pat)
    if matches:
        break

if not matches:
    print("FAIL: no matpnt_t0.500000 found")
    sys.exit(1)

FILEPATH = matches[0]
print(f"Found output file: {FILEPATH}")

data   = np.loadtxt(FILEPATH, skiprows=5)
fields = build_field_dict(AMREX_SPACEDIM, USE_TEMP)
T_idx  = fields["temperature"]
x      = data[:, 0]
T_num  = data[:, T_idx]

T_ex   = T_exact_flux(x, T_TIME, q=Q_FLUX, k=K_COND, L=L_DOMAIN)
rms    = np.sqrt(np.mean((T_num - T_ex) ** 2))

print(f"RMS error vs exact: {rms:.4e}  (tolerance: {RMS_TOL:.0e})")
if rms < RMS_TOL:
    print("PASS")
    sys.exit(0)
else:
    print("FAIL")
    sys.exit(1)
