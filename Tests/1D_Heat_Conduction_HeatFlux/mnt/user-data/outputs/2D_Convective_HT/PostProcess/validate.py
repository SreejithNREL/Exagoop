"""
validate.py — 2D Convective Heat Transfer test case
=====================================================
Domain  : x in [0,1]
BCs     : T(0,t)   = 1        (Dirichlet)
          -k*dT/dx(1,t) = h*(T(1,t) - T_inf)   with h=2, k=1, T_inf=0
ICs     : T(x,0) = 0

Exact solution (shifted variable u = T - 1, so u(0,t)=0, u(x,0)=-1):
  Governing: du/dt = d^2u/dx^2   (alpha=k/(rho*cp)=1)
  BCs: u(0,t) = 0
       du/dx(1,t) = -(h/k)*u(1,t) = -2*u(1,t)

  Eigenfunctions: sin(lambda_n * x)
  Eigenvalues satisfy: lambda_n * cos(lambda_n) + (h/k) * sin(lambda_n) = 0
    i.e. lambda_n * cot(lambda_n) = -h/k = -2
    or equivalently: tan(lambda_n) = -lambda_n / (h/k) = -lambda_n / 2

  Coefficients:
    C_n = -2 * integral_0^1 sin(lambda_n * x) dx
             / integral_0^1 sin^2(lambda_n * x) dx
        = -2 * (1 - cos(lambda_n)) / lambda_n
             / (1/2 - sin(2*lambda_n)/(4*lambda_n))

  T(x,t) = 1 + sum_{n=1}^{N} C_n * sin(lambda_n * x) * exp(-lambda_n^2 * t)

Parameters: k=1, h=2, T_inf=0, T_wall=1, L=1, alpha=1
"""

import sys
import glob
import numpy as np
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


def find_eigenvalues(h_over_k, L=1.0, N=50):
    """
    Find first N positive roots of:  lambda * cot(lambda*L) = -h/k
    Equivalently: lambda*cos(lambda*L) + (h/k)*sin(lambda*L) = 0
    Roots are bracketed in ((n-0.5)*pi/L, (n+0.5)*pi/L) for n=1,2,...
    """
    def f(lam):
        return lam * np.cos(lam * L) + h_over_k * np.sin(lam * L)

    lambdas = []
    for n in range(1, N + 2):
        lo = (n - 0.5) * np.pi / L + 1e-8
        hi = (n + 0.5) * np.pi / L - 1e-8
        try:
            root = brentq(f, lo, hi, xtol=1e-12)
            lambdas.append(root)
        except ValueError:
            pass
        if len(lambdas) == N:
            break
    return np.array(lambdas)


def T_exact_convective(x, t, T_wall=1.0, h=2.0, k=1.0, T_inf=0.0,
                        L=1.0, N=50):
    """
    T(x,t) = T_wall + sum_n C_n * sin(lambda_n*x) * exp(-lambda_n^2*t)
    where u = T - T_wall satisfies homogeneous Dirichlet at x=0 and
    Robin at x=L.
    """
    x    = np.asarray(x, dtype=float)
    lams = find_eigenvalues(h / k, L=L, N=N)

    T = np.full_like(x, T_wall)
    for lam in lams:
        # norm^2 = integral_0^L sin^2(lam*x) dx
        norm2 = L / 2.0 - np.sin(2.0 * lam * L) / (4.0 * lam)
        # integral_0^L u(x,0)*sin(lam*x) dx  with u(x,0) = -T_wall
        num   = -T_wall * (1.0 - np.cos(lam * L)) / lam
        C_n   = num / norm2
        T    += C_n * np.sin(lam * x) * np.exp(-lam**2 * t)

    return T


# ---------------------------------------------------------------
# Test parameters — must match Inputs_2DConvectiveHT.inp
# ---------------------------------------------------------------
T_TIME   = 0.05
T_WALL   = 1.0    # bc_lower_tempval  (Dirichlet at x=0)
H_CONV   = 2.0    # bc_upper_tempval  (h coefficient)
T_INF    = 0.0    # bc_upper_Tinf
K_COND   = 1.0
L_DOMAIN = 1.0
RMS_TOL  = 1e-2

# ---------------------------------------------------------------
# Find output file
# ---------------------------------------------------------------
patterns = [
    "Solution/ascii_files/*/matpnt_t0.050000",
    "2D_Convective_HT/matpnt_t0.050000",
    "*/matpnt_t0.050000",
]
matches = []
for pat in patterns:
    matches = glob.glob(pat)
    if matches:
        break

if not matches:
    print("FAIL: no matpnt_t0.050000 found")
    sys.exit(1)

FILEPATH = matches[0]
print(f"Found output file: {FILEPATH}")

data   = np.loadtxt(FILEPATH, skiprows=5)
fields = build_field_dict(AMREX_SPACEDIM, USE_TEMP)
T_idx  = fields["temperature"]
x      = data[:, 0]
T_num  = data[:, T_idx]

T_ex  = T_exact_convective(x, T_TIME, T_wall=T_WALL, h=H_CONV,
                            k=K_COND, T_inf=T_INF, L=L_DOMAIN)
rms   = np.sqrt(np.mean((T_num - T_ex) ** 2))

print(f"RMS error vs exact: {rms:.4e}  (tolerance: {RMS_TOL:.0e})")
if rms < RMS_TOL:
    print("PASS")
    sys.exit(0)
else:
    print("FAIL")
    sys.exit(1)
