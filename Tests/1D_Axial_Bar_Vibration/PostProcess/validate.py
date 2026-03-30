import sys
import os
import glob
import numpy as np

def exact_velocity(x, t, V0=0.1, L=25.0, E=100, rho=1):
    omega1 = (np.pi / (2 * L)) * np.sqrt(E / rho)
    return V0 * np.sin(np.pi * x / (2 * L)) * np.cos(omega1 * t)

T = 50.0
RMS_TOL = 1e-1
matches = glob.glob("Solution/ascii_files/*/matpnt_t50.000000")
if not matches:
    matches = glob.glob("CI_Output/matpnt_t50.000000")
if not matches:
    matches = glob.glob("*/matpnt_t50.000000")

if not matches:
    print("FAIL: no matpnt_t50.000000 found anywhere")
    print("Searched: Solution/ascii_files/*/matpnt_t50.000000")
    sys.exit(1)

FILEPATH = matches[0]
print(f"Found output file: {FILEPATH}")

data = np.loadtxt(FILEPATH, skiprows=5)

x = data[:, 0]
vx_num = data[:, 3]

vx_exact = exact_velocity(x, T)
rms = np.sqrt(np.mean((vx_num - vx_exact) ** 2))

if rms < RMS_TOL:
    print(f"PASS: RMS={rms:.2e}")
    sys.exit(0)
else:
    print(f"FAIL: RMS={rms:.2e} exceeds {RMS_TOL:.0e}")
    sys.exit(1)
