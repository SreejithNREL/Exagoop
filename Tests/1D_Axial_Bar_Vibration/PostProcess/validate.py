import sys
import os
import numpy as np

def exact_velocity(x, t, V0=0.1, L=25.0, E=100, rho=1):
    omega1 = (np.pi / (2 * L)) * np.sqrt(E / rho)
    return V0 * np.sin(np.pi * x / (2 * L)) * np.cos(omega1 * t)

T = 0.5
RMS_TOL = 1e-3
FILEPATH = "1D_Axial_Bar_Vibration_dim1_npcx1_ord3_flip1.0_sus1_CFL0.1_8825d8/matpnt_t0.500000"

if not os.path.exists(FILEPATH):
    print(f"FAIL: output file not found: {FILEPATH}")
    sys.exit(1)

data = np.loadtxt(FILEPATH, skiprows=5)
assert data.shape[1] == 59, f"Expected 59 columns, got {data.shape[1]}"

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
