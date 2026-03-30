import sys
import os
import glob
import numpy as np

def Texact_2D(x, y, T0, T1, L, H, t, N_terms=25):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    T = np.full_like(x, T1)
    factor = 16 * (T0 - T1) / (np.pi ** 2)
    for i in range(1, 2 * N_terms, 2):   # odd i: 1, 3, 5, ...
        for j in range(1, 2 * N_terms, 2):  # odd j: 1, 3, 5, ...
            decay = np.exp(-np.pi ** 2 * (i ** 2 / L ** 2 + j ** 2 / H ** 2) * t)
            T += factor * decay / (i * j) * np.sin(i * np.pi * x / L) * np.sin(j * np.pi * y / H)
    return T

T_TIME = 0.05
T0 = 0.0
T1 = 1.0
L = 1.0
H = 1.0
N_terms = 25
RMS_TOL = 1e-2
matches = glob.glob("Solution/ascii_files/*/matpnt_t0.050000")
if not matches:
    matches = glob.glob("CI_Output/matpnt_t0.050000")
if not matches:
    matches = glob.glob("*/matpnt_t0.050000")

if not matches:
    print("FAIL: no matpnt_t0.050000 found anywhere")
    print("Searched: Solution/ascii_files/*/matpnt_t0.050000")
    sys.exit(1)

FILEPATH = matches[0]
print(f"Found output file: {FILEPATH}")

data = np.loadtxt(FILEPATH, skiprows=5)
assert data.shape[1] >= 49, f"Expected at least 49 columns, got {data.shape[1]}"

x = data[:, 0]
y = data[:, 1]
T_num = data[:, 48]

T_ex = Texact_2D(x, y, T0, T1, L, H, T_TIME, N_terms)
rms = np.sqrt(np.mean((T_num - T_ex) ** 2))

if rms < RMS_TOL:
    print(f"PASS: RMS={rms:.2e}")
    sys.exit(0)
else:
    print(f"FAIL: RMS={rms:.2e} exceeds {RMS_TOL:.0e}")
    sys.exit(1)
