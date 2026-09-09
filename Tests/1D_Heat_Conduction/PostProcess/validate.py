import sys
import os
import glob
import numpy as np

USE_TEMP = True
AMREX_SPACEDIM=1

# Column map for ExaGOOP ASCII particle dumps: shared helper, kept in sync
# with Source/mpm_specs.H (was a hard-coded per-script copy before).
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "..", "..", "..", "Tools", "PostProcess"))
from exagoop_columns import build_field_dict

def logical_index(name, fields):
    if name not in fields:
        raise KeyError(f"{name} not in dictionary")
    return fields[name]



def Texact(x, T0, T1, t, N=100):
    x = np.asarray(x)
    base = T0 + (T1 - T0) * x
    series = np.zeros_like(x)
    for n in range(1, N + 1):
        series += ((-1) ** n) / (n * np.pi) * np.exp(-(n * np.pi) ** 2 * t) * np.sin(n * np.pi * x)
    return base + 2 * (T1 - T0) * series

T_TIME = 0.05
T0 = 0.0
T1 = 1.0
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
fields = build_field_dict(AMREX_SPACEDIM, True)
temperature_idx = fields["temperature"]
x=data[:,0]
T_num = data[:, temperature_idx]  

T_ex = Texact(x, T0, T1, T_TIME)
rms = np.sqrt(np.mean((T_num - T_ex) ** 2))

if rms < RMS_TOL:
    print(f"PASS: RMS={rms:.2e}")
    sys.exit(0)
else:
    print(f"FAIL: RMS={rms:.2e} exceeds {RMS_TOL:.0e}")
    sys.exit(1)
