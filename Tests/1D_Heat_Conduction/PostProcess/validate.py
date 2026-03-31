import sys
import os
import glob
import numpy as np

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
