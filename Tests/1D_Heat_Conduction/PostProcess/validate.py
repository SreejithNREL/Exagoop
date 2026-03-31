import sys
import os
import glob
import numpy as np

USE_TEMP = True
AMREX_SPACEDIM=1

BASE_SCALAR_FIELDS = {
    "radius": None,   # index assigned later
    "xvel": None,
    "yvel": None,
    "zvel": None,
    "xvel_prime": None,
    "yvel_prime": None,
    "zvel_prime": None,
    "strainrate": None,
    "strain": None,
    "stress": None,
    "deformation_gradient": None,
    "volume": None,
    "mass": None,
    "density": None,
    "jacobian": None,
    "pressure": None,
    "vol_init": None,
    "E": None,
    "nu": None,
    "Bulk_modulus": None,
    "Gama_pressure": None,
    "Dynamic_viscosity": None,
    "yacceleration": None,
}
TEMP_FIELDS = {
    "temperature": None,
    "specific_heat": None,
    "thermal_conductivity": None,
    "heat_flux": None,
    "heat_source": None,
}
INT_FIELDS = {
    "phase": None,
    "rigid_body_id": None,
    "constitutive_model": None,
}

def build_field_dict(dim: int, use_temp: bool):
    if dim not in (1, 2, 3):
        raise ValueError("AMREX_SPACEDIM must be 1, 2, or 3")

    fields = {}
    idx = 0

    # 1. Position fields
    for d in range(dim):
        fields[f"pos{'xyz'[d]}"] = idx
        idx += 1

    # 2. Scalar fields (dimension‑independent)
    for name in BASE_SCALAR_FIELDS:
        fields[name] = idx
        idx += 1

    # 3. Temperature fields (conditional)
    if use_temp:
        for name in TEMP_FIELDS:
            fields[name] = idx
            idx += 1

    # 4. Integer fields appended at the end
    for name in INT_FIELDS:
        fields[name] = idx
        idx += 1

    return fields

def field_index(name: str, fields: dict) -> int:
    if name not in fields:
        raise KeyError(
            f"Field '{name}' not available. Valid fields: {list(fields.keys())}"
        )
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


x = data[:, 0]
FIELDS = build_field_dict(dim=AMREX_SPACEDIM, use_temp=USE_TEMP)
temperature_idx = FIELDS["temperature"]

print("Temperature index = ",temperature_idx)

T_num = data[:, temperature_idx]

T_ex = Texact(x, T0, T1, T_TIME)
rms = np.sqrt(np.mean((T_num - T_ex) ** 2))

if rms < RMS_TOL:
    print(f"PASS: RMS={rms:.2e}")
    sys.exit(0)
else:
    print(f"FAIL: RMS={rms:.2e} exceeds {RMS_TOL:.0e}")
    sys.exit(1)
