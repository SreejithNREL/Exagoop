#!/usr/bin/env python3
"""
Validation for: 3D twisted column, Johnson-Cook
(Nguyen et al., Sect. 10.3.3, Fig. 10.26 setup, Table 10.3 material,
Fig. 10.27 FEM / Fig. 10.28 ULMPM.)

Checks:
  1  Plastic-work -> heat energy balance (global identity; exact for C = m = 0):
         sum_p m_p c_p dT  ==  chi * sum_p V_p [A ep + B ep^(1+n)/(1+n)]
  2  Johnson-Cook yield surface: no particle above von Mises = A + B ep^n.
  3  Clamped base: the z = 0 end must stay at rest.
  4  Twist profile: |v_xy| must increase monotonically with z, and the top
     surface must match the imposed rigid rotation |v| = omega * r.
  5  Reported against the book: sigma_eq (scale 440 MPa) and T (scale 76 C).

Usage:  python3 validate.py [path/to/matpnt_tXXXXXX]
"""

import glob
import os
import sys

import numpy as np

# Column map for ExaGOOP ASCII particle dumps: shared helper kept in sync with
# Source/mpm_specs.H (replaces the hard-coded map this script used to carry).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "Tools", "PostProcess"))
from exagoop_columns import build_field_dict
_F = build_field_dict(3, True)          # DIM=3, USE_TEMP=TRUE build
C_X, C_Y, C_Z = _F["posx"], _F["posy"], _F["posz"]
C_VX = _F["xvel"]                       # vx, vy, vz consecutive
C_STRESS = _F["stress"]                 # 6 comps, 3-D Voigt order XX XY XZ YY YZ ZZ
C_VOLUME, C_DENSITY = _F["volume"], _F["density"]
C_TEMPERATURE, C_SPHEAT = _F["temperature"], _F["specific_heat"]
C_EP, C_DAMAGE = _F["isv_0"], _F["isv_7"]

JC_A, JC_B, JC_n, JC_chi = 0.065, 0.356, 0.37, 0.9
OMEGA = 2.0 * np.pi     # rad/ms
COL_H = 100.0
BOOK_SIGMA_EQ_MAX = 440.0   # MPa
BOOK_T_MAX = 76.0           # C


def von_mises_3d(s):
    sxx, sxy, sxz, syy, syz, szz = (s[:, i] for i in range(6))
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)
    )


def main():
    root = "Solution/ascii_files/3D_Twisted_Column_JC"
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(root, "matpnt_t*")))[-1]
    t = float(os.path.basename(path).split("_t")[-1])
    d = np.loadtxt(path, skiprows=5)

    x, y, z = d[:, C_X], d[:, C_Y], d[:, C_Z]
    vxy = np.hypot(d[:, C_VX], d[:, C_VX + 1])
    ep, T = d[:, C_EP], d[:, C_TEMPERATURE]
    V, rho, cp = d[:, C_VOLUME], d[:, C_DENSITY], d[:, C_SPHEAT]
    svm = von_mises_3d(d[:, C_STRESS:C_STRESS + 6])
    mass = V * rho
    fails = []

    print(f"\n=== 3D twisted column (JC) -- t = {t:.6f} ms, {len(d)} particles ===")

    # 1 -- energy balance
    W = JC_A * ep + JC_B * np.power(np.maximum(ep, 0), 1 + JC_n) / (1 + JC_n)
    E_th, E_pl = float((mass * cp * T).sum()), float(JC_chi * (V * W).sum())
    ratio = E_th / E_pl if E_pl > 0 else 0.0
    print(f"\n[1] energy balance  E_th/(chi*W_pl) = {ratio:.5f}   (target 1)")
    if E_pl <= 0 or abs(ratio - 1) > 0.10:
        fails.append(f"energy balance off by {abs(ratio-1)*100:.1f}%")

    # 2 -- yield surface
    act = ep > 1e-6
    sigf = JC_A + JC_B * np.power(np.maximum(ep, 0), JC_n)
    err = (svm[act] - sigf[act]) / sigf[act]
    above = int((err > 1e-3).sum())
    print(f"[2] yield surface   plastic {int(act.sum())}, on-surface "
          f"{int((np.abs(err)<=1e-3).sum())}, ABOVE {above}, "
          f"max overshoot {err.max():.2e}")
    if above:
        fails.append(f"{above} particles above the yield surface")

    # 3 -- clamped base
    base = z < 2.0
    print(f"[3] clamped base    |v_xy|max at z<2 = {vxy[base].max():.3e} mm/ms")
    if vxy[base].max() > 0.05 * OMEGA * 5.0:
        fails.append("base is not clamped")

    # 4 -- twist profile
    print("[4] twist profile")
    meds = []
    for zlo in range(0, 100, 10):
        sl = (z >= zlo) & (z < zlo + 10)
        if sl.sum():
            meds.append(np.median(vxy[sl]))
            print(f"      z[{zlo:3d},{zlo+10:3d})  |v_xy| med {meds[-1]:8.4f}  "
                  f"max {vxy[sl].max():8.4f}  T max {T[sl].max():7.4f}")
    if any(b < a - 1e-9 for a, b in zip(meds, meds[1:])):
        fails.append("twist profile not monotonic in z")
    top = z > COL_H - 2.5
    imposed = OMEGA * np.hypot(x[top], y[top])
    frac = np.median(vxy[top] / np.maximum(imposed, 1e-12))
    print(f"      top surface |v|/(omega*r) median = {frac:.3f}  (1 = rigid rotation)")

    # 5 -- vs the book
    print(f"\n[5] vs book   sigma_eq max {svm.max()*1000:7.1f} MPa "
          f"(Fig. 10.28 scale {BOOK_SIGMA_EQ_MAX})")
    print(f"              T        max {T.max():7.3f} C   "
          f"(Fig. 10.28 scale {BOOK_T_MAX})")
    print(f"              ep       max {ep.max():7.4f}")
    print(f"              imposed top rotation = omega*t = "
          f"{np.degrees(OMEGA*t):.2f} deg")

    print("\n" + "=" * 60)
    print("RESULT: " + ("FAIL" if fails else "PASS"))
    for f in fails:
        print("  - " + f)
    print("=" * 60 + "\n")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
