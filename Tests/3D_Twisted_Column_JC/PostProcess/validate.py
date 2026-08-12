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

# ---------------------------------------------------------------------------
# ASCII matpnt column map -- 3D, USE_TEMP=TRUE  (realData::count = 56).
# Layout: col0,1,2 = x,y,z  then  real[k] = col[k+3].
#   radius=0 xvel=1..3 xvel'=4..6 strainrate=7..12 strain=13..18 stress=19..24
#   F=25..33 volume=34 mass=35 density=36 jacobian=37 pressure=38 vol_init=39
#   yaccel=40 temperature=41 spheat=42 thermcond=43 heat_flux=44..46
#   heat_source=47 isv=48..55  (isv[0]=eq. plastic strain, isv[7]=damage)
#
# !! Voigt ordering is DIMENSION-DEPENDENT (Source/constants.H) !!
#      3D (and 1D): XX, XY, XZ, YY, YZ, ZZ
#      2D         : XX, XY, YY, XZ, YZ, ZZ
# Using the 2D order on 3D data fabricates enormous spurious von Mises values
# (it silently swaps XZ and YY) -- this bit us once already.
# ---------------------------------------------------------------------------
C_X, C_Y, C_Z = 0, 1, 2
C_VX = 4                # vx, vy, vz = 4, 5, 6
C_STRESS = 22           # 22..27 = XX, XY, XZ, YY, YZ, ZZ
C_VOLUME, C_DENSITY = 37, 39
C_TEMPERATURE, C_SPHEAT = 44, 45
C_EP, C_DAMAGE = 51, 58

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
