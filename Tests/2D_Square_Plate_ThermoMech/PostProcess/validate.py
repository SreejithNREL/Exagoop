#!/usr/bin/env python3
"""
Validation for: square plate with thermal and mechanical loadings
(Nguyen et al., Sect. 10.3.3, Fig. 10.21, Table 10.2, Fig. 10.22; after
Tao et al. 2016).

SCOPE: thermal half only. ExaGOOP has no thermo-elastic coupling (no thermal
strain term), so Fig. 10.22(b) -- sigma_xx at point A -- is out of reach. With
no gravity, no external load and no thermal strain the material stays exactly
at rest, so the thermal problem is faithful and this reproduces Fig. 10.22(a).

Checks:
  1  T at point A (0.1, 0.1) m: starts at 20 C, rises monotonically, and
     reaches ~29 C at t = 10 s (book Fig. 10.22a).
  2  Physical bounds: T never exceeds the ambient T_inf = 30 C by more than a
     small margin, and never drops below the 20 C initial value.
  3  The material is static (this is what makes the thermal half faithful).
  4  The insulated left wall stays colder than the convective right wall.

Usage:  python3 validate.py
"""

import glob
import os
import sys

import numpy as np

# 2D, USE_TEMP=TRUE: col0,1 = x,y then real[k] = col[k+2]; temperature = real41.
C_X, C_Y, C_VX, C_TEMPERATURE = 0, 1, 3, 43

T0, T_INF = 20.0, 30.0
BOOK_T_A_FINAL = 29.0        # Fig. 10.22(a) at t = 10 s
POINT_A = (0.1, 0.1)


def main():
    root = "Solution/ascii_files/2D_Square_Plate_ThermoMech"
    fs = sorted(glob.glob(os.path.join(root, "matpnt_t*")),
                key=lambda f: float(f.split("_t")[-1]))
    if not fs:
        raise SystemExit(f"[ERROR] no frames under {root}")

    d0 = np.loadtxt(fs[0], skiprows=5)
    iA = int(np.argmin(np.hypot(d0[:, C_X] - POINT_A[0], d0[:, C_Y] - POINT_A[1])))
    print(f"\n=== Square plate, thermal + mechanical loading ===")
    print(f"frames {len(fs)}   point A particle at "
          f"({d0[iA, C_X]:.4f}, {d0[iA, C_Y]:.4f}) m")

    ts, TA, vmax, Tmin, Tmax = [], [], [], [], []
    left_last = right_last = None
    for f in fs:
        t = float(f.split("_t")[-1])
        d = np.loadtxt(f, skiprows=5)
        T = d[:, C_TEMPERATURE]
        ts.append(t); TA.append(T[iA]); Tmin.append(T.min()); Tmax.append(T.max())
        vmax.append(np.hypot(d[:, C_VX], d[:, C_VX + 1]).max())
        left_last = T[d[:, C_X] < 0.01].mean()
        right_last = T[d[:, C_X] > 0.19].mean()
    ts, TA = np.array(ts), np.array(TA)
    fails = []

    print("\n[1] T at point A")
    for tt in range(0, 11, 2):
        k = int(np.argmin(np.abs(ts - tt)))
        print(f"      t = {ts[k]:5.1f} s   T_A = {TA[k]:8.4f} C")
    print(f"      start {TA[0]:.4f} -> end {TA[-1]:.4f} C "
          f"(book Fig. 10.22a: {T0:.0f} -> ~{BOOK_T_A_FINAL:.0f})")
    if abs(TA[0] - T0) > 1e-6:
        fails.append("point A does not start at the 20 C initial condition")
    if np.any(np.diff(TA) < -1e-9):
        fails.append("T at A is not monotonically increasing")
    if abs(TA[-1] - BOOK_T_A_FINAL) > 1.0:
        fails.append(f"T_A(10 s) = {TA[-1]:.2f} C, more than 1 C from the "
                     f"book's ~{BOOK_T_A_FINAL:.0f} C")

    print("\n[2] physical bounds")
    print(f"      global T range over the run: "
          f"[{min(Tmin):.4f}, {max(Tmax):.4f}] C   (T_inf = {T_INF})")
    if max(Tmax) > T_INF + 0.5:
        fails.append(f"T exceeds ambient by >0.5 C (max {max(Tmax):.3f})")
    if min(Tmin) < T0 - 1e-6:
        fails.append("T drops below the initial 20 C")

    print("\n[3] material static (no mechanics without thermal strain)")
    print(f"      max |v| over all frames = {max(vmax):.3e} m/s")
    if max(vmax) > 1e-12:
        fails.append("material is moving; the thermal half is not isolated")

    print("\n[4] insulated left vs convective right wall (final frame)")
    print(f"      mean T: left {left_last:.4f} C   right {right_last:.4f} C")
    if left_last >= right_last:
        fails.append("insulated wall is not colder than the convective wall")

    print("\n" + "=" * 58)
    print("RESULT: " + ("FAIL" if fails else "PASS"))
    for f in fails:
        print("  - " + f)
    print("=" * 58 + "\n")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
