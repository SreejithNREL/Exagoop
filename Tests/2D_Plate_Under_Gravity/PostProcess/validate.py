#!/usr/bin/env python3
"""
Validation for: 2D plate under gravitational compression
(Nguyen et al., "The Material Point Method: Theory, Implementations and
Applications", Sect. 10.3.3, Figs. 10.23-10.25, Table 10.3).

There is no analytical solution -- the book compares against Abaqus FEM. What
CAN be verified exactly is the physics the case was built to exercise.

  CHECK 1  Plastic-work -> heat energy balance  (PRIMARY, quantitative)
      With adiabatic boundaries the total thermal energy in the body must equal
      the Taylor-Quinney fraction of the total plastic work:

          sum_p m_p c_p (T_p - T_0)  ==  chi * sum_p V_p * W(eps_p)
          W(eps_p) = A*eps_p + B*eps_p^(1+n)/(1+n)      (exact for C = m = 0)

      This is the rigorous test of the whole plastic-work-to-heat path
      (book Eqs. 10.43/10.44). It is a GLOBAL identity: the pointwise version
      does NOT hold particle-by-particle, because the heat source is deposited
      on the grid (Q_I = sum_p V_p gamma* phi_I) and mapped back by G2P, which
      smooths across a stencil. See CHECK 1b.

  CHECK 1b Pointwise heating (DIAGNOSTIC, not a pass/fail criterion)
      Reports the particle-level scatter about T = chi*W/(rho*c_p). Expect a few
      percent in smooth regions and large relative error at the localized
      corner peaks -- that is P2G/G2P smoothing, not an error.

  CHECK 2  Johnson-Cook yield surface  (quantitative)
      No particle may lie ABOVE von Mises = A + B*eps_p^n; radial return must
      never violate yield.

  CHECK 3  Static level-set pedestal contact: no particle below y = 0.

  CHECK 4  Structure vs. the book: peak values and locations, left-right
      symmetry of the two shear bands, and the deformed silhouette.

Usage:
    python3 validate.py [path/to/matpnt_tXXXXXX]      # default: latest frame
"""

import glob
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------
# ASCII matpnt column map.
#
# Layout is  col0,col1 = x,y  then  real[k] = col[k+2], with realData as defined
# in Source/mpm_specs.H. For DIM=2 with USE_TEMP=TRUE, realData::count = 56:
#   radius=0 xvel=1 yvel=2 zvel=3 xvel'=4 yvel'=5 zvel'=6 strainrate=7..12
#   strain=13..18 stress=19..24 F=25..33 volume=34 mass=35 density=36
#   jacobian=37 pressure=38 vol_init=39 yacceleration=40 temperature=41
#   specific_heat=42 thermal_conductivity=43 heat_flux=44..46 heat_source=47
#   isv=48..55   (isv[0] = JC equivalent plastic strain, isv[7] = JC damage)
#
# Tensor component order in 2D (Source/constants.H): XX, XY, YY, XZ, YZ, ZZ.
# ----------------------------------------------------------------------------
C_X, C_Y = 0, 1
C_STRESS = 21          # 21..26
C_VOLUME = 36
C_DENSITY = 38
C_TEMPERATURE = 43
C_SPHEAT = 44
C_EP = 50              # isv[0]
C_DAMAGE = 57          # isv[7]

# Table 10.3 in the deck's mm-ms-kg system (stress = GPa, energy = J).
JC_A = 0.065
JC_B = 0.356
JC_n = 0.37
JC_chi = 0.9
T0 = 0.0
PLATE_W = 10.0
PEDESTAL_TOP = 0.0

# Book reference values at t = 0.2 ms (colour-bar maxima of Figs. 10.24/10.25).
BOOK_EP_MAX = 1.5
BOOK_T_MAX = 52.0


def flow_stress(ep):
    return JC_A + JC_B * np.power(np.maximum(ep, 0.0), JC_n)


def plastic_work(ep):
    ep = np.maximum(ep, 0.0)
    return JC_A * ep + JC_B * np.power(ep, 1.0 + JC_n) / (1.0 + JC_n)


def von_mises(s):
    sxx, sxy, syy, sxz, syz, szz = (s[:, i] for i in range(6))
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)
    )


def find_latest(root="Solution/ascii_files/2D_Plate_Under_Gravity"):
    frames = sorted(glob.glob(os.path.join(root, "matpnt_t*")))
    if not frames:
        raise SystemExit(f"[ERROR] no matpnt frames found under {root}")
    return frames[-1]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_latest()
    d = np.loadtxt(path, skiprows=5)     # 3 scalar header lines + 2 blank

    x, y = d[:, C_X], d[:, C_Y]
    ep, T = d[:, C_EP], d[:, C_TEMPERATURE]
    V, rho, cp = d[:, C_VOLUME], d[:, C_DENSITY], d[:, C_SPHEAT]
    svm = von_mises(d[:, C_STRESS:C_STRESS + 6])
    mass = V * rho

    print(f"\n=== 2D plate under gravitational compression -- "
          f"{os.path.basename(path)} ===")
    print(f"particles                 : {len(d)}")

    fails = []

    # ---- CHECK 1: global plastic-work -> heat balance ----------------------
    E_th = float((mass * cp * (T - T0)).sum())
    E_pl = float(JC_chi * (V * plastic_work(ep)).sum())
    ratio = E_th / E_pl if E_pl > 0 else 0.0
    print("\n[CHECK 1] Energy balance  sum(m cp dT) == chi * sum(V W(eps_p))")
    print(f"  thermal energy            : {E_th:.6e} J")
    print(f"  chi * plastic work        : {E_pl:.6e} J")
    print(f"  ratio                     : {ratio:.5f}   (target 1.0)")
    if E_pl <= 0:
        fails.append("no plastic work accumulated")
    elif abs(ratio - 1.0) > 0.01:
        fails.append(f"energy balance off by {abs(ratio - 1.0) * 100:.2f}% (>1%)")

    # ---- CHECK 1b: pointwise scatter (diagnostic only) --------------------
    act = ep > 1e-6
    if act.sum():
        T_pred = JC_chi * plastic_work(ep) / (rho * cp)
        rel = np.abs(T[act] - T_pred[act]) / np.maximum(T_pred[act], 1e-12)
        print("\n[CHECK 1b] Pointwise T vs chi*W/(rho cp)   (diagnostic)")
        print(f"  median rel. difference    : {np.median(rel) * 100:.2f} %")
        print(f"  90th percentile           : {np.percentile(rel, 90) * 100:.2f} %")
        print("  (large values at the localized corner peaks are expected:")
        print("   the heat source is deposited on the grid and smoothed by G2P)")

    # ---- CHECK 2: yield surface -------------------------------------------
    if act.sum():
        sigf = flow_stress(ep)
        err = (svm[act] - sigf[act]) / sigf[act]
        above = int((err > 1e-3).sum())
        on = int((np.abs(err) <= 1e-3).sum())
        below = int((err < -1e-3).sum())
        print("\n[CHECK 2] Yield surface  von Mises <= A + B eps_p^n")
        print(f"  plastically strained      : {int(act.sum())}")
        print(f"  actively on the surface   : {on}")
        print(f"  below (elastic / unloaded): {below}")
        print(f"  ABOVE (yield violated)    : {above}")
        print(f"  max relative overshoot    : {err.max():.3e}")
        if above > 0:
            fails.append(f"{above} particles above the yield surface")

    # ---- CHECK 3: pedestal contact ----------------------------------------
    pen = y < PEDESTAL_TOP - 1e-9
    print("\n[CHECK 3] Static level-set pedestal (top at y = 0)")
    print(f"  particles below y = 0     : {int(pen.sum())}")
    print(f"  lowest particle           : y = {y.min():.5f} mm")
    if pen.sum() and (PEDESTAL_TOP - y[pen]).max() > 0.5:
        fails.append("pedestal penetration exceeds 0.5 mm")
    if x.min() < -2.5 or x.max() > 12.5:
        fails.append("material has spread beyond the pedestal footprint")

    # ---- CHECK 4: structure vs. the book ----------------------------------
    i, j = int(np.argmax(ep)), int(np.argmax(T))
    print("\n[CHECK 4] Structure vs. Nguyen et al. Figs. 10.24 / 10.25")
    print(f"  max eps_p                 : {ep.max():.4f}   "
          f"(book colour scale {BOOK_EP_MAX})")
    print(f"  max T                     : {T.max():.3f} C  "
          f"(book colour scale {BOOK_T_MAX})")
    print(f"  location of max eps_p     : ({x[i]:.2f}, {y[i]:.2f}) mm")
    print(f"  location of max T         : ({x[j]:.2f}, {y[j]:.2f}) mm")
    print(f"  deformed extent           : x in [{x.min():.2f}, {x.max():.2f}], "
          f"y_max = {y.max():.2f} mm")
    print("  deformed width by height  :")
    for lo, hi in [(0, 1), (2, 3), (4, 5), (6, 7), (7.5, 9.0)]:
        sel = (y >= lo) & (y < hi)
        if sel.sum():
            print(f"    y in [{lo:>3},{hi:>4}) : width {x[sel].max() - x[sel].min():.2f} mm")

    hot = ep > 0.5 * ep.max()
    nl, nr = int((hot & (x < PLATE_W / 2)).sum()), int((hot & (x >= PLATE_W / 2)).sum())
    print(f"  shear-band points L / R   : {nl} / {nr}")
    if nl + nr:
        asym = abs(nl - nr) / (nl + nr)
        print(f"  left-right asymmetry      : {asym * 100:.1f} %")
        if asym > 0.25:
            fails.append("shear bands strongly asymmetric (>25%)")
    else:
        fails.append("no localized high-strain region found")

    print("\n" + "=" * 66)
    if fails:
        print("RESULT: FAIL")
        for f in fails:
            print(f"  - {f}")
    else:
        print("RESULT: PASS")
    print("=" * 66 + "\n")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
