#!/usr/bin/env python3
"""
Johnson-Cook yield-surface check (inertial uniaxial compression).

A rectangular JC block starts with the linear velocity field
v_x = -eps_dot (x - x_c) and compresses under its own inertia (free
surfaces, no gravity). The check is path-independent: for every particle that
has yielded (ep > 0) the radial-return update must leave the von Mises stress
either ON the Johnson-Cook flow surface sigma_f = A + B ep^n (particle
plastically loading) or BELOW it (particle elastically unloading as the block's
momentum reverses) - never above.

Pass criteria (per output time after t = 0):
  * no yielded particle above the surface (relative overshoot > 1e-6)
  * median relative distance of yielded particles from the surface < 1e-3
    (i.e. the bulk of the block is plastically loading on the surface)
  * all fields finite
The strain-rate (JC_C) and thermal (JC_m) factors are switched off in this
case, so sigma_f depends on ep only.
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "Tools", "PostProcess"))
from exagoop_columns import build_field_dict

DIM, USE_TEMP = 2, False
A, B, N = 1.0, 2.0, 0.5          # must match PreProcess/config.json
ABOVE_TOL, MEDIAN_TOL = 1e-6, 1e-3

def main():
    files = sorted(glob.glob("Solution/ascii_files/*/matpnt_t*"))
    if len(files) < 2:
        sys.exit("need at least one output after t=0; run the case first")
    f = build_field_dict(DIM, USE_TEMP)
    ok = True
    for fname in files[1:]:
        d = np.loadtxt(fname, skiprows=5)
        if d.shape[1] != f["ncols"]:
            sys.exit(f"{fname}: {d.shape[1]} columns, layout predicts {f['ncols']}")
        ep = d[:, f["isv_0"]]
        # 2-D Voigt order: XX=0, XY=1, YY=2, ZZ=5 (sigma_zz is non-zero: plane strain)
        sxx, sxy, syy, szz = (d[:, f["stress"] + k] for k in (0, 1, 2, 5))
        svm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * sxy ** 2)
        y = ep > 1e-8
        finite = np.isfinite(d).all()
        if y.sum() == 0:
            print(f"{os.path.basename(fname)}: no yielded particles")
            ok = False
            continue
        sf = A + B * ep[y] ** N
        rel = (svm[y] - sf) / sf
        above = (rel > ABOVE_TOL).sum()
        med = np.median(np.abs(rel))
        status = "PASS" if (above == 0 and med < MEDIAN_TOL and finite) else "FAIL"
        ok &= status == "PASS"
        print(f"{os.path.basename(fname)}: yielded {y.sum():5d}/{len(ep)}  ep_max {ep.max():.4f}  "
              f"above-surface {above}  median |svm-sf|/sf {med:.1e}  max overshoot {rel.max():+.1e}  "
              f"finite {finite}  {status}")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
