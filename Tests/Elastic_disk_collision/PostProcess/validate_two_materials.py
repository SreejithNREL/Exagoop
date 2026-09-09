#!/usr/bin/env python3
"""
Two-material regression check (tracker task 3.5).

Two elastic disks with DIFFERENT Young's moduli (material 0: E = 1000 Pa,
material 1: E = 10000 Pa, both nu = 0.3) collide. For a linear-elastic
particle the solver computes sigma = C(E, nu) : eps exactly, so the Young's
modulus each particle was actually given can be back-solved from its stress
and strain in the ASCII dump and compared, per material_indx, with the value
in the input file. This verifies that the per-particle material index really
selects the right entry of the material table.

Usage:  python3 PostProcess/validate_two_materials.py [matpnt file]
        (default: the latest matpnt_t* file of the two-materials output tag)
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "Tools", "PostProcess"))
from exagoop_columns import build_field_dict

DIM, USE_TEMP = 2, False
NU = 0.3
E_EXPECTED = {0: 1000.0, 1: 10000.0}
TOL = 1e-6          # relative; the relation is exact for linear elasticity
MIN_STRAIN = 1e-8   # ignore particles that have not been strained yet

def plane_strain_stress_unitE(exx, eyy, exy, nu):
    """sigma for E = 1 (2-D plane strain, as in linear_elastic())."""
    c1 = 1.0 / ((1 + nu) * (1 - 2 * nu))
    c2 = 1.0 / (1 + nu)
    sxx = c1 * ((1 - nu) * exx + nu * eyy)
    syy = c1 * ((1 - nu) * eyy + nu * exx)
    sxy = c2 * exy
    return sxx, syy, sxy

def main():
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        files = sorted(glob.glob("Solution/ascii_files/*two_materials*/matpnt_t*"))
        if not files:
            sys.exit("No matpnt file found; run the two-materials case first.")
        fname = files[-1]
    print("Reading", fname)

    data = np.loadtxt(fname, skiprows=5)
    f = build_field_dict(DIM, USE_TEMP)
    assert data.shape[1] == f["ncols"], \
        f"{data.shape[1]} columns but the layout predicts {f['ncols']}"

    mid = data[:, f["material_indx"]].astype(int)
    # 2-D Voigt order (Source/constants.H): XX=0, XY=1, YY=2
    exx, exy, eyy = (data[:, f["strain"] + k] for k in (0, 1, 2))
    sxx, sxy, syy = (data[:, f["stress"] + k] for k in (0, 1, 2))
    uxx, uyy, uxy = plane_strain_stress_unitE(exx, eyy, exy, NU)

    ok = True
    for m, E_exp in E_EXPECTED.items():
        sel = (mid == m) & (np.abs(exx) + np.abs(eyy) + np.abs(exy) > MIN_STRAIN)
        if sel.sum() == 0:
            print(f"material {m}: no strained particles yet (run longer)")
            ok = False
            continue
        # least-squares E: sigma = E * sigma_unit
        num = (sxx[sel] * uxx[sel] + syy[sel] * uyy[sel] + sxy[sel] * uxy[sel]).sum()
        den = (uxx[sel] ** 2 + uyy[sel] ** 2 + uxy[sel] ** 2).sum()
        E_fit = num / den
        # worst single-particle relative residual
        res = np.sqrt((sxx[sel] - E_fit * uxx[sel]) ** 2 + (syy[sel] - E_fit * uyy[sel]) ** 2
                      + (sxy[sel] - E_fit * uxy[sel]) ** 2)
        scale = np.sqrt(sxx[sel] ** 2 + syy[sel] ** 2 + sxy[sel] ** 2).max()
        rel = abs(E_fit - E_exp) / E_exp
        status = "PASS" if rel < TOL and res.max() / scale < 1e-6 else "FAIL"
        ok &= status == "PASS"
        print(f"material {m}: {sel.sum():6d} particles  E_fit = {E_fit:12.6f}  "
              f"expected {E_exp:8.1f}  rel.err = {rel:.2e}  worst residual = {res.max()/scale:.1e}  {status}")

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
