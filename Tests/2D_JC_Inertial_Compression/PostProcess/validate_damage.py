#!/usr/bin/env python3
"""
Johnson-Cook damage / failure check (inertial compression, JC_D1 = 0.15).

With D2..D5 = 0 the failure strain is constant, eps_f = D1, so the damage
variable must satisfy D = ep / eps_f exactly on every particle that has not
failed. A particle fails when D reaches 1: its deviatoric stress is removed
(von Mises = 0) and only compressive pressure is retained.

Pass criteria (last output):
  * unfailed particles (D < 1): |D - ep/eps_f| < 1e-10
  * failed particles (D == 1): von Mises stress < 1e-8 * max von Mises, and
    ep >= eps_f
  * at least one particle has failed (otherwise the run was too short)
  * unfailed yielded particles still obey the flow surface (none above)
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "Tools", "PostProcess"))
from exagoop_columns import build_field_dict

DIM, USE_TEMP = 2, False
A, B, N, EPS_F = 1.0, 2.0, 0.5, 0.15   # must match PreProcess/config_damage.json

def main():
    files = sorted(glob.glob("Solution/ascii_files/*Damage*/matpnt_t*"))
    if len(files) < 2:
        sys.exit("run the damage case first")
    f = build_field_dict(DIM, USE_TEMP)
    d = np.loadtxt(files[-1], skiprows=5)
    assert d.shape[1] == f["ncols"], (d.shape[1], f["ncols"])
    ep, dmg = d[:, f["isv_0"]], d[:, f["isv_7"]]
    sxx, sxy, syy, szz = (d[:, f["stress"] + k] for k in (0, 1, 2, 5))
    svm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * sxy ** 2)

    failed = dmg >= 1.0
    unf = ~failed
    print(f"{os.path.basename(files[-1])}: N={len(ep)} failed={failed.sum()} ep_max={ep.max():.4f} damage_max={dmg.max():.4f}")

    ok = True
    err_D = np.abs(dmg[unf] - ep[unf] / EPS_F)
    c1 = err_D.max() < 1e-10
    print(f"  unfailed: max |D - ep/eps_f| = {err_D.max():.1e}  {'PASS' if c1 else 'FAIL'}")
    ok &= c1

    c2 = failed.sum() > 0
    print(f"  failed particles present: {failed.sum()}  {'PASS' if c2 else 'FAIL (run longer)'}")
    ok &= c2
    if c2:
        c3 = svm[failed].max() < 1e-8 * svm.max() and ep[failed].min() >= EPS_F - 1e-12
        print(f"  failed: max svm = {svm[failed].max():.1e} (block max {svm.max():.3f}), min ep = {ep[failed].min():.4f}  {'PASS' if c3 else 'FAIL'}")
        ok &= c3

    y = unf & (ep > 1e-8)
    sf = A + B * ep[y] ** N
    above = ((svm[y] - sf) / sf > 1e-6).sum()
    c4 = above == 0
    print(f"  unfailed yielded above surface: {above}  {'PASS' if c4 else 'FAIL'}")
    ok &= c4

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
