#!/usr/bin/env python3
"""
Diagnostics for the twisted-column run (any output time, any number of frames).

  python3 PostProcess/diagnose_twist.py                 # growth table over all frames
  python3 PostProcess/diagnose_twist.py --profile 1.5   # z-profiles at the frame nearest t=1.5 ms

Growth table: per frame, max |v_z| (should stay a few mm/ms - a rising value with
neighbouring particles moving in antiphase is the FLIP grid-scale mode), its z,
number of particles with |v_z| > 5, T_max, eps_p max.

Profile: per z-band, angular velocity / omega about the band centroid (uniform
twist -> z/L), median/max plastic strain and temperature, median |sigma_zz|,
median von Mises stress, and the torque transmitted across the section
M_z = integral (x sigma_yz - y sigma_xz) dA (uniform once the plastic front has
passed and the profile is steady).
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "Tools", "PostProcess"))
from exagoop_columns import build_field_dict

C = build_field_dict(3, True)
OMEGA = 2.0 * np.pi
ROOT = "Solution/ascii_files/3D_Twisted_Column_JC"

def load(f):
    d = np.loadtxt(f, skiprows=5)
    assert d.shape[1] == C["ncols"], f"{d.shape[1]} cols, layout predicts {C['ncols']}"
    return d

def growth(files):
    print(" t[ms]  |vz|max  z@max  n(|vz|>5)   T_max   ep_max")
    for f in files:
        d = load(f); t = float(f.split("_t")[-1]); vz = d[:, C["zvel"]]; i = np.argmax(np.abs(vz))
        print(f"{t:6.3f} {np.abs(vz).max():8.2f} {d[i,2]:6.1f} {(np.abs(vz)>5).sum():8d} {d[:,C['temperature']].max():8.1f} {d[:,C['isv_0']].max():8.3f}")

def profile(f):
    d = load(f); t = float(f.split("_t")[-1])
    x, y, z = d[:, 0], d[:, 1], d[:, 2]; V = d[:, C["volume"]]
    ep, T = d[:, C["isv_0"]], d[:, C["temperature"]]
    vx, vy = d[:, C["xvel"]], d[:, C["yvel"]]
    sxx, sxy, sxz, syy, syz, szz = d[:, C["stress"]:C["stress"] + 6].T   # 3-D Voigt order
    svm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2))
    print(f"\n=== {os.path.basename(f)}  t = {t:.3f} ms  N = {len(z)}  T[min,max] = [{T.min():.2f}, {T.max():.1f}]")
    print("  z-band     omega/OM (uniform: z/L)  ep_med  ep_max   T_med   T_max  |szz|_med  svm_med   Mz")
    for zlo, zhi in [(99,100),(98,99),(96,98),(90,96),(80,90),(60,80),(40,60),(20,40),(5,20),(0,5)]:
        s = (z >= zlo) & (z < zhi)
        cx, cy = x[s].mean(), y[s].mean(); X, Y = x[s] - cx, y[s] - cy; r = np.hypot(X, Y)
        om = (-Y * vx[s] + X * vy[s]) / np.maximum(r * r, 1e-9); so = r > 3
        Mz = ((x[s] * syz[s] - y[s] * sxz[s]) * V[s]).sum() / (zhi - zlo)
        print(f"  [{zlo:3d},{zhi:3d})   {np.median(om[so])/OMEGA:6.3f}  ({(zlo+zhi)/200:.2f})   {np.median(ep[s]):.3f}   {ep[s].max():.3f}   {np.median(T[s]):6.1f}  {T[s].max():6.1f}   {np.median(np.abs(szz[s])):.4f}   {np.median(svm[s]):.4f}   {Mz:7.2f}")

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(ROOT, "matpnt_t*")))
    if not files:
        sys.exit(f"no frames under {ROOT}")
    if len(sys.argv) > 2 and sys.argv[1] == "--profile":
        want = float(sys.argv[2])
        f = min(files, key=lambda p: abs(float(p.split("_t")[-1]) - want))
        profile(f)
    else:
        growth(files)
