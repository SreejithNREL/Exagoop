#!/usr/bin/env python3
"""
Crack-shape analysis for the vertical blind crack under vertical load.

Per matpnt_t* dump:
  * crack walls: bin rock particles (material 0) in y (one grid cell per bin);
    in each bin, within a window |x - x_center| < window, the left wall is
    the largest x below the centreline and the right wall the smallest x above
    it. Aperture b(y) = (xR - xL) - s (particle spacing), clipped at 0; the
    crack is "closed" where b < 0.5 s.  Crack length = highest open bin.
  * sigma_yy (compression positive) averaged over rock far from the crack
    (|x - xc| > 0.25) in a band above the tip; sigma_xx the same way, so the
    ratio can be compared with nu/(1-nu) (plane strain, rollers).
  * max principal stress sigma_1 per particle (tension positive) for the
    tip-region map.
Outputs: crack_shapes.png (outlines at several loads), aperture_profiles.png,
         tip_stress.png, crack_data.csv
Usage: python3 PostProcess/crack_shape_analysis.py [--tag TAG] [--xc 0.5] [--ytip 0.55]
"""
import argparse, glob, os, re, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "Tools", "PostProcess"))
from exagoop_columns import build_field_dict
DIM, USE_TEMP = 2, False


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="2D_Vertical_Blind_Crack_coarse")
    p.add_argument("--folder", default=None)
    p.add_argument("--xc", type=float, default=0.5)
    p.add_argument("--ytip", type=float, default=0.55)
    p.add_argument("--window", type=float, default=0.18)
    p.add_argument("--ny", type=int, default=70); p.add_argument("--ymin", type=float, default=0.0); p.add_argument("--ymax", type=float, default=1.4)
    p.add_argument("--nx", type=int, default=50); p.add_argument("--ppc", type=int, default=2)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--nshapes", type=int, default=5)
    p.add_argument("--tmax", type=float, default=1e9)
    return p.parse_args()


def tfromname(f):
    m = re.search(r"matpnt_t([0-9.]+)", f); return float(m.group(1)) if m else np.nan


def analyse(fname, a, F):
    d = np.loadtxt(fname, skiprows=5)
    x, y = d[:, F["posx"]], d[:, F["posy"]]
    mid = d[:, F["material_indx"]].astype(int)
    sxx, sxy, syy = (d[:, F["stress"] + k] for k in (0, 1, 2))
    rock = mid == 0
    dy = (a.ymax - a.ymin) / a.ny; dx = 1.0 / a.nx; s = dx / a.ppc
    edges = np.arange(a.ymin, a.ytip + 3 * dy, dy)
    nb = len(edges) - 1
    ib = np.clip(np.digitize(y, edges) - 1, 0, nb - 1)
    near = rock & (np.abs(x - a.xc) < a.window) & (y < edges[-1])
    xL = np.full(nb, np.nan); xR = np.full(nb, np.nan)
    for k in range(nb):
        m = near & (ib == k)
        l = m & (x < a.xc); r = m & (x >= a.xc)
        if l.any(): xL[k] = x[l].max()
        if r.any(): xR[k] = x[r].min()
    b = np.clip(xR - xL - s, 0, None)
    yc = 0.5 * (edges[1:] + edges[:-1])
    openb = b > 0.5 * s
    length = yc[openb].max() if openb.any() else 0.0
    # far-field stresses: rock, |x-xc|>0.25, y in [0.65, 0.9] (above the tip, below the platen)
    ff = rock & (np.abs(x - a.xc) > 0.25) & (y > 0.65) & (y < 0.9)
    syy_ff = -syy[ff].mean(); sxx_ff = -sxx[ff].mean()
    # max principal stress (tension positive) for all rock particles
    sm = 0.5 * (sxx + syy); R = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy ** 2)
    s1 = sm + R
    return dict(yc=yc, xL=xL, xR=xR, b=b, length=length, syy=syy_ff, sxx=sxx_ff,
                x=x, y=y, rock=rock, s1=s1, syy_p=syy, mouth=b[0] if nb else np.nan, bmean=np.nanmean(b[yc < a.ytip]))


def main():
    a = parse()
    folder = a.folder or os.path.join("Solution", "ascii_files", a.tag)
    files = sorted(glob.glob(os.path.join(folder, "matpnt_t*")), key=tfromname)
    files = [f for f in files if tfromname(f) <= a.tmax]
    if not files: sys.exit(f"no matpnt files in {folder}")
    F = build_field_dict(DIM, USE_TEMP)
    res = []
    for f in files:
        r = analyse(f, a, F); r["t"] = tfromname(f); res.append(r)
        print(f"t={r['t']:7.2f}  syy_ff={r['syy']:8.3f}  sxx_ff={r['sxx']:8.3f}  ratio={r['sxx']/max(r['syy'],1e-12):6.3f}  "
              f"(nu/(1-nu)={a.nu/(1-a.nu):.3f})  mouth b={r['mouth']:.4f}  <b>={r['bmean']:.4f}  open length={r['length']:.3f}")
    rows = np.array([[r["t"], r["syy"], r["sxx"], r["mouth"], r["bmean"], r["length"]] for r in res])
    np.savetxt("crack_data.csv", rows, delimiter=",", header="t,sigma_yy_farfield,sigma_xx_farfield,mouth_aperture,mean_aperture,open_length", comments="")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    idx = np.unique(np.linspace(0, len(res) - 1, a.nshapes).astype(int))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(idx)))
    # --- crack outlines overlaid
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for c, i in zip(cols, idx):
        r = res[i]; lab = f"t={r['t']:.1f}, σyy={r['syy']:.1f}"
        ax[0].plot(r["xL"], r["yc"], color=c, lw=1.8, label=lab); ax[0].plot(r["xR"], r["yc"], color=c, lw=1.8)
        ax[1].plot(r["b"], r["yc"], color=c, lw=1.8, label=lab)
    ax[0].set_aspect("equal"); ax[0].set_xlim(a.xc - 0.15, a.xc + 0.15); ax[0].set_ylim(0, a.ytip + 0.1)
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y"); ax[0].set_title("crack walls under increasing vertical load"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("aperture b(y)"); ax[1].set_ylabel("y"); ax[1].set_title("aperture profile"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("crack_shapes.png", dpi=140)
    # --- filled shapes side by side (the 'morphology' figure)
    fig, axs = plt.subplots(1, len(idx), figsize=(2.6 * len(idx), 5), squeeze=False)
    for j, i in enumerate(idx):
        r = res[i]; axx = axs[0, j]
        m = r["rock"] & (np.abs(r["x"] - a.xc) < 0.2) & (r["y"] < a.ytip + 0.15)
        axx.scatter(r["x"][m], r["y"][m], s=3, c="#c9b79c")
        ok = ~np.isnan(r["xL"]) & ~np.isnan(r["xR"])
        axx.fill_betweenx(r["yc"][ok], r["xL"][ok], r["xR"][ok], where=r["b"][ok] > 0, color="#5dade2", alpha=.8)
        axx.set_aspect("equal"); axx.set_xlim(a.xc - 0.2, a.xc + 0.2); axx.set_ylim(0, a.ytip + 0.15)
        axx.set_title(f"t={r['t']:.1f}\nσyy={r['syy']:.1f}, L_open={r['length']:.2f}", fontsize=9)
    fig.tight_layout(); fig.savefig("crack_morphology.png", dpi=140)
    # --- tip stress map at the last frame and at ~mid load
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    for axx, i in zip(axs, (idx[len(idx) // 2], idx[-1])):
        r = res[i]; m = r["rock"] & (np.abs(r["x"] - a.xc) < 0.3) & (r["y"] < a.ytip + 0.35)
        sc = axx.scatter(r["x"][m], r["y"][m], c=r["s1"][m], s=9, cmap="RdBu_r", vmin=-np.nanmax(np.abs(r["s1"][m])), vmax=np.nanmax(np.abs(r["s1"][m])))
        axx.set_aspect("equal"); axx.set_title(f"max principal stress σ1 (tension +), t={r['t']:.1f}, σyy={r['syy']:.1f}", fontsize=9)
        plt.colorbar(sc, ax=axx, fraction=0.046)
    fig.tight_layout(); fig.savefig("tip_stress.png", dpi=140)
    # --- stress ratio and closure vs load
    t, syy, sxx, mouth, bm, L = rows.T
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(syy, sxx, "o-", label="MPM far field"); ax[0].plot(syy, a.nu / (1 - a.nu) * syy, "k--", label=f"ν/(1−ν)·σyy = {a.nu/(1-a.nu):.2f}·σyy")
    ax[0].set_xlabel("σyy (far field)"); ax[0].set_ylabel("σxx (far field)"); ax[0].legend(); ax[0].set_title("lateral confinement check")
    ax[1].plot(syy, mouth / mouth[0], "o-", label="mouth aperture / initial"); ax[1].plot(syy, bm / bm[0], "s-", label="mean aperture / initial"); ax[1].plot(syy, L / L[0], "^-", label="open length / initial")
    ax[1].set_xlabel("σyy (far field)"); ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=8); ax[1].set_title("crack closure")
    fig.tight_layout(); fig.savefig("crack_closure.png", dpi=140)
    print("wrote crack_data.csv, crack_shapes.png, crack_morphology.png, tip_stress.png, crack_closure.png")


if __name__ == "__main__":
    main()
