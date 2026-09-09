#!/usr/bin/env python3
"""
Aperture / contact / stress-closure analysis for the 2D rough-fracture
closure case.

For every matpnt_t* dump:
  * bin particles in x (bin = one grid cell); lower rock (material 0) top
    surface = max y per bin, upper rock (material 1) bottom surface = min y
    per bin;  aperture b(x) = (ymin_upper - ymax_lower) - s, where s is the
    particle spacing (centre-to-surface correction), clipped at 0.
  * contact fraction  A_c/A = fraction of bins with b < b_contact
  * mean aperture <b>, closure Db = <b>_0 - <b>, hydraulic aperture
    b_h = <b^3>^(1/3)   (cubic law, k ~ b_h^2)
  * normal stress sigma_n = -<sigma_yy> over a band of lower-rock particles
    next to the fixed base, and the same next to the interface (should agree
    once quasi-static)
Outputs: closure_curve.png, aperture_maps.png, closure_data.csv

Usage: python3 PostProcess/aperture_analysis.py [--tag TAG] [--nx 50] [--ppc 2]
"""
import argparse, glob, os, re, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "Tools", "PostProcess"))
from exagoop_columns import build_field_dict

DIM, USE_TEMP = 2, False


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="2D_Rough_Fracture_Closure_coarse")
    p.add_argument("--folder", default=None, help="override Solution/ascii_files/<tag>")
    p.add_argument("--nx", type=int, default=50)
    p.add_argument("--xmin", type=float, default=0.0)
    p.add_argument("--xmax", type=float, default=1.0)
    p.add_argument("--ppc", type=int, default=2)
    p.add_argument("--base_band", type=float, default=0.15, help="height of band above y=0 for sigma_n")
    p.add_argument("--iface_band", type=float, default=0.10, help="band below lower-rock surface for sigma_n")
    p.add_argument("--nmaps", type=int, default=4)
    return p.parse_args()


def tfromname(f):
    m = re.search(r"matpnt_t([0-9.]+)", f)
    return float(m.group(1)) if m else np.nan


def analyse(fname, a, F):
    d = np.loadtxt(fname, skiprows=5)
    x, y = d[:, F["posx"]], d[:, F["posy"]]
    mid = d[:, F["material_indx"]].astype(int)
    syy = d[:, F["stress"] + 2]          # Voigt 2-D: XX=0, XY=1, YY=2
    dx = (a.xmax - a.xmin) / a.nx
    s = dx / a.ppc                       # particle spacing
    edges = np.linspace(a.xmin, a.xmax, a.nx + 1)
    ib = np.clip(np.digitize(x, edges) - 1, 0, a.nx - 1)

    top0 = np.full(a.nx, np.nan)
    bot1 = np.full(a.nx, np.nan)
    for k in range(a.nx):
        m0 = (ib == k) & (mid == 0)
        m1 = (ib == k) & (mid == 1)
        if m0.any(): top0[k] = y[m0].max()
        if m1.any(): bot1[k] = y[m1].min()
    valid = ~np.isnan(top0) & ~np.isnan(bot1)
    b = np.clip((bot1 - top0) - s, 0.0, None)
    b[~valid] = np.nan
    bv = b[valid]
    b_contact = 0.5 * s
    contact = np.mean(bv < b_contact)
    bmean = bv.mean()
    bh = np.cbrt(np.mean(bv ** 3))

    # normal stress: lower rock near the fixed base, and just below the interface
    band_base = (mid == 0) & (y < a.base_band)
    sig_base = -syy[band_base].mean() if band_base.any() else np.nan
    yi = np.interp(x, 0.5 * (edges[1:] + edges[:-1]), np.nan_to_num(top0, nan=np.nanmean(top0)))
    band_if = (mid == 0) & (y > yi - a.iface_band)
    sig_if = -syy[band_if].mean() if band_if.any() else np.nan

    # cap displacement (mean y of material 2) for a kinematic closure check
    ycap = y[mid == 2].mean() if (mid == 2).any() else np.nan
    xc = 0.5 * (edges[1:] + edges[:-1])
    return dict(b=b, xc=xc, top0=top0, bot1=bot1, bmean=bmean, bh=bh, contact=contact,
                sig_base=sig_base, sig_if=sig_if, ycap=ycap, x=x, y=y, mid=mid, syy=syy)


def bandis(db, kn0, dbmax):
    return kn0 * db / (1.0 - db / dbmax)


def main():
    a = parse()
    folder = a.folder or os.path.join("Solution", "ascii_files", a.tag)
    files = sorted(glob.glob(os.path.join(folder, "matpnt_t*")), key=tfromname)
    if not files:
        sys.exit(f"no matpnt files in {folder}")
    F = build_field_dict(DIM, USE_TEMP)

    rows, res = [], []
    for f in files:
        r = analyse(f, a, F)
        r["t"] = tfromname(f)
        res.append(r)
    b0 = res[0]["bmean"]
    ycap0 = res[0]["ycap"]
    for r in res:
        r["closure"] = b0 - r["bmean"]
        r["cap_disp"] = ycap0 - r["ycap"]
        rows.append([r["t"], r["bmean"], r["bh"], r["closure"], r["cap_disp"],
                     r["contact"], r["sig_base"], r["sig_if"]])
        print(f"t={r['t']:8.3f}  <b>={r['bmean']:.4e}  b_h={r['bh']:.4e}  Db={r['closure']:.4e}  "
              f"u_cap={r['cap_disp']:.4e}  Ac/A={r['contact']:.3f}  "
              f"sig_n(base)={r['sig_base']:.4e}  sig_n(iface)={r['sig_if']:.4e}")
    rows = np.array(rows)
    np.savetxt("closure_data.csv", rows, delimiter=",",
               header="t,b_mean,b_hydraulic,closure,cap_displacement,contact_fraction,sigma_n_base,sigma_n_interface",
               comments="")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; csv written only"); return

    # --- stress-closure curve + hydraulic aperture + contact fraction
    t, bm, bh, db, uc, ac, sb, si = rows.T
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(db, sb, "o-", label=r"$\sigma_n$ (base band)")
    ax[0].plot(db, si, "s--", label=r"$\sigma_n$ (interface band)", alpha=.7)
    ok = (db > 0) & np.isfinite(sb) & (sb > 0)
    if ok.sum() >= 3:
        try:
            from scipy.optimize import curve_fit
            p, _ = curve_fit(bandis, db[ok], sb[ok], p0=[sb[ok][0] / max(db[ok][0], 1e-12), db.max() * 1.2],
                             bounds=([0, db.max() * 1.0001], [np.inf, np.inf]), maxfev=20000)
            xx = np.linspace(0, db.max(), 200)
            ax[0].plot(xx, bandis(xx, *p), "k-", lw=1,
                       label=rf"Bandis fit: $k_{{n0}}$={p[0]:.3g}, $\Delta b_{{max}}$={p[1]:.3g}")
        except Exception as e:
            print("Bandis fit skipped:", e)
    ax[0].set_xlabel(r"closure $\Delta b = \langle b\rangle_0-\langle b\rangle$")
    ax[0].set_ylabel(r"normal stress $\sigma_n$"); ax[0].legend(fontsize=8); ax[0].set_title("stress–closure")
    ax[1].plot(sb, bh / bh[0], "o-", label=r"$b_h/b_{h,0}$")
    ax[1].plot(sb, (bh / bh[0]) ** 2, "s-", label=r"$k/k_0=(b_h/b_{h,0})^2$")
    ax[1].plot(sb, bm / bm[0], "^--", label=r"$\langle b\rangle/\langle b\rangle_0$", alpha=.6)
    ax[1].set_xlabel(r"$\sigma_n$"); ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=8); ax[1].set_title("hydraulic aperture / permeability")
    ax[2].plot(sb, ac, "o-"); ax[2].set_xlabel(r"$\sigma_n$"); ax[2].set_ylabel(r"contact fraction $A_c/A$"); ax[2].set_title("contact area")
    fig.tight_layout(); fig.savefig("closure_curve.png", dpi=140)

    # --- aperture maps at a few load levels
    idx = np.unique(np.linspace(0, len(res) - 1, a.nmaps).astype(int))
    fig, axs = plt.subplots(2, len(idx), figsize=(3.6 * len(idx), 7), squeeze=False)
    for j, i in enumerate(idx):
        r = res[i]
        axx = axs[0, j]
        for k, c in zip((0, 1, 2), ("#b8a88c", "#c9b79c", "#555")):
            m = r["mid"] == k
            axx.scatter(r["x"][m], r["y"][m], s=2, c=c)
        axx.fill_between(r["xc"], r["top0"], r["bot1"], where=r["bot1"] > r["top0"], color="#5dade2", alpha=.6)
        axx.set_aspect("equal"); axx.set_xlim(a.xmin, a.xmax); axx.set_ylim(0.3, 0.8)
        axx.set_title(f"t={r['t']:.2f}  σn={r['sig_base']:.3g}\nAc/A={r['contact']:.2f}", fontsize=9)
        axb = axs[1, j]
        axb.plot(r["xc"], r["b"], "b-")
        axb.fill_between(r["xc"], 0, r["b"], color="#5dade2", alpha=.5)
        axb.set_ylim(0, 1.05 * np.nanmax(res[0]["b"])); axb.set_xlabel("x"); axb.set_ylabel("aperture b(x)")
        axb.set_title(f"<b>={r['bmean']:.3g}, b_h={r['bh']:.3g}", fontsize=9)
    fig.tight_layout(); fig.savefig("aperture_maps.png", dpi=140)
    print("wrote closure_data.csv, closure_curve.png, aperture_maps.png")


if __name__ == "__main__":
    main()
