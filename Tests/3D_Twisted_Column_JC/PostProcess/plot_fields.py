#!/usr/bin/env python3
"""
Plot the 3D twisted column for comparison with Nguyen et al. Figs. 10.27 (FEM)
and 10.28 (ULMPM): von Mises equivalent stress and temperature on the deformed
column.

Book colour scales are sigma_eq 0-440 MPa and T 0-76 C, which are the values
reached at t = 2.99 ms. At early times the fields are far below that, so each
quantity is drawn twice: once on the book scale (for direct comparison with the
figure) and once auto-scaled (to actually see the structure).

Usage:
    python3 plot_fields.py                          # newest frame
    python3 plot_fields.py <matpnt_file> [-o out.png] [--all]
    python3 plot_fields.py --all -o sequence.png    # montage of every frame

Options:
    --all        montage sigma_eq across all available frames (Fig. 10.28 style)
    --shell P    keep only particles with radius above the P-th percentile
                 (default 55) so the surface helix is visible; --shell 0 = all
"""

import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 3D, USE_TEMP=TRUE: col0,1,2 = x,y,z then real[k] = col[k+3].
C_X, C_Y, C_Z = 0, 1, 2
C_STRESS = 22          # 22..27 = XX, XY, XZ, YY, YZ, ZZ   (3-D Voigt order!)
C_TEMPERATURE = 44
C_EP = 51

BOOK_SIGMA_MAX = 440.0   # MPa
BOOK_T_MAX = 76.0        # C
ROOT = "Solution/ascii_files/3D_Twisted_Column_JC"


def von_mises_3d(s):
    """s columns are the 3-D Voigt order XX, XY, XZ, YY, YZ, ZZ.

    NOTE: 2-D uses XX, XY, YY, XZ, YZ, ZZ. Using the 2-D order here silently
    swaps XZ and YY and inflates von Mises by roughly 8x.
    """
    sxx, sxy, sxz, syy, syz, szz = (s[:, i] for i in range(6))
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)
    )


def frames():
    fs = sorted(glob.glob(os.path.join(ROOT, "matpnt_t*")),
                key=lambda f: float(f.split("_t")[-1]))
    if not fs:
        raise SystemExit(f"[ERROR] no frames under {ROOT}")
    return fs


def load(path, shell_pct):
    d = np.loadtxt(path, skiprows=5)
    x, y, z = d[:, C_X], d[:, C_Y], d[:, C_Z]
    svm = von_mises_3d(d[:, C_STRESS:C_STRESS + 6]) * 1000.0   # GPa -> MPa
    T = d[:, C_TEMPERATURE]
    if shell_pct > 0:
        keep = np.hypot(x, y) > np.percentile(np.hypot(x, y), shell_pct)
        x, y, z, svm, T = x[keep], y[keep], z[keep], svm[keep], T[keep]
    return x, y, z, svm, T


def col(ax, x, y, z, v, vmax, title):
    p = ax.scatter(x, y, z, c=v, cmap="jet", vmin=0, vmax=vmax, s=2, linewidths=0)
    ax.set_box_aspect((1, 1, 4))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_zlim(0, 100)
    ax.view_init(elev=12, azim=-60)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    return p


def main():
    argv = sys.argv[1:]
    out = "twisted_fields.png"
    if "-o" in argv:
        k = argv.index("-o")
        out = argv[k + 1]
        argv = argv[:k] + argv[k + 2:]
    shell = 55.0
    if "--shell" in argv:
        k = argv.index("--shell")
        shell = float(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    montage = "--all" in argv
    argv = [a for a in argv if not a.startswith("-")]

    if montage:
        fs = frames()
        step = max(1, len(fs) // 5)
        sel = fs[::step][:5]
        fig = plt.figure(figsize=(3.0 * len(sel), 8))
        for i, f in enumerate(sel):
            x, y, z, svm, T = load(f, shell)
            t = float(f.split("_t")[-1])
            ax = fig.add_subplot(2, len(sel), i + 1, projection="3d")
            p = col(ax, x, y, z, svm, BOOK_SIGMA_MAX,
                    f"t = {t:.3f} ms\n$\\sigma_{{eq}}$ max {svm.max():.0f} MPa")
            if i == len(sel) - 1:
                fig.colorbar(p, ax=ax, shrink=0.5, label="MPa")
            ax = fig.add_subplot(2, len(sel), len(sel) + i + 1, projection="3d")
            p = col(ax, x, y, z, T, BOOK_T_MAX,
                    f"T max {T.max():.2f} C")
            if i == len(sel) - 1:
                fig.colorbar(p, ax=ax, shrink=0.5, label="$^\\circ$C")
        fig.suptitle("ExaGOOP 3D twisted column (JC) — book scales "
                     f"($\\sigma_{{eq}}$ 0-{BOOK_SIGMA_MAX:.0f} MPa, "
                     f"T 0-{BOOK_T_MAX:.0f} C), cf. Fig. 10.28", fontsize=11)
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        print(f"wrote {out}  ({len(sel)} frames)")
        return

    path = argv[0] if argv else frames()[-1]
    t = float(os.path.basename(path).split("_t")[-1])
    x, y, z, svm, T = load(path, shell)
    fig = plt.figure(figsize=(13, 7))
    for i, (v, lab, bookmax) in enumerate(
            [(svm, "$\\sigma_{eq}$ (MPa)", BOOK_SIGMA_MAX),
             (T, "T ($^\\circ$C)", BOOK_T_MAX)]):
        for j, (vm, tag) in enumerate([(bookmax, "book scale"),
                                       (max(v.max(), 1e-12), "auto scale")]):
            ax = fig.add_subplot(2, 2, i * 2 + j + 1, projection="3d")
            p = col(ax, x, y, z, v, vm, f"{lab}  [{tag}, max {v.max():.3g}]")
            fig.colorbar(p, ax=ax, shrink=0.55, pad=0.02)
    fig.suptitle(f"ExaGOOP 3D twisted column (Johnson-Cook), t = {t:.3f} ms   "
                 "|   cf. Nguyen et al. Fig. 10.28", fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
