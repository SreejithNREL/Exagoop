#!/usr/bin/env python3
"""
Plot equivalent plastic strain and temperature for the 2D plate under
gravitational compression, for comparison with Figs. 10.24 and 10.25 of
Nguyen et al.

Two colour scalings are produced for eps_p:
  * the book's full scale (0 .. 1.5), and
  * a clipped scale (0 .. ~95th percentile).
The peak plastic strain is very localized at the two bottom corners, so on the
full scale the arch-shaped shear-band structure saturates to a single colour.
The clipped panel is the one to compare with the book's band pattern.

Usage:
    python3 plot_fields.py [path/to/matpnt_tXXXXXX] [-o out.png]
"""

import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_X, C_Y = 0, 1
C_TEMPERATURE = 43
C_EP = 50

PED_LO, PED_HI = (-2.5, -2.5), (12.5, 0.0)
PLATE = (0.0, 0.0, 10.0, 10.0)


def find_latest(root="Solution/ascii_files/2D_Plate_Under_Gravity"):
    frames = sorted(glob.glob(os.path.join(root, "matpnt_t*")))
    if not frames:
        raise SystemExit(f"[ERROR] no matpnt frames under {root}")
    return frames[-1]


def panel(ax, x, y, v, label, vmax, cmap="jet"):
    sc = ax.scatter(x, y, c=v, s=4, cmap=cmap, vmin=0, vmax=vmax, linewidths=0)
    ax.add_patch(plt.Rectangle(PED_LO, PED_HI[0] - PED_LO[0],
                               PED_HI[1] - PED_LO[1],
                               fc="0.8", ec="0.4", hatch="//", zorder=0))
    x0, y0, w, h = PLATE
    ax.plot([x0, x0 + w, x0 + w, x0, x0], [y0, y0, y0 + h, y0 + h, y0],
            "k--", lw=0.8, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-3.0, 13.0)
    ax.set_ylim(-2.6, 10.6)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(label, fontsize=10)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)


def main():
    argv = sys.argv[1:]
    out = "plate_fields.png"
    if "-o" in argv:
        k = argv.index("-o")
        out = argv[k + 1]
        argv = argv[:k] + argv[k + 2:]
    args = [a for a in argv if not a.startswith("-")]
    path = args[0] if args else find_latest()

    d = np.loadtxt(path, skiprows=5)
    x, y = d[:, C_X], d[:, C_Y]
    ep, T = d[:, C_EP], d[:, C_TEMPERATURE]
    tstr = os.path.basename(path).split("_t")[-1]

    ep_clip = float(np.percentile(ep, 95))

    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    panel(axs[0], x, y, ep, "Equivalent plastic strain (book scale 0-1.5)", 1.5)
    panel(axs[1], x, y, ep,
          f"Equivalent plastic strain (clipped 0-{ep_clip:.2f})", ep_clip)
    panel(axs[2], x, y, T, "Temperature T (C)", float(T.max()))
    fig.suptitle(
        f"ExaGOOP - 2D plate under gravitational compression, t = {tstr} ms   "
        "(dashed: initial 10x10 mm plate; hatched: static level-set pedestal)",
        fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
