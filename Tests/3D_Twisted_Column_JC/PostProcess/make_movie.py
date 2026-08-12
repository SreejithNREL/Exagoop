#!/usr/bin/env python3
"""
Animate the 3D twisted column: render every matpnt frame and assemble them into
a GIF (and optionally an MP4), for watching the column twist over time.
Companion to plot_fields.py, which produces static figures.

The colour scale is held FIXED across the whole sequence (that is the point of a
movie -- a per-frame autoscale makes a growing field look constant). By default
the scale is taken from the last frame; use --vmax to pin it to the book's
values (440 MPa for sigma_eq, 76 C for T) when comparing with Fig. 10.28.

Usage:
    python3 make_movie.py                                  # sigma_eq -> twisted_column.gif
    python3 make_movie.py --field T --vmax 76 -o temp.gif
    python3 make_movie.py --field ep --fps 12 --mp4
    python3 make_movie.py --spin 90                         # also orbit the camera

Options:
    --field  sigma_eq | T | ep        quantity to colour by (default sigma_eq)
    --vmax   VALUE                    fixed colour maximum (default: last frame)
    --fps    N                        frames per second (default 8)
    --mp4                             also write an .mp4 next to the .gif (ffmpeg)
    --spin   DEG                      total camera azimuth sweep over the movie
    --shell  P                        keep particles above the P-th radius
                                      percentile so the surface helix is visible
                                      (default 55; use 0 to keep every particle)
    --stride N                        use every N-th frame (default 1)
    -o FILE                           output filename (default twisted_column.gif)

Requires matplotlib + imageio; --mp4 additionally requires ffmpeg.
"""

import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "Solution/ascii_files/3D_Twisted_Column_JC"
C_X, C_Y, C_Z = 0, 1, 2
C_STRESS = 22           # 22..27 = XX, XY, XZ, YY, YZ, ZZ  (3-D Voigt order)
C_TEMPERATURE = 44
C_EP = 51

LABEL = {"sigma_eq": "$\\sigma_{eq}$ (MPa)",
         "T": "T ($^\\circ$C)",
         "ep": "equivalent plastic strain"}


def von_mises_3d(s):
    """3-D Voigt order XX, XY, XZ, YY, YZ, ZZ (2-D differs -- see plot_fields.py)."""
    sxx, sxy, sxz, syy, syz, szz = (s[:, i] for i in range(6))
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)
    )


def field(d, name):
    if name == "sigma_eq":
        return von_mises_3d(d[:, C_STRESS:C_STRESS + 6]) * 1000.0
    if name == "T":
        return d[:, C_TEMPERATURE]
    if name == "ep":
        return d[:, C_EP]
    raise SystemExit(f"[ERROR] unknown field '{name}'")


def opt(argv, flag, default=None, cast=str):
    if flag in argv:
        k = argv.index(flag)
        val = cast(argv[k + 1])
        del argv[k:k + 2]
        return val
    return default


def main():
    argv = sys.argv[1:]
    out = opt(argv, "-o", "twisted_column.gif")
    fname = opt(argv, "--field", "sigma_eq")
    vmax_arg = opt(argv, "--vmax", None, float)
    fps = opt(argv, "--fps", 8, int)
    spin = opt(argv, "--spin", 0.0, float)
    shell = opt(argv, "--shell", 55.0, float)
    stride = opt(argv, "--stride", 1, int)
    want_mp4 = "--mp4" in argv

    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("[ERROR] imageio is required:  pip install imageio")

    fs = sorted(glob.glob(os.path.join(ROOT, "matpnt_t*")),
                key=lambda f: float(f.split("_t")[-1]))[::stride]
    if not fs:
        raise SystemExit(f"[ERROR] no frames under {ROOT}")
    print(f"{len(fs)} frames, field '{fname}'")

    # Fixed colour scale across the sequence -- from --vmax, else the last frame.
    if vmax_arg is not None:
        vmax = vmax_arg
    else:
        vmax = float(field(np.loadtxt(fs[-1], skiprows=5), fname).max())
        vmax = max(vmax, 1e-12)
    print(f"colour scale 0 .. {vmax:.4g} (fixed)")

    tmpdir = ".movie_frames"
    os.makedirs(tmpdir, exist_ok=True)
    pngs = []
    for i, f in enumerate(fs):
        t = float(f.split("_t")[-1])
        d = np.loadtxt(f, skiprows=5)
        x, y, z = d[:, C_X], d[:, C_Y], d[:, C_Z]
        v = field(d, fname)
        if shell > 0:
            r = np.hypot(x, y)
            keep = r > np.percentile(r, shell)
            x, y, z, v = x[keep], y[keep], z[keep], v[keep]

        fig = plt.figure(figsize=(4.2, 8))
        ax = fig.add_subplot(111, projection="3d")
        p = ax.scatter(x, y, z, c=v, cmap="jet", vmin=0, vmax=vmax,
                       s=3, linewidths=0)
        ax.set_box_aspect((1, 1, 4))
        ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.set_zlim(0, 100)
        ax.view_init(elev=12, azim=-60 + spin * i / max(len(fs) - 1, 1))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_zlabel("z (mm)")
        ax.set_title(f"t = {t:.4f} ms\n{LABEL[fname]}   max {v.max():.4g}",
                     fontsize=10)
        fig.colorbar(p, ax=ax, shrink=0.45, pad=0.02)
        plt.tight_layout()
        png = os.path.join(tmpdir, f"f{i:05d}.png")
        fig.savefig(png, dpi=100)
        plt.close(fig)
        pngs.append(png)
        if (i + 1) % 10 == 0 or i == len(fs) - 1:
            print(f"  rendered {i+1}/{len(fs)}")

    imageio.mimsave(out, [imageio.imread(p) for p in pngs],
                    duration=1.0 / fps, loop=0)
    print(f"wrote {out}")

    if want_mp4:
        mp4 = os.path.splitext(out)[0] + ".mp4"
        rc = os.system(
            f'ffmpeg -y -loglevel error -framerate {fps} '
            f'-i {tmpdir}/f%05d.png -pix_fmt yuv420p '
            f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {mp4}')
        print(f"wrote {mp4}" if rc == 0 else "[WARN] ffmpeg failed; GIF still written")

    print(f"(intermediate PNGs left in {tmpdir}/)")


if __name__ == "__main__":
    main()
