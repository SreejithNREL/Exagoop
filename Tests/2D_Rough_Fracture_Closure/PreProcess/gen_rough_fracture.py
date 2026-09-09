#!/usr/bin/env python3
"""
Rough-walled fracture closure: particle + input-file generator.

Thin wrapper around Tools/Preprocess/Generate_MPs_Inputfile_Generic.py that
registers one extra 2-D shape type, "rough_block", and then hands everything
else (material table, BCs, .inp writing, ascii/hdf5 output) to the generic
preprocessor unchanged.

"rough_block" shape config:
    {"type": "rough_block",
     "xmin": .., "xmax": ..,
     "side": "lower" | "upper",     # lower: y_flat <= y <= profile(x)
                                    # upper: profile(x) <= y <= y_flat
     "y_flat": ..,                  # the flat (loaded / fixed) face
     "y_mean": ..,                  # mean height of the rough face
     "roughness": {                 # sum of sinusoids: amp_k sin(2*pi*n_k*t + phi_k), t=(x-xmin)/(xmax-xmin)
         "amp": [..], "waves": [..], "phase": [..]},
     "rough_offset": 0.0}           # extra constant added to profile (unused normally)

Both faces can share the same "roughness" block (mated, correlated surfaces,
aperture set purely by the y_mean difference) or use different phases /
amplitudes (unmated surfaces, spatially varying aperture — the interesting case).

Usage:  python3 PreProcess/gen_rough_fracture.py --config PreProcess/config.json
"""
import os, sys, math, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
GENERIC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "Tools", "Preprocess",
                                        "Generate_MPs_Inputfile_Generic.py"))
spec = importlib.util.spec_from_file_location("gen_generic", GENERIC)
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def make_profile(cfg):
    xmin, xmax = cfg["xmin"], cfg["xmax"]
    r = cfg["roughness"]
    amp, waves, phase = r["amp"], r["waves"], r.get("phase", [0.0] * len(r["amp"]))
    y0 = cfg["y_mean"] + cfg.get("rough_offset", 0.0)
    L = xmax - xmin

    def prof(x):
        t = (x - xmin) / L
        return y0 + sum(a * math.sin(2.0 * math.pi * n * t + p)
                        for a, n, p in zip(amp, waves, phase))
    return prof


class RoughBlock(G.ShapeBase):
    def __init__(self, cfg):
        self.xmin, self.xmax = cfg["xmin"], cfg["xmax"]
        self.side = cfg["side"]
        self.y_flat = cfg["y_flat"]
        self.prof = make_profile(cfg)

    def contains(self, p):
        x, y = p[0], p[1]
        if not (self.xmin <= x <= self.xmax):
            return False
        yp = self.prof(x)
        if self.side == "lower":
            return self.y_flat <= y <= yp
        return yp <= y <= self.y_flat


_orig_make_shape = G.make_shape


def make_shape(shape_cfg, dimensions):
    if shape_cfg is not None and shape_cfg.get("type") == "rough_block":
        if dimensions != 2:
            G.die("rough_block is 2-D only")
        return RoughBlock(shape_cfg)
    return _orig_make_shape(shape_cfg, dimensions)


G.make_shape = make_shape

if __name__ == "__main__":
    G.main()
