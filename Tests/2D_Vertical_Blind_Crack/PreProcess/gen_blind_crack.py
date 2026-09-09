#!/usr/bin/env python3
"""
Vertical blind (finger) crack in a single elastic block: particle + input
generator. Wraps Tools/Preprocess/Generate_MPs_Inputfile_Generic.py and adds
one 2-D shape type, "block_with_slot":

    {"type": "block_with_slot",
     "xmin", "xmax", "ymin", "ymax",        # the block
     "x_center": ..,                        # slot centreline
     "y_mouth": .., "y_tip": ..,            # slot runs from y_mouth (open end) to y_tip (blind end)
     "halfwidth": ..,                       # half-width at the mouth; tapers as sqrt(1 - t^2) to the tip
     "roughness": {"amp": [..], "waves": [..], "phase_left": [..], "phase_right": [..]}}

A point is in the body if it is inside the block and NOT inside the slot.
The slot is one cut in one body, so the rock above the tip is continuous.

Usage: python3 PreProcess/gen_blind_crack.py --config PreProcess/config.json
"""
import os, math, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
GENERIC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "Tools", "Preprocess",
                                        "Generate_MPs_Inputfile_Generic.py"))
spec = importlib.util.spec_from_file_location("gen_generic", GENERIC)
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


class BlockWithSlot(G.ShapeBase):
    def __init__(self, c):
        self.xmin, self.xmax, self.ymin, self.ymax = c["xmin"], c["xmax"], c["ymin"], c["ymax"]
        self.xc, self.y0, self.y1, self.hw = c["x_center"], c["y_mouth"], c["y_tip"], c["halfwidth"]
        r = c.get("roughness", {"amp": [], "waves": [], "phase_left": [], "phase_right": []})
        self.amp, self.waves = r["amp"], r["waves"]
        self.phl, self.phr = r.get("phase_left", [0.0] * len(self.amp)), r.get("phase_right", [0.0] * len(self.amp))

    def _rough(self, t, ph):
        return sum(a * math.sin(2 * math.pi * n * t + p) for a, n, p in zip(self.amp, self.waves, ph))

    def walls(self, y):
        """left and right wall x at height y (None above the tip)."""
        t = (y - self.y0) / (self.y1 - self.y0)
        if t < 0 or t > 1:
            return None
        w = self.hw * math.sqrt(max(0.0, 1.0 - t * t))
        return (self.xc - w - self._rough(t, self.phl), self.xc + w + self._rough(t, self.phr))

    def contains(self, p):
        x, y = p[0], p[1]
        if not (self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax):
            return False
        wl = self.walls(y)
        if wl is None:
            return True
        return not (wl[0] < x < wl[1])


_orig = G.make_shape


def make_shape(cfg, dim):
    if cfg is not None and cfg.get("type") == "block_with_slot":
        if dim != 2:
            G.die("block_with_slot is 2-D only")
        return BlockWithSlot(cfg)
    return _orig(cfg, dim)


G.make_shape = make_shape

if __name__ == "__main__":
    G.main()
