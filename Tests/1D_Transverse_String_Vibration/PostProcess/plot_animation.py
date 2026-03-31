import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import re
import os

# ---------------------------------------------------------
# Helper: extract time from filename "matpnt_t0.030000"
# ---------------------------------------------------------
def extract_time(fname):
    match = re.search(r"matpnt_t([0-9.]+)", fname)
    return float(match.group(1)) if match else None

# ---------------------------------------------------------
# Load all files and sort by time
# ---------------------------------------------------------
folder = "Solution/ascii_files"
files = glob.glob(os.path.join(folder, "matpnt_t*"))

# Sort by extracted time
files = sorted(files, key=extract_time)

print(f"Found {len(files)} files")

# ---------------------------------------------------------
# Preload all frames (x,y arrays)
# ---------------------------------------------------------
frames = []
times = []

for f in files:
    t = extract_time(f)
    times.append(t)

    # Skip metadata rows if needed (adjust skiprows)
    data = np.loadtxt(f, skiprows=5)

    x = data[:, 0]
    y = data[:, 1]

    frames.append((x, y))

# ---------------------------------------------------------
# Animation setup
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

# Fix axis limits so the animation doesn't jump
all_x = np.concatenate([fr[0] for fr in frames])
all_y = np.concatenate([fr[1] for fr in frames])

ax.set_xlim(all_x.min(), all_x.max())
ax.set_ylim(all_y.min(), all_y.max())

scat = ax.scatter([], [], s=10)
title = ax.set_title("")

# ---------------------------------------------------------
# Animation update function
# ---------------------------------------------------------
def update(frame_idx):
    x, y = frames[frame_idx]
    scat.set_offsets(np.column_stack((x, y)))
    title.set_text(f"Time = {times[frame_idx]:.6f}")
    return scat, title

# ---------------------------------------------------------
# Build animation
# ---------------------------------------------------------
ani = animation.FuncAnimation(
    fig, update, frames=len(frames), interval=80, blit=True
)

plt.show()

