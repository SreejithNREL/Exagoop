import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import re
import os

# ---------------------------------------------------------
# Extract time from filename "matpnt_t0.030000"
# ---------------------------------------------------------
def extract_time(fname):
    match = re.search(r"matpnt_t([0-9.]+)", fname)
    return float(match.group(1)) if match else None

# ---------------------------------------------------------
# Exact solution for vibrating string
# ---------------------------------------------------------
def exact_solution(x, t, v0=0.1, L=1.0, E=50, rho=25*2):
    omega =  np.sqrt(9.8)*np.sqrt(E / rho)    
    return (v0 * L / np.pi) * np.sqrt(rho / E) * np.sin(omega * t) * np.sin(np.pi * x / L)

# ---------------------------------------------------------
# Load all ExaGOOP particle files
# ---------------------------------------------------------
folder = "Solution/ascii_files"
files = glob.glob(os.path.join(folder, "matpnt_t*"))
files = sorted(files, key=extract_time)

print(f"Found {len(files)} files")

frames = []
times = []

for f in files:
    t = extract_time(f)
    times.append(t)

    data = np.loadtxt(f, skiprows=5)
    x = data[:, 0]
    y = data[:, 1]

    frames.append((x, y))

# ---------------------------------------------------------
# Prepare exact solution grid
# ---------------------------------------------------------
x_exact = np.linspace(0, 1.0, 400)

# ---------------------------------------------------------
# Animation setup
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

# Determine axis limits
all_x = np.concatenate([fr[0] for fr in frames])
all_y = np.concatenate([fr[1] for fr in frames])

ax.set_xlim(all_x.min(), all_x.max())
ax.set_ylim(all_y.min(), all_y.max())

# Numerical scatter
scat = ax.scatter([], [], s=10, color="blue", label="Numerical (MPM)")

# Exact solution line
exact_line, = ax.plot([], [], "r-", linewidth=2, label="Exact")

title = ax.set_title("")
ax.legend()

# ---------------------------------------------------------
# Update function
# ---------------------------------------------------------
def update(frame_idx):
    x, y = frames[frame_idx]
    t = times[frame_idx]

    # Update numerical points
    scat.set_offsets(np.column_stack((x, y)))

    # Update exact curve
    y_exact = exact_solution(x_exact, t)
    exact_line.set_data(x_exact, y_exact)

    title.set_text(f"Time = {t:.6f}")
    return scat, exact_line, title

# ---------------------------------------------------------
# Build animation
# ---------------------------------------------------------
ani = animation.FuncAnimation(
    fig, update, frames=len(frames), interval=80, blit=True
)

plt.show()

