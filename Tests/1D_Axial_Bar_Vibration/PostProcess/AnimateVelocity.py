import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import re
import os
import sys


# ---------------------------------------------------------
# Extract time from filename "matpnt_t0.030000"
# ---------------------------------------------------------
def extract_time(fname):
    match = re.search(r"matpnt_t([0-9.]+)", fname)
    return float(match.group(1)) if match else None

# ---------------------------------------------------------
# Exact axial vibration velocity
# ---------------------------------------------------------
def exact_velocity(x, t, v0=0.1, L=25.0, E=100, rho=1):
    omega1 = (np.pi / (2 * L)) * np.sqrt(E / rho) 
    return v0 * np.sin(np.pi * x / (2 * L)) * np.cos(omega1 * t)

# ---------------------------------------------------------
# Load all ExaGOOP particle files
# ---------------------------------------------------------
folder = sys.argv[1]
files = glob.glob(os.path.join(folder, "matpnt_t*"))
files = sorted(files, key=extract_time)

print(f"Found {len(files)} files")

frames = []
times = []

for f in files:
    t = extract_time(f)
    times.append(t)

    # Adjust skiprows if needed
    data = np.loadtxt(f, skiprows=5)

    x = data[:, 0]      # x coordinate
    vel_idx=0
    if(sys.argv[2]=='1'):
        vel_idx=2
    if(sys.argv[2]=='2'):
        vel_idx=3        
    if(sys.argv[2]=='3'):
        vel_idx=4    
    vx = data[:, vel_idx]     # x velocity

    frames.append((x, vx))

# ---------------------------------------------------------
# Prepare exact solution grid
# ---------------------------------------------------------
x_exact = np.linspace(0, 25.0, 400)

# ---------------------------------------------------------
# Animation setup
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

# Determine axis limits
all_x = np.concatenate([fr[0] for fr in frames])
all_v = np.concatenate([fr[1] for fr in frames])

ax.set_xlim(all_x.min(), 30)
ax.set_ylim(all_v.min()*1.2, all_v.max()*1.2)

# Numerical scatter
scat = ax.scatter([], [], s=15, color="blue", label="Numerical (MPM)")

# Exact solution line
exact_line, = ax.plot([], [], "r-", linewidth=2, label="Exact")

title = ax.set_title("")
ax.set_xlabel("x")
ax.set_ylabel("Velocity")
ax.legend()

# ---------------------------------------------------------
# Update function
# ---------------------------------------------------------
def update(frame_idx):
    x, vx = frames[frame_idx]
    t = times[frame_idx]

    # Update numerical points
    scat.set_offsets(np.column_stack((x, vx)))

    # Update exact curve
    vx_exact = exact_velocity(x_exact, t)
    exact_line.set_data(x_exact, vx_exact)

    title.set_text(f"Time = {t:.3f} s")
    #title.set_text(f"Time = {t:.6f}")
    return scat, exact_line, title

# ---------------------------------------------------------
# Build animation
# ---------------------------------------------------------
ani = animation.FuncAnimation(
    fig, update, frames=len(frames), interval=80, blit=True
)
ani.save(sys.argv[3], fps=30, dpi=150)


#plt.show()

