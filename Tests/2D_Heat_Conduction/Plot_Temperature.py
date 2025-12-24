import numpy as np
import matplotlib.pyplot as plt

# Parameters
T0 = 0.0
T1 = 1.0
L = 1.0
H = 1.0
t = 0.05
N_terms = 25  # i, j = 1, 3, ..., 49

# Load data, skipping metadata lines
data = np.loadtxt('./Solution/ascii_files/matpnt005000', skiprows=5)

# Extract x, y, and temperature columns (assuming x=0, y=1, T=2)
x = data[:, 0]
y = data[:, 1]
T_num = data[:, 48]

# Compute exact solution
def Texact_2D(x, y, T0, T1, L, H, t, N_terms):
    x = np.asarray(x)
    y = np.asarray(y)
    T_exact = np.full_like(x, T1, dtype=np.float64)
    factor = 16 * (T0 - T1) / (np.pi**2)

    for i in range(1, 2*N_terms, 2):  # odd i
        for j in range(1, 2*N_terms, 2):  # odd j
            decay = np.exp(-np.pi**2 * (i**2 / L**2 + j**2 / H**2) * t)
            term = decay / (i * j) * np.sin(i * np.pi * x / L) * np.sin(j * np.pi * y / H)
            T_exact += factor * term

    return T_exact

T_ex = Texact_2D(x, y, T0, T1, L, H, t, N_terms)
error = (T_num - T_ex) / T_ex

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# Numerical
c1 = axes[0].tricontourf(x, y, T_num, levels=100, cmap='coolwarm')
axes[0].set_title('Numerical Temperature')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
fig.colorbar(c1, ax=axes[0])

# Exact
c2 = axes[1].tricontourf(x, y, T_ex, levels=100, cmap='coolwarm')
axes[1].set_title('Exact Temperature')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
fig.colorbar(c2, ax=axes[1])

# Error
c3 = axes[2].tricontourf(x, y, error, levels=100, cmap='coolwarm')
axes[2].set_title('Relative Error (Numerical - Exact)/Exact')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
fig.colorbar(c3, ax=axes[2])

plt.suptitle('Comparison of Numerical, Exact, and Error Fields', fontsize=16)
plt.show()
