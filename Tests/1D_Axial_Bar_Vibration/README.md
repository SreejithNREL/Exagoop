# Test Case: 1D Axial Bar Vibration

## Physical Problem

A one-dimensional elastic bar of length $L = 25$ m is fixed at both ends (no-slip BCs at $x = 0$ and $x = L$). The bar is given an initial velocity distribution corresponding to the first axial mode shape and subsequently undergoes free longitudinal vibration with no gravity, no thermal coupling, and no dissipation. This is a classical benchmark for verifying elastic wave propagation and long-time energy conservation in MPM.

## Exact Solution

The governing equation for axial vibration is the 1D wave equation:

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}, \quad c = \sqrt{E/\rho}$$

For a bar fixed at $x = 0$ and $x = L$ with initial velocity $v(x,0) = V_0 \sin(\beta_1 x)$, where $\beta_1 = \pi/(2L)$ (fundamental mode, $n=1$), the exact velocity is:

$$v(x, t) = V_0 \sin\!\left(\frac{\pi x}{2L}\right) \cos(\omega_1 t), \quad \omega_1 = \frac{\pi}{2L}\sqrt{\frac{E}{\rho}}$$

With the default parameters ($E = 100$, $\rho = 1$, $L = 25$, $V_0 = 0.1$), the fundamental frequency is $\omega_1 = \pi/50$ rad/s and the period is $T = 100$ s. The simulation runs to $t = 50$ s (one half-period).

## Simulation Setup in ExaGOOP

| Parameter | Value |
|-----------|-------|
| Dimensions | 1D |
| Domain | $x \in [0, 30]$, 30 cells |
| Body extent | $x \in [0, 25]$ |
| Particles per cell | 2 |
| Constitutive model | Linear elastic, $E = 100$, $\nu = 0$ |
| Initial velocity | Sinusoidal mode shape via `PreProcess/velocity_udf.py` |
| Stress update scheme | MUSL |
| Order scheme | 1 (linear basis functions) |
| PIC/FLIP blending | Pure FLIP ($\alpha = 1.0$) |
| CFL | 0.1 |
| Final time | 50 s |
| Output interval | 0.5 s |
| BCs (xlo, xhi) | `noslip` (clamped ends) |
| Temperature | Disabled |
| Output format | HDF5 |

The initial velocity field is set via the UDF `PreProcess/velocity_udf.py`, which evaluates $v_x = V_0 \sin(\beta_1 x)$ at each particle position at $t=0$. The domain extends slightly beyond the bar ($[25, 30]$) to provide empty buffer cells; no material points are placed there.

## Key Input Parameters

- **`E`** — Young's modulus in `bodies[0].constitutive_model`. Controls wave speed $c = \sqrt{E/\rho}$ and vibration frequency.
- **`CFL`** — time step multiplier. Reduce for finer time resolution; 0.1 is a conservative value for this problem.
- **`alpha_pic_flip`** — PIC/FLIP blending coefficient. `1.0` = pure FLIP (exact energy conservation). Values below 1 introduce dissipation via PIC damping.
- **`order_scheme`** — `1` = linear shape functions. Higher orders reduce numerical diffusion but require more particles per cell.
- **`stress_update_scheme`** — `MUSL` (Modified Update Stress Last). Standard choice for elastic problems.
- **`write_output_time`** — snapshot frequency (0.5 s here); reduce for finer time resolution of diagnostics.
- **`diagnostics.do_calculate_tke_tse`** — `1` enables per-step output of total kinetic and strain energy; used by `PostProcess/plot_energy.py`.
- **Body `x_end`** (`x_end = 25.0`) — bar length. Cells from 25 to 30 are empty buffer cells required by the BC implementation.

## Post-Processing

### Prerequisites

```bash
pip install numpy matplotlib h5py
```

### Step 1 – Generate particles and input file

```bash
bash Generate_MPs_and_InputFiles.sh
```

This runs `PreProcess/Generate_MPs_Inputfile_Generic.py --config PreProcess/config.json` and writes `mpm_particles.h5` (particle file) and the ExaGOOP `.inp` file.

### Step 2 – Build and run ExaGOOP

Configure and build using `cmake_run.sh` from a build directory, then run:

```bash
mkdir -p build && cd build
bash ../cmake_run.sh          # configures and compiles
mpirun -n 4 ./ExaGOOP1d.*.ex ../Inputs_1DAxialBarVibration.inp
```

Output snapshots are written under `Solution/` in the directory named by `output_tag` in the config.

### Step 3 – Compute velocity error vs exact solution

```bash
python3 PostProcess/Calculate_Error.py \
    --folder Solution/ascii_files/<output_tag> \
    --time 50.0 \
    --dim 1 \
    --showplot True
```

Reports min, max, and RMS absolute error of $v_x$ against the exact solution at the requested time, and optionally displays a numerical vs exact overlay plot.

### Step 4 – CI validation

```bash
python3 PostProcess/validate.py
```

Searches for `matpnt_t50.000000` in the solution directory and prints `PASS` if the RMS error is below $10^{-1}$.

### Step 5 – Energy and velocity diagnostics

```bash
python3 PostProcess/plot_energy.py      # TKE and TSE vs time
python3 PostProcess/plot_vel.py         # velocity snapshots
python3 PostProcess/AnimateVelocity.py  # animated velocity field
```

For a pure-FLIP run, total mechanical energy (TKE + TSE) should be conserved to within a few percent over the full simulation.
