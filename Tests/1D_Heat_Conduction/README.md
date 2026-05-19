# Test Case: 1D Heat Conduction (Dirichlet–Dirichlet)

## Physical Problem

A one-dimensional slab of unit length ($x \in [0, 1]$) initially at uniform temperature $T = 0$ evolves under imposed temperature boundary conditions at both ends. The left boundary is held at $T(0,t) = 0$ and the right boundary at $T(1,t) = 1$. As $t \to \infty$ the solution approaches the linear steady state $T = x$. This test verifies the MPM thermal diffusion solver against a classical Fourier-series exact solution with Dirichlet boundary conditions at both ends.

## Exact Solution

The governing equation is the heat equation:

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}, \quad \alpha = \frac{k}{\rho c_p}$$

With $k = 1$, $\rho = 1$ (unit density from particle file), $c_p = 1$, so $\alpha = 1$. The exact transient solution with $T(0,t) = T_0 = 0$, $T(1,t) = T_1 = 1$, and $T(x,0) = 0$ is:

$$T(x,t) = T_0 + (T_1 - T_0)x + 2(T_1 - T_0) \sum_{n=1}^{\infty} \frac{(-1)^n}{n\pi} e^{-(n\pi)^2 t} \sin(n\pi x)$$

The series converges rapidly for $t > 0.01$; $N = 100$ terms are sufficient at $t = 0.05$.

## Simulation Setup in ExaGOOP

| Parameter | Value |
|-----------|-------|
| Dimensions | 1D (config uses `dimensions: 1`) |
| Domain | $x \in [0, 1]$, 100 cells |
| Particles per cell | 2 |
| Constitutive model | Linear elastic, $E = 10^6$, $\nu = 0.3$ (stiff; solid mechanics inactive) |
| Initial velocity | Zero (stationary) |
| Initial temperature | $T = 0$ (uniform) |
| Thermal conductivity | $k = 1$ |
| Specific heat | $c_p = 1$ |
| Stress update scheme | MUSL |
| Order scheme | 1 |
| PIC/FLIP | Pure FLIP ($\alpha = 1.0$) |
| CFL | 0.1 |
| Fixed timestep | Yes, $\Delta t = 10^{-5}$ s |
| Final time | 2.0 s |
| Output interval | 0.01 s |
| BC (xlo) | `noslip` + Dirichlet $T = 0$ |
| BC (xhi) | `noslip` + Dirichlet $T = 1$ |
| Temperature solver | Enabled (`use_temp: true`) |
| Output format | HDF5 |

The mechanical degrees of freedom are suppressed (zero velocity, stiff elastic modulus) so the simulation is effectively a pure heat conduction problem. The validation snapshot is taken at $t = 0.05$ s.

## Key Input Parameters

- **`temperature.thermcond`** — thermal conductivity $k$. Increasing this accelerates diffusion (shortens the time to steady state).
- **`temperature.spheat`** — specific heat $c_p$. Increasing this slows thermal equilibration.
- **`boundary_conditions.xlo.temp.T_wall`** and **`xhi.temp.T_wall`** — imposed wall temperatures. The exact solution is parameterised by $T_0$ and $T_1$.
- **`simulation.fixed_timestep`** — set to `1` (true) so that $\Delta t$ is fixed at `timestep = 1e-5` rather than being CFL-limited. Important because the thermal CFL can be much smaller than the mechanical CFL for stiff solids.
- **`simulation.num_redist`** — number of redistribution steps per output interval. Set to 10 here to maintain particle regularity over the long run.
- **`nx`** — number of grid cells (100). Controls spatial resolution of both the grid-based diffusion operator and the particle sampling.

## Post-Processing

### Prerequisites

```bash
pip install numpy matplotlib h5py
```

### Step 1 – Generate particles and input file

```bash
bash Generate_MPs_and_InputFiles.sh
```

### Step 2 – Build and run ExaGOOP

```bash
mkdir -p build && cd build
bash ../cmake_run.sh
mpirun -n 4 ./ExaGOOP1d.*.ex ../Inputs_1DHeatConduction.inp
```

### Step 3 – Plot temperature profile

```bash
python3 PostProcess/Plot_Temperature.py \
    --folder Solution/ascii_files/<output_tag> \
    --time 0.05
```

Plots the simulated temperature profile alongside the exact Fourier-series solution at the specified time.

### Step 4 – CI validation

```bash
python3 PostProcess/validate.py
```

Reads `matpnt_t0.050000`, computes the RMS error against the exact solution at $t = 0.05$, and prints `PASS` if below $10^{-2}$.
