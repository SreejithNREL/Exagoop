# Test Case: 1D Heat Conduction with Convective (Robin) Boundary Condition

## Physical Problem

A one-dimensional slab ($x \in [0, 1]$) is initially at uniform temperature $T = 0$. The left end is held at a fixed temperature $T(0,t) = 1$ (Dirichlet). The right end loses heat to an ambient environment at $T_\infty = 0$ via Newton's law of cooling:

$$-k \frac{\partial T}{\partial x}\bigg|_{x=1} = h\,(T(1,t) - T_\infty)$$

with convective coefficient $h = 2$ and conductivity $k = 1$. This test verifies the implementation of the Robin (mixed/convective) thermal boundary condition, exercised via the UDF `UDF/udf_temp_convective.cpp`.

## Exact Solution

Decompose $T = T_{ss}(x) + w(x,t)$ where $T_{ss}$ satisfies the steady-state problem. With homogeneous Robin BC at $x = L$:

$$T_{ss}(x) = T_{wall} + (T_\infty - T_{wall})\frac{h/k}{1 + hL/k}\,x$$

The transient part $w$ satisfies the heat equation with homogeneous BCs:

$$w(0,t) = 0, \quad -k\,\frac{\partial w}{\partial x}(L,t) = h\,w(L,t)$$

The eigenfunctions are $\sin(\lambda_n x)$ where $\lambda_n$ are roots of:

$$\lambda_n \cos(\lambda_n L) + \frac{h}{k}\sin(\lambda_n L) = 0$$

The full solution is:

$$T(x,t) = T_{ss}(x) + \sum_{n=1}^{N} C_n \sin(\lambda_n x)\,e^{-\lambda_n^2 t}$$

with coefficients $C_n$ determined from the initial condition $w(x,0) = -T_{ss}(x)$.

## Simulation Setup in ExaGOOP

| Parameter | Value |
|-----------|-------|
| Dimensions | 2D (thin slab, effectively 1D in $x$) |
| Domain | $x \in [0,1] \times y \in [0, 0.1]$, 40 × 4 cells |
| Particles per cell | 2 × 2 |
| Constitutive model | Linear elastic, $E = 10^6$, $\nu = 0.3$ |
| Initial temperature | $T = 0$ (uniform) |
| Thermal conductivity | $k = 1$ |
| Specific heat | $c_p = 1$ |
| Stress update scheme | MUSL |
| Fixed timestep | Yes, $\Delta t = 10^{-5}$ s |
| Final time | 2.0 s |
| Output interval | 0.01 s |
| BC (xlo) | `noslip` + Dirichlet $T = 1$ |
| BC (xhi) | `noslip` + Convective: $h = 2$, $T_\infty = 0$ |
| Temperature solver | Enabled |
| Output format | HDF5 |

The domain is nominally 2D but the thin $y$-direction (4 cells, height 0.1) together with the `noslip` BC in $y$ means the solution is effectively 1D in $x$. The convective BC at $x = 1$ is implemented in `UDF/udf_temp_convective.cpp`.

## Key Input Parameters

- **`boundary_conditions.xhi.temp.h`** — convective heat transfer coefficient $h$. Larger $h$ drives the right boundary temperature closer to $T_\infty$ faster.
- **`boundary_conditions.xhi.temp.T_inf`** — ambient temperature $T_\infty$.
- **`boundary_conditions.xlo.temp.T_wall`** — fixed temperature at the heated left end.
- **`temperature.thermcond`** — $k$; the ratio $h/k$ controls the Biot number $\text{Bi} = hL/k$ and thus the character of the steady-state profile.
- **`simulation.fixed_timestep`** / **`simulation.timestep`** — thermal diffusion problems with stiff elasticity benefit from a prescribed $\Delta t$ rather than acoustic CFL.
- **`simulation.num_redist`** — set to 10; redistribution of particles is needed over long runs with stationary material.

## Post-Processing

### Prerequisites

```bash
pip install numpy matplotlib scipy h5py
```

### Step 1 – Generate particles and input file

```bash
bash Generate_MPs_and_InputFiles.sh
```

### Step 2 – Build and run ExaGOOP

The UDF must be compiled before running. Then:

```bash
mkdir -p build && cd build
bash ../cmake_run.sh
mpirun -n 4 ./ExaGOOP2d.*.ex ../Inputs_1DHeatConduction_Convective.inp
```

### Step 3 – Plot temperature profile

```bash
python3 PostProcess/Plot_Temperature.py \
    --folder Solution/ascii_files/<output_tag> \
    --time 0.5
```

### Step 4 – CI validation

```bash
python3 PostProcess/validate.py
```

Reads `matpnt_t0.500000`, computes the eigenvalue expansion exact solution for the Robin BC problem, and prints `PASS` if RMS error < $10^{-2}$.
