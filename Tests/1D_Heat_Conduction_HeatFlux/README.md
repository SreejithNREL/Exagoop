# Test Case: 1D Heat Conduction with Prescribed Heat Flux (Neumann BC)

## Physical Problem

A one-dimensional slab ($x \in [0, 1]$) starts at uniform temperature $T = 0$. The left end is held at $T(0,t) = 0$ (Dirichlet). The right end is subject to a prescribed heat flux into the domain:

$$-k \frac{\partial T}{\partial x}\bigg|_{x=1} = -q, \quad q = 1.0 \text{ W/m}^2$$

with $k = 1$ W/(m·K). This test verifies the Neumann thermal boundary condition implementation, which is exercised via the pair of UDF files in `UDF/`.

## Exact Solution

The problem is solved by separation of variables. Decompose $T = T_s(x) + v(x,t)$ where the steady state is $T_s(x) = (q/k)\,x = x$. The transient part $v$ satisfies the heat equation with $v(0,t) = 0$ (Dirichlet) and $\partial_x v(1,t) = 0$ (Neumann), giving eigenfunctions:

$$\phi_n(x) = \sin(\lambda_n x), \quad \lambda_n = \frac{(2n-1)\pi}{2L}$$

The exact solution is:

$$T(x,t) = \frac{q}{k}\,x + \sum_{n=1}^{\infty} C_n \sin(\lambda_n x)\,e^{-\lambda_n^2 t}$$

with $C_n = 2(-1)^n / (L\,\lambda_n^2)$ (derived analytically using $\cos(\lambda_n L) = 0$, $\sin(\lambda_n L) = (-1)^{n+1}$). Reference: Carslaw & Jaeger, *Conduction of Heat in Solids*, §3.4.

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
| BC (xlo) | `noslip` + Dirichlet $T = 0$ |
| BC (xhi) | `noslip` + Heat flux $q = 1.0$ |
| Temperature solver | Enabled |
| UDFs | `UDF/udf_temp_dirichlet.cpp`, `UDF/udf_temp_heatflux.cpp` |
| Output format | HDF5 |

The two UDF files respectively implement the Dirichlet condition at $x = 0$ and the Neumann (flux) condition at $x = 1$.

## Key Input Parameters

- **`boundary_conditions.xhi.temp.flux`** — the prescribed normal heat flux $q$ (positive = heat entering the domain). The steady-state temperature gradient equals $q/k$.
- **`boundary_conditions.xlo.temp.T_wall`** — Dirichlet temperature at the cold end.
- **`temperature.thermcond`** — $k$; sets both the diffusivity and the steady-state gradient $q/k$.
- **`simulation.fixed_timestep`** / **`simulation.timestep`** — fix the time step to decouple thermal from mechanical CFL. Essential when the elastic modulus is large.
- **`simulation.final_time`** — 2.0 s; at this time the solution is close to steady state. Validation is performed at $t = 0.5$ s where the Fourier series converges well.
- **`nx`** — grid resolution (40 cells). Controls accuracy of the flux BC gradient evaluation.

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

Compile the UDFs first, then:

```bash
cd $MPM_HOME/Build_Gnumake/
#make necessary changes in GNUmakefile
make -j
cd ../Tests/1D_Heat_Conduction_HeatFlux
cp ../../Build_Gnumake/ExaGOOP1d.*.ex .
./ExaGOOP1d.*.ex Inputs_1DHeatConduction_HeatFlux.inp
mpirun -n 4 ./ExaGOOP1d.*.ex ../Inputs_1DHeatConduction_HeatFlux.inp
```

### Step 3 – Plot temperature profile

```bash
python3 PostProcess/Plot_Temperature.py \
    --fileloc Solution/ascii_files/<output_tag> \
    --time 2.0
```

Plots the simulated temperature profile against the exact Carslaw–Jaeger solution at $t = 2.0$ s.

### Step 4 – CI validation

```bash
python3 PostProcess/validate.py
```

Reads `matpnt_t0.500000` and prints `PASS` if the RMS error against the exact solution is below $10^{-2}$.
