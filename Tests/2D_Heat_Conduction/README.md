# Test Case: 2D Heat Conduction (Dirichlet on All Faces)

## Physical Problem

A two-dimensional square domain $[0,1]^2$ initially at uniform temperature $T = 0$ evolves under Dirichlet temperature boundary conditions $T = 1$ imposed on all four faces. As $t \to \infty$ the solution reaches the uniform steady state $T = 1$ everywhere. The transient behaviour is governed by 2D heat diffusion and provides a stringent test of the MPM thermal solver's ability to handle isotropic diffusion with simultaneous heating from all boundaries.

## Exact Solution

The governing equation is:

$$\frac{\partial T}{\partial t} = \alpha \left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}\right), \quad \alpha = \frac{k}{\rho c_p} = 1$$

With $T = T_1 = 1$ on all four boundaries and $T(x,y,0) = T_0 = 0$, the exact solution is:

$$T(x,y,t) = T_1 + \frac{16(T_0 - T_1)}{\pi^2} \sum_{\substack{i=1,3,5,\ldots \\ j=1,3,5,\ldots}} \frac{1}{ij} e^{-\pi^2(i^2/L^2 + j^2/H^2)\,t} \sin\left(\frac{i\pi x}{L}\right)\sin\left(\frac{j\pi y}{H}\right)$$

with $L = H = 1$. The double series involves only odd indices; 25 terms in each direction are sufficient at $t = 0.05$.

## Simulation Setup in ExaGOOP

| Parameter | Value |
|-----------|-------|
| Dimensions | 2D |
| Domain | $[0,1] \times [0,1]$, 32 × 32 cells |
| Particles per cell | 2 × 2 |
| Constitutive model | Linear elastic, $E = 10^6$, $\nu = 0.3$ |
| Initial velocity | Zero |
| Initial temperature | $T = 0$ (uniform) |
| Thermal conductivity | $k = 1$ |
| Specific heat | $c_p = 1$ |
| Stress update scheme | MUSL |
| Order scheme | 1 |
| Fixed timestep | Yes, $\Delta t = 10^{-5}$ s |
| Final time | 0.05 s |
| Output interval | 0.01 s |
| BC (xlo, xhi) | `noslip` + Dirichlet $T = 1$ |
| BC (ylo, yhi) | `periodic` (momentum) + Dirichlet $T = 1$ |
| Temperature solver | Enabled |
| Output format | HDF5 |

Note that while the momentum BCs in $y$ are periodic (to avoid reflections of the — effectively zero — mechanical waves), the thermal BCs in $y$ are still Dirichlet $T = 1$.

## Key Input Parameters

- **`boundary_conditions.*.temp.T_wall`** — wall temperature on each face. All set to 1.0 here, giving uniform Dirichlet conditions.
- **`temperature.thermcond`** ($k$) and **`temperature.spheat`** ($c_p$) — together define the diffusivity $\alpha = k/(\rho c_p)$. The time to reach steady state scales as $\sim L^2/\alpha$.
- **`simulation.fixed_timestep`** / **`simulation.timestep`** — thermal diffusion stability requires $\Delta t \le h^2 / (2\alpha)$ for explicit methods; fixed $\Delta t = 10^{-5}$ satisfies this for $h = 1/32$.
- **`nx`**, **`ny`** — grid resolution (32 × 32). Finer grids resolve the boundary layers near corners.
- **`ppc`** — 2 × 2 particles per cell. Higher values reduce statistical noise in the temperature field.
- **`simulation.num_redist`** — `1` here; the material is stationary so redistribution is minimal.

## Post-Processing

### Prerequisites

```bash
pip install numpy matplotlib h5py
```

### Step 1 – Generate particles and input file

```bash
bash Generate_MPs_and_InputFiles.sh
```

Also available: `PreProcess/plot_particles.py` for a quick visualisation of the initial particle layout.

### Step 2 – Build and run ExaGOOP

```bash
cd $MPM_HOME/Build_Gnumake/
#make necessary changes in GNUmakefile
make -j
cd ../Tests/2D_Heat_Conduction
cp ../../Build_Gnumake/ExaGOOP1d.*.ex .
./ExaGOOP1d.*.ex Inputs_2DHeatConduction.inp
mpirun -n 4 ./ExaGOOP1d.*.ex ../Inputs_2DHeatConduction.inp
```

### Step 3 – Plot temperature field

```bash
python3 PostProcess/Plot_Temperature.py \
    --folder Solution/ascii_files/<output_tag> \
    --time 0.05 --outputpic <output_pic.png>
```

Produces a 2D colour map of the temperature field at the specified time, with an overlay of the exact solution.

### Step 4 – CI validation

```bash
python3 PostProcess/validate.py
```

Reads `matpnt_t0.050000`, evaluates the double Fourier series exact solution at each particle position, and prints `PASS` if the RMS error is below $5 \times 10^{-2}$.
