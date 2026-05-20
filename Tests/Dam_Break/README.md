# Test Case: 2D Dam Break

## Physical Problem

A column of water initially at rest in the region $[0, 0.1] \times [0, 0.2]$ m collapses under gravity ($g = 9.81$ m/s² downward) in a $0.4 \times 0.4$ m tank with slip (frictionless) walls. The water is modelled as a weakly compressible Newtonian fluid using the Murnaghan–Tait equation of state. As the dam collapses the leading edge of the water front accelerates horizontally; the non-dimensional waterfront position $x^* = x_\text{front}/H$ (with $H = 0.2$ m the initial water height) vs non-dimensional time $t^* = t/\sqrt{H/g}$ is compared against the experimental data of Martin & Moyce (1952).

This is the primary validation case for the ExaGOOP weakly compressible MPM fluid solver.

## Exact Solution

There is no closed-form exact solution for the dam-break problem. Validation is performed by comparing the simulated waterfront position against the Martin & Moyce (1952) experimental dataset. The non-dimensional scaling is:

$$x^* = \frac{x_\text{front}(t)}{H}, \quad t^* = \frac{t}{\sqrt{H/g}}, \quad H = 0.2 \text{ m}$$

The waterfront position $x_\text{front}(t)$ is extracted from the maximum $x$-coordinate of all fluid particles at each output time, tracked via the `do_calculate_minmaxpos` diagnostic.

## Simulation Setup in ExaGOOP

| Parameter | Value |
|-----------|-------|
| Dimensions | 2D |
| Domain | $[0, 0.4] \times [0, 0.4]$ m, 100 × 100 cells |
| Cell size | $\Delta x = \Delta y = 4 \times 10^{-3}$ m |
| Initial water column | $[0, 0.1] \times [0, 0.2]$ m |
| Particles per cell | 1 × 1 |
| Fluid model | Weakly compressible: Murnaghan–Tait EOS |
| Bulk modulus $K$ | 20 000 Pa |
| Pressure exponent $\Gamma$ | 7 |
| Dynamic viscosity $\mu$ | 0.001 Pa·s |
| Density (from particle file) | 997.5 kg/m³ |
| Gravity | $(0, -9.81, 0)$ m/s² |
| Stress update scheme | MUSL |
| Order scheme | 1 |
| PIC/FLIP blending | $\alpha = 0.9$ |
| CFL | 0.1 (variable timestep) |
| Final time | 2.5 s |
| Output interval | 0.01 s |
| BCs (all faces) | `slip` (frictionless walls) |
| Temperature | Disabled |
| Output format | HDF5 |

**Important:** The fluid density must be set to $\rho = 997.5$ kg/m³ in the particle file. If density is accidentally left at the preprocessor default of 1.0 kg/m³ the pressure and wave speed will be entirely wrong and the simulation will give junk results. Always verify the density in the generated particle file before running.

## Key Input Parameters

- **`constitutive_model.Bulk_modulus`** — $K$ in the Tait EOS $p = K[(\rho/\rho_0)^\Gamma - 1]/\Gamma$. Controls the numerical speed of sound $c_s = \sqrt{K/\rho_0}$. Increasing $K$ makes the fluid stiffer (more incompressible) but reduces the allowable CFL timestep.
- **`constitutive_model.Gamma_pressure`** — pressure exponent $\Gamma = 7$ (standard for water). Larger values make the pressure more sensitive to density changes.
- **`constitutive_model.Dynamic_viscosity`** — $\mu$; small viscosity here (0.001 Pa·s ≈ water). Increasing this damps free-surface sloshing.
- **`alpha_pic_flip`** — 0.9 (near-FLIP). Small PIC fraction (0.1) provides just enough numerical dissipation to keep the free surface smooth. Increasing towards 1.0 may lead to pressure noise.
- **`gravity`** — `[0, -9.81, 0]`; must be set correctly (negative y for downward gravity in 2D).
- **`boundary_conditions.*.mom: "slip"`** — frictionless walls. Changing to `noslip` would add wall friction and slow the waterfront.
- **`diagnostics.do_calculate_minmaxpos`** — `1`; essential for waterfront tracking. Writes a diagnostics file with the maximum particle $x$-position at each output step.
- **`simulation.num_redist`** — `1`; particle redistribution is light because the free surface deforms significantly and redistribution after every step is unnecessary.

## Post-Processing

### Prerequisites

```bash
pip install numpy matplotlib
```

### Step 1 – Generate particles and input file

```bash
bash Generate_MPs_and_InputFiles.sh
```

Verify that the generated `mpm_particles.h5` (or `.dat`) has density = 997.5 kg/m³.

### Step 2 – Build and run ExaGOOP

```bash
cd $MPM_HOME/Build_Gnumake/
#make necessary changes in GNUmakefile
make -j
cd ../Tests/Dam_Break
cp ../../Build_Gnumake/ExaGOOP1d.*.ex .
./ExaGOOP1d.*.ex Inputs_DamBreak.inp
mpirun -n 4 ./ExaGOOP1d.*.ex ../Inputs_DamBreak.inp
```

The simulation runs for 2.5 s of physical time and produces output snapshots and a diagnostics file.

### Step 3 – Plot waterfront position vs experiments

```bash
python3 PostProcess/plot_waterfront.py \
    --folder    Solution/ascii_files/<output_tag> \
    --minmaxfile Diagnostics	/diag_<output_tag>/MinMaxPosition.dat \
    --expdata   /path/to/martin_moyce_1952.dat \
    --outputpic waterfront.png \
    --outputmovie dambreak.mp4
```

This script:
1. Creates an animation of the collapsing water column, with particles coloured by speed.
2. Plots the non-dimensionalised waterfront position $x^*$ vs time $t^*$ alongside the Martin & Moyce experimental data.

The non-dimensional scaling uses $H = 0.2$ m and $g = 9.81$ m/s².

### Step 4 – Qualitative check

At early times ($t^* < 1$) the waterfront should accelerate roughly as $x^* \approx 1 + 2\sqrt{g H}\,t / H$ (shallow-water theory). At later times wall interaction and energy dissipation slow the front. Good MPM results should closely follow the experimental scatter.
