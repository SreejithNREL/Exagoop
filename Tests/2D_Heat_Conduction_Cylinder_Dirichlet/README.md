# Test Case: 2D Heat Conduction Around a Cylinder (Embedded Boundary)

## Physical Problem

A two-dimensional square domain $[0,1]^2$ contains a solid cylindrical obstacle of radius $R = 0.15$ m centred at $(0.5, 0.5)$. The cylinder surface is held at $T = 1$ and the four outer walls at $T = 0$. The material (fluid) occupies the annular region between the cylinder and the domain boundaries. At long times the solution approaches a steady-state temperature distribution $\nabla^2 T = 0$ in this non-trivial geometry. This test exercises the **Embedded Boundary (EB)** capability of ExaGOOP: the cylinder is represented as a level-set body (`geom_type: sphere`) rather than a body-fitted mesh, and the thermal BC at the curved surface is enforced via the level-set framework.

## Exact Solution (Reference)

There is no closed-form analytical solution for steady heat conduction in a square domain with a centred cylinder. The validation reference is computed numerically: `PostProcess/validate.py` solves Laplace's equation $\nabla^2 T = 0$ on a 400 × 400 finite-difference grid with the same geometry and BCs, then interpolates onto the MPM particle positions for comparison.

The steady-state problem is well-posed (elliptic) so the FD reference converges rapidly with grid refinement; at 400 × 400 it is effectively exact for the purposes of this benchmark.

## Simulation Setup in ExaGOOP

| Parameter | Value |
|-----------|-------|
| Dimensions | 2D |
| Domain | $[0,1] \times [0,1]$, 128 × 128 cells |
| Particles per cell | 2 × 2 |
| Constitutive model | Linear elastic, $E = 10^6$, $\nu = 0.3$ |
| Initial temperature | $T = 0$ (uniform) |
| Thermal conductivity | $k = 1$ |
| Specific heat | $c_p = 1$ |
| Stress update scheme | MUSL |
| Fixed timestep | Yes, $\Delta t = 10^{-5}$ s |
| Final time | 0.1 s (approaches steady state) |
| Output interval | 0.01 s |
| BC (all outer faces) | `slip` (momentum) + Dirichlet $T = 0$ |
| Cylinder BC | `slipwall` (momentum) + `isothermal` $T = 1$ |
| Embedded boundary | Enabled (`use_eb: true`) |
| Build system | gnumake |
| Output format | HDF5 |

The cylinder is described under `levelset_bodies` in the config:

```json
{
  "name": "cylinder",
  "geom_type": "sphere",
  "sphere_radius": 0.15,
  "sphere_center": [0.5, 0.5, 0.0],
  "sphere_has_fluid_inside": false,
  "ls_refinement": 2,
  "levelset_mom": "slipwall",
  "temp_bc_type": "isothermal",
  "lset_T_wall": 1.0
}
```

`ls_refinement: 2` means the level-set grid is refined 2 levels relative to the base grid for accurate interface representation.

## Key Input Parameters

- **`use_eb: true`** — must be set to enable the embedded boundary solver. Requires an EB-enabled build (`gnumake` build system here).
- **`levelset_bodies[0].sphere_radius`** — cylinder radius. Changing this alters the geometry and the steady-state temperature distribution.
- **`levelset_bodies[0].sphere_center`** — cylinder centre coordinates.
- **`levelset_bodies[0].lset_T_wall`** — temperature imposed on the cylinder surface ($T = 1$ here).
- **`levelset_bodies[0].ls_refinement`** — level of AMR refinement for the level-set. Higher values give a sharper interface representation.
- **`boundary_conditions.*.temp.T_wall`** — outer boundary temperature ($T = 0$ on all four walls).
- **`nx`, `ny`** — grid resolution (128 × 128). The high resolution here is needed to represent the curved cylinder boundary adequately with the level-set approach.
- **`simulation.num_redist`** — `10`; particle redistribution keeps the particle density uniform near the curved EB interface.

## Post-Processing

### Prerequisites

```bash
pip install numpy matplotlib scipy h5py
```

### Step 1 – Generate particles and input file

```bash
bash Generate_MPs_and_InputFiles.sh
```

Use `PreProcess/plot_particles.py` to verify that particles are present only in the fluid region (outside the cylinder).

### Step 2 – Build and run ExaGOOP

This test requires an EB-enabled build. Use the gnumake build system:

```bash
cd $MPM_HOME/Build_Gnumake/
#make necessary changes in GNUmakefile
make -j
cd ../Tests/2D_Heat_Conduction_Cylinder_Dirichlet
cp ../../Build_Gnumake/ExaGOOP1d.*.ex .
./ExaGOOP1d.*.ex Inputs_2DHeat_Conduction_Cylinder_Dirichlet.inp
mpirun -n 4 ./ExaGOOP1d.*.ex ../Inputs_2DHeat_Conduction_Cylinder_Dirichlet.inp
```

### Step 3 – Validate and plot

```bash
python3 PostProcess/validate.py --time 0.1
```

This script:
1. Solves Laplace's equation on a 400 × 400 FD grid.
2. Interpolates the FD solution onto the MPM particle positions in the annular region $r \in [0.15, 0.5]$ from the cylinder centre.
3. Produces a three-panel figure: numerical temperature, FD reference, and pointwise error.
4. Reports the RMS error and prints `PASS` if below $0.1$.

The output figure is saved to `--outputpic/temperature_contours_t<time>.png`.
