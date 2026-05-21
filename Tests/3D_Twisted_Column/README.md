# Test Case: 3D Twisted Elastic Column

## Physical Problem

A three-dimensional elastic column of cross-section $0.5 \times 0.5$ m and height $L = 2$ m is clamped at its base ($z = 0$, implicit from the BC) and twisted at its top face ($z = L$) by an angular velocity $\omega = 0.5$ rad/s applied via a user-defined function (UDF). The column material is linear elastic with $E = 10^7$ Pa, $\nu = 0.3$, and $\rho = 1000$ kg/m³. Under the applied rotation the column develops a helical deformation pattern; the twist angle grows linearly in both time and height.

This test verifies the 3D elastic solver and the UDF-driven moving wall boundary condition. It is the primary verification test for the UDF moving boundary feature of ExaGOOP.

## Exact Solution (Quasi-static/Kinematic)

For small deformations and slow angular velocity the twist angle at height $z$ and time $t$ is the rigid-body rotation:

$$\theta(z, t) = \omega\,t\,\frac{z}{L}$$

The top face (at $z = L$) rotates at $\omega\,t$ radians; the base is clamped ($\theta = 0$); the profile is linear in $z$. The UDF `UDF/wall_vel_twist.c` prescribes:

$$v_x = -\omega\,y, \quad v_y = +\omega\,x, \quad v_z = 0$$

at the top face ($z_\text{hi}$), producing solid-body rotation at the boundary. At $t = 4$ s the total rotation of the top face is $\omega t = 2$ rad ≈ 115°.

## Simulation Setup in ExaGOOP

| Parameter | Value |
|-----------|-------|
| Dimensions | 3D |
| Domain | $x \in [-0.6, 0.6]$, $y \in [-0.6, 0.6]$, $z \in [0, 2]$ |
| Grid | 12 × 12 × 20 cells |
| Body | Block $[-0.25, 0.25]^2 \times [0, 2]$ |
| Particles per cell | 2 × 2 × 2 |
| Constitutive model | Linear elastic, $E = 10^7$, $\nu = 0.3$, $\rho = 1000$ |
| Initial velocity | Zero |
| Stress update scheme | USL (Update Stress Last) |
| Order scheme | 1 |
| CFL | 0.3 |
| Final time | 4.0 s |
| Output interval | 0.1 s (→ 40 snapshots) |
| UDF (z-hi BC) | `UDF/libwall_twist.dylib` / `wall_vel_twist` |
| Angular velocity $\omega$ | 0.5 rad/s |
| Output format | ASCII |
| Temperature | Disabled |

The top-face velocity UDF is specified in `config.json` under the `udf` key. The shared library must be compiled from `UDF/wall_vel_twist.c` before running.

## Key Input Parameters

- **`udf.omega`** — angular velocity applied at the top face. Controls the twist rate and thus the deformation magnitude at a given time.
- **`udf.zhi_lib`** — path to the compiled UDF shared library (`.dylib` on macOS, `.so` on Linux).
- **`udf.zhi_func`** — name of the C function inside the library (`wall_vel_twist`).
- **`bodies[0].constitutive_model.E`** — Young's modulus. A stiffer column resists torsion more (higher stress for the same twist).
- **`bodies[0].constitutive_model.nu`** — Poisson's ratio. Affects the coupling between axial strain and lateral deformation.
- **`CFL`** — 0.3; can be increased slightly for this quasi-static problem but stability should be checked.
- **`stress_update_scheme`** — `USL` is used here. `MUSL` is also valid but may give slightly different energy behaviour under large deformation.
- **`physics.final_time`** and **`physics.write_output_time`** — simulation duration and output frequency. The 40 snapshots at 0.1 s intervals capture the full twist from 0 to 2 rad.
- **`grid.nx`, `grid.ny`, `grid.nz`** — grid resolution. Increasing `nz` improves the resolution of the linear twist-angle gradient along the column height.

## Compilation of the UDF

On macOS:

```bash
cd UDF
clang -O2 -shared -fPIC -o libwall_twist.dylib wall_vel_twist.c -lm
```

On Linux:

```bash
cd UDF
gcc -O2 -shared -fPIC -o libwall_twist.so wall_vel_twist.c -lm
```

Update `config.json → udf.zhi_lib` to point to the correct library path (`libwall_twist.so` on Linux).

## Post-Processing

### Prerequisites

```bash
pip install numpy matplotlib
```

### Step 1 – Generate particles and input file

```bash
bash Generate_MPs_and_InputFiles.sh
```

### Step 2 – Build and run ExaGOOP

```bash
mkdir -p build && cd build
bash ../cmake_run.sh   # 3D build required (EXAGOOP_DIM=3)
mpirun -n 8 ./ExaGOOP3d.*.ex ../Inputs_3DTwistedColumn.inp
```

Output particle files (`matpnt_000000`, `matpnt_000001`, …) are written under `Solution/<output_tag>/`.

### Step 3 – Visualise and validate

```bash
python3 PostProcess/visualise_twist.py \
    --solution_dir Solution \
    --output_tag 3D_Twisted_Column \
    --omega 0.5 \
    --write_output_time 0.1 \
    --outdir PostProcess/Figures
```

This produces three figures saved to `--outdir`:

1. **`column_3d_*.png`** — 3D scatter plot of particle positions at the chosen snapshot, coloured by height.
2. **`twist_profile_*.png`** — twist angle (degrees) vs $z$, comparing MPM to the analytical linear profile $\theta = \omega t z / L$.
3. **`top_displacement_vs_time.png`** — mean transverse displacement of the top face vs time, showing the steady growth under constant $\omega$.

Use `--snapshot 000040` to select a specific snapshot (e.g. the last one at $t = 4$ s). The twist profile plot is the primary validation: the MPM points should fall on the analytical straight line $\theta(z) = \omega t z / L$ to within a few percent.
