# Test Case: 2D Elastic Disk Collision

## Physical Problem

Two identical elastic disks of radius $R = 0.2$ m are placed diagonally opposite each other in a $[0,1]^2$ m periodic domain and given initial velocities directed towards each other. Disk 1 is centred at $(0.2, 0.2)$ with velocity $(+0.1, +0.1)$ m/s; disk 2 is centred at $(0.8, 0.8)$ with velocity $(-0.1, -0.1)$ m/s. Both disks are linear elastic ($E = 1000$ Pa, $\nu = 0.3$). The disks collide near the centre of the domain, deform elastically, and then separate. Because the domain is periodic, the disks re-enter from the opposite side and collide again. This is a canonical MPM test for multi-body contact (handled naturally by MPM without explicit contact algorithms), elastic rebound, and long-time energy conservation.

## Exact Solution

No closed-form exact solution exists for the deforming-disk collision. The primary validation quantity is **total mechanical energy conservation**: since the material is elastic and the domain is periodic with no dissipation (pure FLIP, $\alpha = 1.0$), the sum of total kinetic energy (TKE) and total strain energy (TSE) should remain constant throughout the simulation. Any significant energy drift indicates numerical dissipation or instability.

For a head-on collision of two disks the qualitative expectation is:
- Pre-collision: TKE constant, TSE = 0.
- During collision: TKE decreases as kinetic energy converts to elastic strain energy.
- Post-collision: TKE returns to initial value, TSE returns to 0 (elastic rebound).
- Over many collision cycles (via periodic BCs) the pattern repeats.

## Simulation Setup in ExaGOOP

| Parameter | Value |
|-----------|-------|
| Dimensions | 2D |
| Domain | $[0,1] \times [0,1]$ m, 40 × 40 cells |
| Particles per cell | 4 × 4 = 16 per cell |
| Body 1 | Circle, centre $(0.2, 0.2)$, $R = 0.2$ m |
| Body 2 | Circle, centre $(0.8, 0.8)$, $R = 0.2$ m |
| Initial velocity (body 1) | $(+0.1, +0.1)$ m/s (uniform) |
| Initial velocity (body 2) | $(-0.1, -0.1)$ m/s (uniform) |
| Constitutive model | Linear elastic, $E = 1000$ Pa, $\nu = 0.3$ |
| Stress update scheme | MUSL |
| Order scheme | 1 |
| PIC/FLIP blending | Pure FLIP ($\alpha = 1.0$) |
| CFL | 0.1 (variable timestep) |
| Final time | 3.5 s |
| Output interval | 0.1 s (→ 35 snapshots) |
| BCs (all faces) | `periodic` |
| Gravity | Zero |
| Temperature | Disabled |
| Output format | HDF5 |

The 4 × 4 particles per cell gives 16 particles per cell, which is denser than most other test cases. This improves the accuracy of the contact zone representation but increases memory and run time. The periodic BCs allow the disks to exit one side and re-enter from the other, enabling multiple collision cycles in a single simulation.

## Key Input Parameters

- **`bodies[*].constitutive_model.E`** — Young's modulus. Controls the wave speed $c = \sqrt{E/\rho}$ and the stiffness of the contact. Very soft ($E \ll 1000$) disks deform excessively; very stiff ($E \gg 1000$) disks require smaller CFL timesteps.
- **`bodies[*].constitutive_model.nu`** — Poisson's ratio. Affects the lateral expansion during collision.
- **`ppc`** — `[4, 4]` (16 per cell). Reducing to `[2, 2]` speeds up the simulation but reduces contact zone accuracy.
- **`alpha_pic_flip`** — `1.0` for pure FLIP (no dissipation). Reducing this value introduces numerical damping and causes the total energy to decay, which is clearly visible in the energy plot.
- **`boundary_conditions.*.mom: "periodic"`** — allows the disks to pass through the domain boundaries and collide multiple times.
- **`diagnostics.do_calculate_tke_tse`** — `1`; writes TKE and TSE to the diagnostics file. Essential for the energy conservation check.
- **`simulation.final_time`** — 3.5 s; long enough for several collision cycles.
- **`simulation.write_output_time`** — 0.1 s; coarser than many other tests since the main output is the energy diagnostics rather than spatial snapshots.

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
cd $MPM_HOME/Build_Gnumake/
#make necessary changes in GNUmakefile
make -j
cd ../Tests/Elastic_disk_collision
cp ../../Build_Gnumake/ExaGOOP1d.*.ex .
./ExaGOOP1d.*.ex Inputs_ElasticDiskCollision.inp
mpirun -n 4 ./ExaGOOP1d.*.ex ../Inputs_ElasticDiskCollision.inp
```

### Step 3 – Energy conservation plot

```bash
python3 PostProcess/plot_energy.py \
    --folder       Solution/ascii_files/<output_tag> \
    --energyfile   Diagnostics/<output_tag>/Total_Energies.dat \
    --outputpic    energy.png \
    --outputmovie  collision.mp4
```

This script produces:
1. An animation (`--outputmovie`) of the disk collision, with particles coloured by speed magnitude.
2. An energy plot (`--outputpic`) showing TKE (red), TSE (blue), and total energy TE = TKE + TSE (black) vs time.

The total energy trace should be flat (within a few percent) throughout the simulation. Any steady downward drift indicates excessive numerical dissipation.

### Step 4 – Qualitative validation

During each collision:
- The TKE dips and the TSE rises simultaneously, with the sum remaining constant.
- After separation the disks should recover close to their initial shapes (elastic rebound).
- For the periodic domain, the disks approach each other again after travelling approximately one domain length, and the energy exchange pattern repeats.
