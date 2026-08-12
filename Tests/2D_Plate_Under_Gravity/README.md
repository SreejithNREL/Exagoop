# Test Case: 2D Plate Under Gravitational Compression

Reproduces the *plate under gravitational compression* benchmark of

> V.P. Nguyen, A. de Vaucorbeil, S. Bordas, *The Material Point Method: Theory,
> Implementations and Applications*, Springer.
> Section 10.3.3, Fig. 10.23 (setup), Table 10.3 (parameters),
> Fig. 10.24 (equivalent plastic strain), Fig. 10.25 (temperature).

## Physical Problem

A 10 mm x 10 mm plate of OFHC copper is supported at its base and released under
a very large downward body force, `g = -5000 mm/ms^2` (~5x10^6 m/s^2). The
material is modelled with **Johnson-Cook plasticity** and a **Mie-Grüneisen**
equation of state. The plate squats under its own weight, two shear bands form
from the bottom corners, and plastic dissipation heats the material along those
bands.

The book states the purpose of the test explicitly:

1. **stability against cell crossing** — many crossing events occur as the
   plate deforms, and
2. **heat generation from plastic work**.

There is **no analytical solution**; the book's reference is an Abaqus FEM
calculation.

### Thermal coupling

The only heat source is plastic dissipation. An increment of plastic strain
`d(eps_p)` raises the temperature by (book Eq. 10.43)

```
dT = (chi / (rho * c_p)) * sigma_f * d(eps_p)
```

which enters the energy balance as the volumetric source (Eq. 10.44)

```
gamma* = chi * sigma_f * eps_p_dot
```

The book drops the heat-flux term `q` from the energy equation, i.e. the
temperature evolution is **adiabatic**. In this deck conduction is left enabled
at the physical value of `k`, but it is negligible over the 0.2 ms of the run:
the diffusion length is `sqrt(alpha*t) ~ 0.15 mm` against a 10 mm plate
(`alpha = k/(rho*c_p) ~ 0.11 mm^2/ms`), i.e. below one cell.

> **Note on `JC_C` and `JC_m`.** Table 10.3 sets both to zero. The flow stress
> therefore reduces to `sigma_f = A + B*eps_p^n`, with **no** strain-rate
> sensitivity and **no** thermal softening. Temperature is computed but never
> feeds back into the mechanics — the coupling is one-way. This case validates
> the Taylor-Quinney plastic-work-to-heat path; it does **not** exercise the
> rate or thermal-softening terms of the Johnson-Cook model.

## Units

The deck uses a consistent **mm - ms - kg** system, in which

| Quantity | Unit | Note |
|---|---|---|
| length | mm | |
| time | ms | |
| mass | kg | |
| stress | kg/(mm ms^2) | **= 1 GPa** |
| energy | kg mm^2/ms^2 | **= 1 J** |
| power | J/ms | = 1 kW |
| velocity | mm/ms | = 1 m/s |

So Table 10.3 converts as: `E = 115 GPa -> 115`, `A = 65 MPa -> 0.065`,
`B = 356 MPa -> 0.356`, `c_0 = 3586 m/s -> 3586`, `c_p = 384 J/kg/K -> 384`,
`k = 3.86e-4 kW/mm/K -> 3.86e-4`, `eps_dot_0 = 1 s^-1 -> 1e-3 ms^-1`.

Consistency check: `sqrt(E/rho) = sqrt(115/8.94e-6) = 3586 mm/ms`, exactly the
tabulated bulk sound speed `c_0` — the table is internally consistent.

## Simulation Setup in ExaGOOP

| Parameter | Value |
|-----------|-------|
| Dimensions | 2D (plane strain) |
| Domain | `[-5, 15] x [-2.5, 13.5]` mm, 80 x 64 cells (`dx = dy = 0.25` mm) |
| Plate | `[0, 10] x [0, 10]` mm, 2 x 2 particles per cell (6400 points) |
| Constitutive model | `johnson_cook` (cm_id = 2) |
| Support | **static level-set pedestal**, `box` from `(-2.5,-2.5)` to `(12.5, 0)`, `noslipwall` |
| Gravity | `(0, -5000, 0)` mm/ms^2 |
| Order scheme | 3 (cubic B-splines, as in the book) |
| Stress update | MUSL |
| PIC/FLIP blend | `alpha_pic_flip = 0.99` (book uses FLIP MUSL) |
| Initial temperature | 0 C (= `JC_Tr`) |
| Thermal BCs | adiabatic on all faces and on the pedestal |
| Final time | 0.2 ms |
| CFL | 0.2 |

### Why a level-set pedestal instead of a `noslip` domain boundary

The book clamps the bottom edge of the plate. Here the support is a **static
level set** wider than the plate (15 mm vs 10 mm), so that

* the plate rests on a rigid surface rather than on the domain boundary,
* the full 10 mm footprint stays supported even as the base spreads laterally,
  and
* the contact condition (`noslipwall`) is applied by the same level-set
  machinery used for moving tools elsewhere in the code, which is what we want
  exercised.

The pedestal top sits at `y = 0`, aligned with a grid node line, and the lowest
particle row starts at `y = 0.0625` mm.

## Material Parameters (Table 10.3)

| Parameter | Book value | Deck value (mm-ms-kg) |
|---|---|---|
| Density `rho` | 8.94e-6 kg/mm^3 | 8.94e-6 |
| Young's modulus `E` | 115 GPa | 115.0 |
| Poisson's ratio `nu` | 0.31 | 0.31 |
| Reference temperature `T_0` | 0 C | `JC_Tr = 0` |
| Heat capacity `c` | 384 J/kg/K | `spheat = 384` |
| Heat conductivity `k` | 3.86e-4 kW/mm/K | `thermcond = 3.86e-4` |
| Inelastic heat fraction `chi` | 0.9 | `JC_chi = 0.9` |
| `A` | 65 MPa | `JC_A = 0.065` |
| `B` | 356 MPa | `JC_B = 0.356` |
| `n` | 0.37 | `JC_n = 0.37` |
| `C` | 0 | `JC_C = 0` |
| `m` | 0 | `JC_m = 0` |
| `T_melt` | 1600 C | `JC_Tm = 1600` |
| `eps_dot_0` | 1.0 s^-1 | `JC_eps_dot_0 = 1e-3` |
| `c_0` | 3586 m/s | `JC_c0 = 3586` |
| `S_alpha` | 1.50 | `JC_Salpha = 1.5` |
| `Gamma_0` | 0 | `JC_Gamma0 = 0` |

Note the heat-expansion coefficient `alpha = 1.67e-5 /K` from Table 10.3 is not
used: with `JC_m = 0` there is no thermo-mechanical feedback, and ExaGOOP's
Johnson-Cook path has no thermal-strain term.

## Running

```bash
# 1. particles + input deck
./Generate_MPs_and_InputFiles.sh

# 2. build (2D, embedded boundary for the level set, temperature solver on)
make -j4 MPM_HOME=../../ DIM=2 USE_EB=TRUE USE_TEMP=TRUE USE_MPI=FALSE

# 3. run
./ExaGOOP2d.gnu.ex Inputs_2D_Plate_Under_Gravity.inp
```

`USE_EB=TRUE` is required — the level-set pedestal lives behind `#if USE_EB`.

If your environment limits how long a single command may run, `resume_chunk.sh`
runs the case in wall-clock-bounded pieces, resuming from the newest checkpoint
each time:

```bash
./resume_chunk.sh 40     # repeat until the log reaches t = 0.2
```

## Post-Processing

```bash
python3 PostProcess/validate.py            # newest frame
python3 PostProcess/validate.py Solution/ascii_files/2D_Plate_Under_Gravity/matpnt_t0.200000
```

The script runs four checks:

1. **Plastic-work → heat energy balance (primary, analytic).** With adiabatic
   boundaries the total thermal energy must equal the Taylor-Quinney fraction of
   the total plastic work,
   `sum_p m_p c_p dT_p == chi * sum_p V_p * [A*eps_p + B*eps_p^(1+n)/(1+n)]`,
   the integral being exact because `C = m = 0`. This is the rigorous test of the
   whole plastic-work-to-heat path.
   *This is a global identity, not a pointwise one*: the heat source is deposited
   on the grid (`Q_I = sum_p V_p gamma* phi_I`) and mapped back by G2P, which
   smooths across a stencil. The script also reports the pointwise scatter, but
   only as a diagnostic — large relative differences at the localized corner
   peaks are expected and are not an error.
2. **Yield surface (analytic).** No particle may lie *above*
   `von Mises = A + B*eps_p^n` — radial return must never violate yield.
3. **Pedestal contact.** No material point may sit below `y = 0`, and the
   material must not spread past the pedestal footprint.
4. **Structure vs. the book.** Peak values and locations, the deformed
   silhouette by height, and left-right symmetry of the two shear bands.

### Plotting

```bash
python3 PostProcess/plot_fields.py            # newest frame -> plate_fields.png
```

## Results (t = 0.2 ms, 80 x 64 grid, 2x2 ppc)

| Quantity | ExaGOOP | Book |
|---|---|---|
| Energy balance `E_thermal / (chi * W_plastic)` | **0.99854** | — (exact identity) |
| Particles above the yield surface | **0** (max overshoot `-5.4e-9`) | — |
| Particles penetrating the pedestal | **0** (`y_min = 0.020` mm) | — |
| Shear-band points, left / right | **19 / 19** (0 % asymmetry) | symmetric |
| max `eps_p` | 1.362 | colour scale 1.5 |
| max `T` | 71.7 C | colour scale 52 C |
| Location of both peaks | bottom corners, `(11.0, 0.05)` | bottom corners |
| Deformed height | 8.97 mm (from 10) | squats |
| Deformed width, base → top | 12.60 → 9.97 mm | widens at base |

The plate reaches a **static equilibrium at t ≈ 0.05 ms** and does not evolve
further; `eps_p`, the extent and the height are unchanged from t = 0.05 to
t = 0.2, and `T_max` decays slowly (74.7 → 71.7 C) under the residual
conduction. Running to the book's t = 0.2 ms therefore mostly confirms the
steady state.

**Shape and band structure agree with Fig. 10.24**: concave top surface, sides
bulging outward toward the base, and an arch-shaped plastic zone rising from the
two bottom corners with an undeformed cap on top.

> **Plot on a clipped colour scale.** The peak plastic strain is extremely
> localized at the two bottom corners (95th percentile is 0.35 against a maximum
> of 1.36). On the book's full 0–1.5 scale the arch structure saturates into a
> nearly uniform blue and looks like a thin basal layer. `plot_fields.py`
> therefore renders both the full and the clipped scale; compare the *clipped*
> panel with the book's figure.

### A caveat on the book's own figures

Figs. 10.24 and 10.25 report `eps_p` up to **1.5** and `T` up to **52 C** at the
same instant. Those two numbers are **mutually inconsistent** under the book's
own Eq. 10.43 and Table 10.3: integrating the flow stress to `eps_p = 1.5` gives

```
W = A*1.5 + B*1.5^1.37/1.37 = 550 MPa   ->   dT = 0.9*550/(8.94e-6*384) = 144 K
```

not 52 K; conversely `T = 52 C` corresponds to `eps_p ~ 0.68`. Part of that gap
is physical — the hottest particle is not the most-strained one once the grid
smooths the source — but the two colour-bar maxima cannot both be taken at face
value. Do **not** use simultaneous agreement with both numbers as a pass
criterion. Our own pair (1.362 / 71.7 C) shows the same qualitative offset and
is internally consistent, since the global energy balance closes to 0.15 %.

## Files

```
2D_Plate_Under_Gravity/
├── GNUmakefile                        DIM=2, USE_EB=TRUE, USE_TEMP=TRUE
├── Generate_MPs_and_InputFiles.sh     runs the preprocessor
├── run.sh                             generate + build + run
├── resume_chunk.sh                    checkpoint-resumed chunked run
├── PreProcess/
│   ├── config.json                    case definition
│   └── Generate_MPs_Inputfile_Generic.py
└── PostProcess/
    └── validate.py                    quantitative checks
```

## Note on the bundled preprocessor

`PreProcess/Generate_MPs_Inputfile_Generic.py` is the shared generic
preprocessor with two local fixes:

1. **Particle density.** Upstream copies hardcode `dens = 1.0`, so
   `constitutive_model.density` never reached the particles — it only reached
   the material table. That is harmless for the existing non-dimensional tests
   (which use `rho = 1`) but fatal here: the Mie-Grüneisen EOS would see
   `eta = rho/rho_0 = 1/8.94e-6 ~ 1.1e5`. Fixed to read
   `constitutive_model["density"]`.
2. **Optional imports.** `h5py`/`matplotlib` are imported lazily so the
   preprocessor runs where they are absent (this case uses ASCII output).
