# Test Case: Square Plate with Thermal and Mechanical Loadings

Reproduces the *square plate with thermal and mechanical loadings* verification of

> V.P. Nguyen, A. de Vaucorbeil, S. Bordas, *The Material Point Method: Theory,
> Implementations and Applications*, Springer. Sect. 10.3.3,
> Fig. 10.21 (setup), Table 10.2 (material), Fig. 10.22 (results),
> after Tao et al. (2016).

## Physical Problem

A 0.2 x 0.2 m thermo-elastic plate, initially at 20 °C.

| Face | Thermal condition |
|---|---|
| top (`yhi`) | convection, `h = 2000` W/m²°C, `T_inf = 30` °C |
| right (`xhi`) | convection, `h = 2000` W/m²°C, `T_inf = 30` °C |
| bottom (`ylo`) | constant heat flux `q = 5000` |
| left (`xlo`) | insulated (adiabatic) |

A pressure ramps 0 -> 100 over 0-5 s and is then held to 10 s. The book monitors
temperature and `sigma_xx` at point **A** = (0.1, 0.1) m against FEM (Fig. 10.22).

### Material (Table 10.2), SI units

| Quantity | Value |
|---|---|
| density `rho` | 2100 kg/m³ |
| Young's modulus `E` | 70 GPa |
| Poisson's ratio `nu` | 0.33 |
| heat conductivity `k` | 500 W/m°C |
| heat capacity `c` | 50 J/kg°C |
| heat expansion `alpha` | 25e-8 /°C |

## SCOPE: thermal half only

**ExaGOOP has no thermo-elastic coupling.** There is no thermal-strain term
anywhere in `Source/` — `linear_elastic` takes only `(eps, sigma, E, nu)`, and
`alpha` is unused. Book Eq. 10.38 is exactly `sigma = C : (eps - alpha dT I)`,
so `alpha` *is* the entire mechanical content of this test; **Fig. 10.22(b)
(`sigma_xx` at A) cannot be reproduced** without adding that term (plus a
time-ramped traction BC — `external_loads` currently supports only a constant
`extforce` over a fixed slab).

This is not a limitation of the setup: with no gravity, no external load and no
thermal strain, the material stays exactly at rest (`|v|max = 0` at every
frame), so the thermal problem runs on a static grid. That makes the thermal
half — Fig. 10.22(a) — **fully faithful**, and it is what this test verifies.

It is also the first case in the suite to exercise convection, heat flux and
adiabatic boundary conditions simultaneously on one domain.

A second consequence of leaving mechanics out: the run can use a fixed
`dt = 2e-3 s` set by the explicit diffusion limit (`dt <= h²/(4*alpha_th)`
= 5.2e-3 s with `alpha_th = k/(rho c) = 4.76e-3` m²/s) instead of the mechanical
CFL, which at `E = 70` GPa would demand `dt ~ 3.5e-7` s and ~3e7 steps for 10 s.
The whole case now runs in well under a minute.

## Setup in ExaGOOP

| Parameter | Value |
|---|---|
| Domain | `[0, 0.2]²` m, 20 x 20 cells (`h = 0.01` m) |
| Particles | 2 x 2 per cell = 1600 |
| Constitutive model | linear elastic, Table 10.2 |
| Momentum BCs | `slip` on all faces (nothing moves) |
| Time stepping | fixed, `dt = 2e-3` s, to `t = 10` s |
| Order scheme / stress update | 1 / MUSL |

## Running

```bash
./Generate_MPs_and_InputFiles.sh
make -j4 MPM_HOME=../../ DIM=2 USE_TEMP=TRUE USE_MPI=FALSE
./ExaGOOP2d.gnu.ex Inputs_2D_Square_Plate_ThermoMech.inp
python3 PostProcess/validate.py
```

## Results

`T` at point A rises smoothly and monotonically from **20.000 °C to 29.058 °C**
over 10 s, still climbing at the end and approaching but not reaching
`T_inf = 30`. The book's Fig. 10.22(a) shows the same: a smooth rise from 20 to
**about 29 °C** at t = 10 s, with FEM and the ULMPM variants all on top of each
other. The field stays bounded by the ambient throughout and the material never
moves.

See `PostProcess/pointA_temperature.png` and `pointA_history.csv`.

## Bug found and fixed while building this case

`Source/nodal_data_ops.cpp`, domain-boundary convective (Robin) BC. The Biot
number was computed as

```c
amrex::Real Bi = hc * dx_g[d];        // WRONG - missing / k
```

The Robin condition `k dT/dn = h (T_inf - T)` discretises to
`T_node = (T_nb + Bi T_inf) / (1 + Bi)` with **`Bi = h dx / k`**. Without the
`/k`, `Bi` is `k` times too large — a factor of **500** here — which collapses
the convective BC into a Dirichlet `T = T_inf`.

Symptom: at `t = 0`, with a uniform 20 °C initial condition, boundary particles
jumped straight to 29.18 °C. Predicted exactly by the bug —
`(20 + 20*30)/21 = 29.52` at a node against the correct
`(20 + 0.04*30)/1.04 = 20.38`. `T` at A then saturated at `T_inf` by t ~ 5 s
instead of tracking the book's slower rise.

**Why it survived:** the only other test using this BC,
`1D_Heat_Conduction_Convective`, has `thermcond = 1.0`, where `h*dx/k` and
`h*dx` are numerically identical. That test is unaffected by the fix.

The level-set convective BC (same file, ~line 356) already had the correct
`Bi = h_conv_v * dx[dom_dir] / k_node`; only the domain-boundary branch was
wrong.
