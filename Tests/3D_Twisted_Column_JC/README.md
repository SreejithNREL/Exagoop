# Test Case: 3D Twisted Column (Johnson-Cook)

Reproduces the *twisted column* benchmark of

> V.P. Nguyen, A. de Vaucorbeil, S. Bordas, *The Material Point Method: Theory,
> Implementations and Applications*, Springer. Sect. 10.3.3,
> Fig. 10.26 (setup), Table 10.3 (material, shared with the plate case),
> Fig. 10.27 (FEM), Fig. 10.28 (ULMPM, cubic B-splines + MUSL).

## Physical Problem

A 100 mm copper column of 10 x 10 mm square section is clamped at `z = 0`. Its
top surface is driven as a rigid body about the z axis at `omega = 2*pi` rad/ms
— one full turn per millisecond, three turns over the reference 3 ms run. The
book states the purpose as assessing **robustness under large, highly nonlinear
deformation**. Material is Johnson-Cook with a Mie-Grüneisen EOS; there is no
gravity. The book's reference is an Abaqus FEM run (Fig. 10.27).

Boundary velocity (book Eq. 10.45), applied to every node of the top face using
the node's **current** coordinates:

```
v_x(t) = -omega * y(t),    v_y(t) = +omega * x(t),    v_z = 0
```

with `omega = omega_0 * n / T`, `omega_0 = 2*pi` rad/ms, `n` turns over final
time `T`. The reference run is `n = 3`, `T = 3 ms`, hence `omega = 2*pi`.

As in the plate case, `JC_C = JC_m = 0`, so the flow stress reduces to
`sigma_f = A + B*eps_p^n` — no rate sensitivity, no thermal softening, and
temperature is a passive output.

## Setup in ExaGOOP

| Parameter | Value |
|---|---|
| Domain | `[-9,9] x [-9,9] x [0,100]` mm, 18 x 18 x 20 cells (`dx=dy=1`, `dz=5`) |
| Column | `[-5,5]^2 x [0,100]` mm, 2x2x2 ppc = 16,000 particles |
| Periodicity | **`0 0 0`** — see the warning below |
| Constitutive model | `johnson_cook` (cm_id = 2), Table 10.3 in mm-ms-kg |
| `zlo` | `noslip` (clamped base) |
| `zhi` | `noslip` + UDF wall velocity (`UDF/libwall_twist.so`) |
| `xlo/xhi/ylo/yhi` | `slip` (column never reaches them) |
| Thermal BCs | adiabatic on all faces |
| Order scheme / stress update | 3 (cubic B-splines) / MUSL |
| `alpha_pic_flip`, CFL | 0.99, 0.3 |
| Gravity | none |

Two geometric constraints that are easy to get wrong:

* **the domain z-extent must equal the column length.** The twist is a *domain
  face* BC, so if `prob_hi[z]` sits above the column top the BC drives empty
  space and nothing happens.
* **x,y must exceed `5*sqrt(2) = 7.07 mm`**, the radius the section corners
  sweep to once the top has rotated. `±9` leaves ~2 cells of margin.

### WARNING — periodicity

The upstream preprocessor **hardcoded `mpm.is_it_periodic = 0 0 1` for every
3-D case** (a periodic z), with no way to override it. That identifies the
`k = 0` and `k = nz` nodal planes: the twist imposed on the top face also lands
on the clamped base, `nodal_bcs` zeroes it, the FLIP update then injects
`-v_backup` into the base particles, and the run diverges (T ~ 1e11 C by
t = 0.01 ms). The local copy of the generator now reads `"periodic": [x,y,z]`
from `config.json` (default `[0,0,0]`).

**Check any other 3-D deck for this line** — `3D_Compression_Column` still
carries `0 0 1`.

## Running

```bash
./Generate_MPs_and_InputFiles.sh          # builds UDF/, writes particles + deck
make -j4 MPM_HOME=../../ DIM=3 USE_EB=FALSE USE_TEMP=TRUE USE_MPI=FALSE
./ExaGOOP3d.gnu.ex Inputs_3D_Twisted_Column_JC.inp
```

The full 3 ms is ~42,000 steps at ~0.23 s/step — about **3 hours**, i.e. a
cluster run. `resume_chunk.sh SECONDS TEND WRITE_OUT` runs it in
wall-clock-bounded pieces resuming from the newest checkpoint, e.g.
`./resume_chunk.sh 40 0.09 0.005`.

For production, refine `nz` to >= 40. `dt` is set by `dx = 1`, so that costs
about 2x, not 8x — at `dz = 5` there are only ~6.7 cells per helical turn.

## Post-Processing

```bash
python3 PostProcess/validate.py           # newest frame
python3 PostProcess/validate.py Solution/ascii_files/3D_Twisted_Column_JC/matpnt_t0.090048
```

> **Voigt ordering is dimension-dependent** (`Source/constants.H`):
> 3-D is `XX, XY, XZ, YY, YZ, ZZ`, but 2-D is `XX, XY, YY, XZ, YZ, ZZ`.
> Using the 2-D order on 3-D data silently swaps `XZ` and `YY` and fabricates
> von Mises values ~8x too large, which looks exactly like a broken radial
> return. `validate.py` documents and uses the 3-D order.

## Results at t = 0.090 ms (book Fig. 10.28 frame 1 is t = 0.09 ms)

| Check | Result |
|---|---|
| Particles above the yield surface | **0** (max overshoot `-1.26e-07`) |
| On the yield surface | 15,396 of 15,448 plastically strained |
| Energy balance `E_th / (chi * W_pl)` | **0.958** |
| Clamped base, `\|v_xy\|` max at z < 2 | **0.030** mm/ms |
| Twist profile monotonic in z | yes (median 0.045 -> 22.0 mm/ms) |
| Top surface `\|v\| / (omega*r)` | 1.041 |
| Imposed top rotation | `omega*t` = 32.4 deg |
| `sigma_eq` max | 210 MPa (book scale 440) |
| `T` max | 3.11 C (book scale 76) |
| `eps_p` max | 0.088 |

Consistent with the book's first frame, which shows a nearly uniform column with
only the top beginning to twist and a temperature field still essentially at the
reference value. Deformation and heating concentrate near the driven end, as
expected this early.

### Known residuals

* **Energy balance closes to 4%**, against 0.15% for the 2-D plate. Suspected to
  be the massless-node issue below, which inflates strain rates near the driven
  face; worth re-checking at later times.
* **Top surface over-speeds by ~4%** (`|v|/(omega*r) = 1.041`). The UDF is
  imposed on *every* node of the boundary face, out to the domain corners, not
  only where material sits. With cubic B-splines a particle at `r = 7.07` reaches
  nodes at `r = 9`, whose imposed speed is `omega*9 = 56.5` mm/ms against a
  rigid-body maximum of `omega*7.07 = 44.4`. Massless nodes should not
  contribute to G2P. Settle this before trusting quantitative results at the
  later frames.
