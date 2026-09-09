# 2D Johnson-Cook inertial compression

Yield-surface verification of the Johnson-Cook constitutive model
(plasticity with von Mises radial return, Mie-Grüneisen equation of state).

A rectangular block ($x \in [0.2, 0.8]$, $y \in [0.35, 0.65]$ m) in a
$[0,1]^2$ domain is given the initial velocity field
$v_x = -\dot{\varepsilon}\,(x - 0.5)$, $\dot{\varepsilon} = 2\ \mathrm{s^{-1}}$,
and compresses uniaxially under its own inertia. Free surfaces, no gravity,
slip walls. Nondimensional material: $E = 1000$, $\nu = 0.3$, $\rho = 1$,
Johnson-Cook $A = 1$, $B = 2$, $n = 0.5$ (strain-rate and thermal factors off:
$C = m = 0$), Mie-Grüneisen $c_0 = 30$, $S = 1.5$, $\Gamma_0 = 2$.

## What is checked

The flow stress is $\sigma_f = A + B\,\varepsilon_p^{\,n}$. The check is
path-independent: at every output, every yielded particle must have a von
Mises stress on or below $\sigma_f$ (on = plastically loading, below = elastic
unloading as the block's momentum reverses), never above; and the median
distance from the surface over the block must be $< 10^{-3}$. Typical result:
0 particles above, median $\sim 10^{-4}$.

## Running

```bash
./Generate_MPs_and_InputFiles.sh          # mpm_particles.dat + .inp (2D, USE_TEMP=FALSE build)
./ExaGOOP2d.gnu.ex Inputs_2DJCInertialCompression.inp
python3 PostProcess/validate.py
```

Note: `mpm.applied_strainrate` cannot be used to drive this case — it only
adds to the accumulated strain array and never enters the strain-rate field
that rate-based models (Johnson-Cook, fluid) consume. The initial velocity
field is the honest way to load the block.

Internal state variables written per particle (see `Solution/materials.txt`):
`isv_0` equivalent plastic strain, `isv_1..6` un-rotated deviatoric stress,
`isv_7` damage (0 here), `isv_8` Mie-Grüneisen pressure.

## Damage variant

`PreProcess/config_damage.json` adds `JC_D1 = 0.15` (D2..D5 = 0, so the
failure strain is the constant $\varepsilon_f = 0.15$) and doubles the initial
strain rate so that particles reach it within 0.08 s. Damage accumulates as
$D = \sum \Delta\varepsilon_p / \varepsilon_f$; at $D = 1$ the particle fails:
its deviatoric stress is removed and only compressive pressure is retained.

```bash
./Generate_MPs_and_InputFiles_Damage.sh
./ExaGOOP2d.gnu.ex Inputs_2DJCInertialCompression_Damage.inp
python3 PostProcess/validate_damage.py
```

Checks on the last output: unfailed particles have $D = \varepsilon_p/\varepsilon_f$
to round-off; failed particles have $\varepsilon_p \ge \varepsilon_f$ and zero
von Mises stress; unfailed yielded particles still sit on or below the flow
surface. Typical result: ~600 of 1152 particles failed at t = 0.08 s.
