# 2D vertical blind crack under vertical load (technology demo, coarse)

One linear-elastic block (material 0) with a finger-shaped crack cut from
the base: mouth half-width 0.04, elliptic taper to a blind tip at y = 0.55,
rough walls (two sinusoids, different phases left/right). A heavy platen
(material 1, rho 1e5 x rock) with prescribed velocity compresses the block
vertically; the side walls are rollers (slip), so the blocked Poisson
expansion produces sigma_xx = nu/(1-nu) sigma_yy, which is what closes the
crack. The tip carries a stress concentration under the vertical load.

Coarse nondimensional first pass (50 x 70 cells, 2x2 ppc, ~9.5k particles,
E = 1000, rho = 1000, nu = 0.3, v_platen = 0.005). Not to scale with rock.

```bash
./Generate_MPs_and_InputFiles.sh
./ExaGOOP2d.gnu.MPI.ex Inputs_VerticalBlindCrack.inp
python3 PostProcess/crack_shape_analysis.py        # crack_morphology.png, crack_shapes.png, tip_stress.png, crack_closure.png
```

Checks: far-field sigma_xx/sigma_yy should approach nu/(1-nu) = 0.43;
mouth and mean aperture and open crack length vs sigma_yy give the closure
story; tip_stress.png shows the max-principal-stress concentration at the
tip (where a wing crack would nucleate once damage/plasticity is in).
Stop reading the curve once the mouth is at the particle spacing.
