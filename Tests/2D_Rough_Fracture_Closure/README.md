# 2D rough-walled fracture closure under normal compression (coarse, exploratory)

Geothermal seed LDRD demo: two linear-elastic rock blocks with unmated rough
faces (sum of three sinusoids, different phases) separated by an initial
aperture, compressed by a heavy "rigid-like" platen (material 2, density
1e5 x rock) that carries a prescribed downward velocity. Displacement control
without a moving level set: because MPM uses a single velocity field, the
nodal velocity where the platen touches rock is mass-weighted ~ v_platen, and
the platen's momentum change from the rock reaction is negligible.

Nondimensional / coarse first pass (runs on a laptop): domain 1 x 1.4,
50 x 70 cells, 2x2 ppc (~9k particles), E = 1000, rho = 1000 (c = 1),
v_platen = 0.005 (closing the 0.10 mean gap over ~16 wave transits).
Not meant to look physical — it exercises the workflow. Refine (nx, ppc,
amplitude/dx ratio, v_platen/c) afterwards.

BCs: xlo/xhi slip (rollers), ylo noslip (fixed base), yhi slip.

```bash
./Generate_MPs_and_InputFiles.sh
./ExaGOOP2d.gnu.MPI.ex Inputs_RoughFractureClosure.inp      # or mpirun -n 4 ...
python3 PostProcess/aperture_analysis.py                    # -> closure_curve.png, aperture_maps.png, closure_data.csv
```

Things to watch on the first run: (1) sigma_n(base) vs sigma_n(interface)
should agree once the loading is quasi-static — if not, lower v_platen or
raise alpha_pic_flip damping; (2) the platen mean velocity should stay at
-0.005 (check TKE in Diagnostics); (3) if asperities interpenetrate, raise
ppc or reduce the roughness amplitude relative to dx.

Alternative load-control route (not used here): `mpm.external_loads = 1`
with `mpm.force_slab_lo/hi` and `mpm.extforce` applies a body force to a
slab — usable on the platen as a dead weight.
