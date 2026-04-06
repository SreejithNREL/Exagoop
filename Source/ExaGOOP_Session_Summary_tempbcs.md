# ExaGOOP Development Session Summary
## Branch: `generic_levelset_capabilities`
## Repos
- Mac: `/Users/sreejith/Documents/01_Research/Code_Developments/ExaGOOP_Dev`
- Windows/WSL: `/mnt/e/02_CODE_DEVELOPMENTS/02_EXAGOOP`

---

## 1. COMPLETED: Levelset Geometry Backend
**Branch:** `levelset-clean` (merged into `generic_levelset_capabilities`)

Three-path unified `init_eb()` in `Source/mpm_eb.cpp`:
- Path A `udf_cpp`: dlopen UDF `.so`
- Path B `stl`: AMReX `EB2::Build`
- Path C analytic: AMReX built-in shapes

---

## 2. COMPLETED: Nodal Thermal BCs (Types 1, 3, 4)

### Architecture
All BC types use **ghost-point approach** applied post `deposit_onto_grid_temperature`.

| Type | Name | Formula |
|------|------|---------|
| 0/2 | None/Adiabatic | `T_b = T_interior` (extrapolate to prevent spurious T=0) |
| 1 | Dirichlet | `T_b = T_wall` |
| 3 | Heat flux | `T_b = T_interior + (q/k)*dx` |
| 4 | Convective | `T_b = (T_int + Bi*T_inf)/(1+Bi)`, `Bi=h*dx/k` |

### Key Design Decisions
- `bc_applied` flag prevents corner nodes being overwritten by weaker BCs from other directions
- `store_delta_temperature` called explicitly in `main.cpp` while `MASS_SPHEAT` still populated from first P2G
- `Apply_Nodal_BCs_Temperature(dirichlet_only)` parameter: `true` in outer block for scheme 1 (ghost-point deferred to corrector), `false` for scheme 0 and scheme 1 corrector
- `Nodal_Time_Update_Temperature` sets T=0 at boundary nodes (no particles) — all non-Dirichlet BCs must extrapolate from interior to prevent spurious flux

### Files Changed
| File | Change |
|------|--------|
| `Source/nodal_data_ops.H` | New 9-arg `nodal_bcs_temperature` + `nodal_levelset_bcs_temperature` declarations |
| `Source/nodal_data_ops.cpp` | Ghost-point implementation of all BC types + `nodal_levelset_bcs_temperature` (type 1 only on EB) |
| `Source/utilities.H` | `Apply_Nodal_BCs_Temperature` with `dirichlet_only=false` default |
| `Source/utilities.cpp` | Updated call + removed `store_delta_temperature` (moved to main.cpp) |
| `Source/mpm_specs.H` | Added `bclo_Tinf`, `bchi_Tinf`, `levelset_temp_*` members |
| `Source/mpm_init.cpp` | Fixed old 6-arg call → new 9-arg with `bclo_temp.data()` |
| `Source/main.cpp` | scheme-aware BC application + explicit `store_delta_temperature` |

### Input Keys
```
mpm.bc_lower_temp      = 1 0 0   # type per face (0=none,1=Dir,2=adiab,3=flux,4=conv)
mpm.bc_upper_temp      = 4 0 0
mpm.bc_lower_tempval   = 0.0 0.0 0.0   # T_wall / q / h
mpm.bc_upper_tempval   = 2.0 0.0 0.0
mpm.bc_lower_Tinf      = 0.0 0.0 0.0
mpm.bc_upper_Tinf      = 0.0 0.0 0.0
mpm.levelset_temp_bc   = 1        # EB thermal BC type
mpm.levelset_temp_Twall = 300.0
mpm.levelset_temp_flux  = 0.0
mpm.levelset_temp_h     = 0.0
mpm.levelset_temp_Tinf  = 0.0
```

### Validated Test Cases
| Test | Location | RMS | t |
|------|----------|-----|---|
| 1D/2D Dirichlet | `Tests/1D_Heat_Conduction/`, `Tests/2D_Heat_Conduction/` | 2.7e-02 | 0.05 |
| 2D Heat Flux | `Tests/1D_Heat_Conduction_HeatFlux/` | 2.87e-03 | 2.0 |
| 2D Convective | `Exec/2D_Convective_HT/` | 3.68e-03 | 0.5 |

### Convective Test Exact Solution (corrected)
Steady-state decomposition: `T = T_ss(x) + transient`
- `T_ss = T_wall + (T_inf - T_wall)*(h/k)*x / (1 + h*L/k)`
- Eigenvalues: `λ*cos(λL) + (h/k)*sin(λL) = 0`
- Transient coefficients via `quad(lambda xi: -T_ss(xi)*sin(lam*xi), 0, L)`

---

## 3. COMPLETED: Levelset Thermal BC Types 3/4 on EB Surface

Implemented in `nodal_levelset_bcs_temperature` in `nodal_data_ops.cpp`:
- Type 3: `T_node = T_current + (q/k)*d` where `d = |lsval|`
- Type 4: `T_node = (T_current + Bi*T_inf)/(1+Bi)` where `Bi = h*d/k`
- **k=1 assumed** — variable k is a TODO (next task)

---

## 4. PENDING: Variable k in EB Levelset Thermal BCs

Current `nodal_levelset_bcs_temperature` assumes `k=1` for types 3 and 4.
Need to either:
- Pass `k` as a parameter to `nodal_levelset_bcs_temperature`
- Or read it from particle data nodally

---

## 5. PENDING: 2D Cylinder Dirichlet EB Thermal Test Case

Test case files generated but NOT yet run/validated:
- `Exec/2D_Cylinder_DirichletT/Inputs_2DCylinderDirichletT.inp`
- `Exec/2D_Cylinder_DirichletT/PreProcess/config.json`
- `Exec/2D_Cylinder_DirichletT/PostProcess/validate.py`
- `Exec/2D_Cylinder_DirichletT/PostProcess/Plot_Temperature.py`
- `Exec/2D_Cylinder_DirichletT/Generate_MPs_and_InputFiles.sh`

**Setup:** Square `[0,1]²`, cylinder at `(0.5,0.5)` R=0.15, EB Dirichlet T=1, outer faces Dirichlet T=0.
**Exact solution:** `T(r) = ln(r/0.5) / ln(0.15/0.5)` (annular 2D)
**Validation:** RMS vs exact restricted to `0.17 < r < 0.42`, tolerance 5e-02

---

## 6. COMPLETED: CUDA Fixes on `generic_levelset_capabilities`

### Problem: Dam break dt collapse on CUDA
**Root causes identified and fixed:**

#### 6a. Stream sync missing between icolor loops (`mpm_particle_grid_ops.cpp`)
`amrex::ParallelFor` launches GPU kernels asynchronously — without sync, all color groups execute concurrently, defeating the coloring scheme and causing race conditions.
```cpp
// Added after each ParallelFor closing inside icolor loop:
amrex::Gpu::streamSynchronize();
```
**Lines:** after `deposit_onto_grid_momentum` icolor close, after `deposit_onto_grid_temperature` icolor close.

#### 6b. lsphi_coarse DM mismatch (`mpm_particle_timestep.cpp`)
Using `lsphi->DistributionMap()` with particle MFIter causes CUDA error 700.
```cpp
// Fix in moveParticles():
BoxArray nodal_ba = amrex::convert(ParticleBoxArray(lev), IntVect::TheNodeVector());
lsphi_coarse.define(nodal_ba, ParticleDistributionMap(lev), 1, 1);
// Use lsphi_coarse.array(mfi) NOT mpm_ebtools::lsphi->array(mfi)
```

#### 6c. Density guard in `Calculate_time_step` (`mpm_particle_timestep.cpp`)
Negative density from J<0 gives NaN wave speed, corrupting `ReduceMin` on GPU.
```cpp
amrex::Real rho = p.rdata(realData::density);
if (rho <= 0.0) return std::numeric_limits<amrex::Real>::max();
```

#### 6d. Wrong 2D Newtonian fluid stress (`constitutive_models.H`)
2D case incorrectly used `1/2` instead of `1/3` for deviatoric subtraction.
```cpp
// Wrong:
amrex::Real one_by_two = 0.5;
sigma[XX] = 2μ*(epsdot[XX] - one_by_two*trace) - p;
// Correct:
amrex::Real trace_epsdot = epsdot[XX] + epsdot[YY] + epsdot[ZZ];
amrex::Real one_by_three = 1.0 / 3.0;
sigma[XX] = 2μ*(epsdot[XX] - one_by_three*trace) - p;
```

#### 6e. GPU qualifier fixes (`interpolants.H`, `mpm_eb.H`)
`basisval`, `basisvalder`, `basisval_and_grad`, `interval_idx_quadratic`,
`get_levelset_value`, `get_levelset_grad` changed from `AMREX_GPU_DEVICE` to
`AMREX_GPU_HOST_DEVICE` so they can be called from host+device contexts.

#### 6f. `build_udf_eb_only` declaration unguarded (`mpm_eb.H`)
`EBFArrayBoxFactory` used outside `#if USE_EB` causing compile failure when EB disabled.
```cpp
// Wrapped declaration in #if USE_EB
#if USE_EB
void build_udf_eb_only(..., amrex::EBFArrayBoxFactory *&ebfactory_out);
namespace mpm_ebtools { ... }
#endif
```

### Status
Currently in the middle of a git rebase:
```
* (no branch, rebasing generic_levelset_capabilities)
  generic_levelset_capabilities
  levelset-clean
  main
```
**Next step:** Complete rebase then rebuild and test dam break on CUDA.

---

## 7. PENDING: Moving Rigid Bodies (RigidBodyManager Phase 2)

Not yet started.

## 8. PENDING: Multi-body Level Set

Not yet started.

---

## Key Debugging Lessons

1. **Partial commit NaN bug**: Header declares 9-arg function, implementation has 6-arg — compiles but corrupts stack at runtime.
2. **`store_delta_temperature` timing**: Must run while `MASS_SPHEAT` populated from first P2G. Corrector P2G resets MASS_SPHEAT.
3. **Boundary node T=0**: `Nodal_Time_Update_Temperature` sets T=0 at boundary (no particles). All non-Dirichlet BCs must extrapolate from interior.
4. **Corner node overwrite**: `dir` loop applies BCs per direction — later direction overwrites earlier. `bc_applied` flag prevents weaker BCs overwriting stronger ones.
5. **CUDA icolor race**: `amrex::ParallelFor` async launches — all color groups run concurrently without explicit `streamSynchronize()`.
6. **CUDA DM mismatch**: `lsphi->DistributionMap()` ≠ particle DM → illegal memory access on GPU (CUDA error 700).
7. **AMReX ASCII particle output**: Always writes 3 position columns regardless of `AMREX_SPACEDIM`. Temperature column = `3 + enum_index`.

---

## AMReX Particle File Format
```
line 1: num_particles
line 2: num_real_fields (55 with USE_TEMP in 2D — includes 2 AMReX internal fields)
line 3: num_int_fields (3)
line 4: 0
line 5: 0
line 6+: x y z real[0]...real[54] int[0] int[1] int[2]
```
Temperature column (0-indexed): `3 + 46 = 49`

## realData Enum (with USE_TEMP)
```
temperature=46, specific_heat=47, thermal_conductivity=48,
heat_flux=49 (SPACEDIM components), heat_source=52, count=53
```
