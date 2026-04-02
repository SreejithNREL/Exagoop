# ExaGOOP Development Summary
## Context Document for Continuing Development

---

## What ExaGOOP Is

ExaGOOP is an AMReX-based Material Point Method (MPM) solver. It uses a background Eulerian grid and Lagrangian material points. The code supports 2D and 3D, MPI parallelism, GPU (CUDA/HIP/SYCL), and embedded boundary (EB) geometry via AMReX's EB2 infrastructure.

**Build systems:** CMake (`Build_Cmake/`) and GNUmake (`Build_Gnumake/`)  
**Key source directory:** `Source/`  
**AMReX location:** `Submodules/amrex/`  
**UDF templates:** `Exec/udf_templates/`

---

## What Was Built in This Session

### 1. Level Set BC System (velocity)

**Files changed:** `nodal_data_ops.cpp`, `nodal_data_ops.H`, `mpm_specs.H`

`nodal_levelset_bcs()` was a stub — it computed the wall normal correctly but never called `applybc()`. Fixed to enforce all four velocity BC types on nodes where φ < 0:

- `0` = no BC
- `1` = no-slip
- `2` = free-slip
- `3` = Coulomb friction (μ = `levelset_wall_mu`)

**Key fix:** `lsphi` is on a refined nodal BoxArray; calling `lsphi->array(mfi)` with a coarse MFIter was reading the wrong tile. Fixed using `average_down_nodal` to produce a coarse copy before the MFIter loop. All subsequent level set functions use this same fix.

**Input:**
```
mpm.levelset_bc      = 2
mpm.levelset_wall_mu = 0.3
```

---

### 2. UDF Level Set System (runtime geometry, no recompilation)

**New files:** `Source/mpm_udf_loader.H`, `CMake/udf_builder/CMakeLists.txt`, `Tools/exagoop-build-udf.in`, `Tools/exagoop-build-udf.bat.in`, `Tools/GNUmakefile.udf`  
**Modified:** `Source/mpm_eb.cpp`, `CMakeLists.txt`, `Build_Gnumake/GNUmakefile`

Users write a `.cpp` file exporting `extern "C" double levelset_phi(double x, double y, double z)`, compile it into a `.so`/`.dylib`/`.dll` using `exagoop-build-udf`, and point the input file at it. No ExaGOOP recompilation needed.

**`mpm_udf_loader.H`:** Cross-platform `dlopen`/`LoadLibrary` wrapper. `UDFLoader` owns the handle, `UDFImplicitFunction` adapts it to AMReX's ImplicitFunction concept. `get_fn()` returns the raw function pointer for use in lambdas.

**GPU safety:** `fill_lsphi_from_udf` uses `amrex::LoopOnCpu` instead of `amrex::ParallelFor` — UDF function pointers are CPU-only and cannot be called from GPU kernels.

**`FillSignedDistance` problem:** With small boxes (`max_grid_size=16`), `FillSignedDistance` only propagates within each box's ghost-cell halo, giving constant ±0.008 values everywhere. Fix: fill `lsphi` directly from the analytic UDF using `LoopOnCpu`.

**CMake build:**
```bash
exagoop-build-udf Exec/my_case/my_geometry.cpp Exec/my_case
```
**GNUmake build:**
```bash
make udf UDF_SRC=../Exec/my_case/my_geometry.cpp
```

**Input:**
```
eb2.geom_type   = udf_cpp
eb2.udf_so_file = /path/to/Exec/my_case/udf_build/liblevelset_udf.dylib
eb2.ls_refinement = 2
```

**STL path** also added (`eb2.geom_type = stl`). Guarded with `EXAGOOP_AMREX_HAS_STLGEOM` CMake feature detection since `EB2::STLGeom` was added in AMReX ~23.05.

---

### 3. Nodal Face Thermal BCs (types 0–5)

**Files changed:** `nodal_data_ops.cpp`, `nodal_data_ops.H`, `mpm_specs.H`, `utilities.cpp`  
**New files:** `Source/mpm_thermal_udf_loader.H`, `Exec/udf_templates/thermal_udf_template.cpp`, `Exec/udf_templates/thermal_udf_varying_template.cpp`

New function `nodal_bcs_temperature_extended()` handles five BC types per domain face independently:

- `0` = no BC
- `1` = Dirichlet — set T = T_wall (applied **after** time update)
- `2` = Adiabatic — zero flux (no-op)
- `3` = Heat flux — add q·A to `SOURCE_TEMP_INDEX` (**before** time update)
- `4` = Convective — add h·(T_inf − T)·A to source (**before** time update)
- `5` = Convective UDF — h(x,y,z) and T_inf(x,y,z) from shared library

**Call order in time step (critical):**
```
1. Apply_Nodal_BCs_Temperature(..., pre_update=true)   — types 3,4,5: add to source
2. Nodal_Time_Update_Temperature                        — integrate T
3. Apply_Nodal_BCs_Temperature(..., pre_update=false)   — type 1: override T
```

`Apply_Nodal_BCs_Temperature` in `utilities.cpp` now takes a `bool pre_update` parameter. `utilities.H` must also declare this parameter.

**Input:**
```
mpm.nodal_temp_bc.y_lo.type   = 3
mpm.nodal_temp_bc.y_lo.flux   = 5000.0
mpm.nodal_temp_bc.y_hi.type   = 4
mpm.nodal_temp_bc.y_hi.h      = 150.0
mpm.nodal_temp_bc.y_hi.T_inf  = 300.0
mpm.nodal_temp_bc.x_lo.type   = 5
mpm.nodal_temp_bc.x_lo.udf    = /path/to/thermal.dylib
```

---

### 4. Level Set Thermal BCs (on EB surface)

**Files changed:** `nodal_data_ops.cpp`, `nodal_data_ops.H`, `mpm_specs.H`, `utilities.cpp`

New function `nodal_levelset_bcs_temperature()` — same six types as nodal face thermal BCs, applied to nodes where φ < 0. Same `average_down_nodal` fix for lsphi access. Types 3/4/5 use `LoopOnCpu` for UDF function pointers.

Nodal area for curved EB: average of `dx[1]·dx[2]`, `dx[0]·dx[2]`, `dx[0]·dx[1]` — exact for flat walls, good approximation for smooth curves.

**`mpm_specs.H` new members:**
```cpp
int         levelset_bc_temp     = 0;
amrex::Real levelset_bc_temp_val = 0.0;
amrex::Real levelset_Tinf        = 0.0;
std::string levelset_temp_udf    = "";
```

**Input:**
```
mpm.levelset_bc_temp     = 4
mpm.levelset_bc_temp_val = 200.0
mpm.levelset_Tinf        = 350.0
```

---

### 5. Clean Input Format + RigidBodyManager

**New file:** `Source/mpm_rigidBody.H`  
**Modified:** `Source/mpm_specs.H`

`RigidBodyManager` class reads a clean human-readable input format and writes back into the legacy arrays so the rest of ExaGOOP is unchanged.

**New input format:**
```
mpm.use_levelset    = true
mpm.num_rigidbodies = 2

# Nodal velocity BCs (domain faces)
mpm.nodal_vel_bc.x_lo.type     = 2    # 0=none 1=noslip 2=slip 3=Coulomb
mpm.nodal_vel_bc.y_lo.type     = 2
mpm.nodal_vel_bc.y_lo.mu       = 0.3
mpm.nodal_vel_bc.x_lo.velocity = 0.0 0.0 0.0

# Nodal temperature BCs (domain faces)
mpm.nodal_temp_bc.y_lo.type   = 1
mpm.nodal_temp_bc.y_lo.T_wall = 300.0
mpm.nodal_temp_bc.y_hi.type   = 4
mpm.nodal_temp_bc.y_hi.h      = 150.0
mpm.nodal_temp_bc.y_hi.T_inf  = 300.0

# Level-set velocity BCs (per rigid body)
mpm.lset_vel_bc.rb_0.type    = 2
mpm.lset_vel_bc.rb_1.type    = 1

# Level-set temperature BCs (per rigid body)
mpm.lset_temp_bc.rb_0.type   = 4
mpm.lset_temp_bc.rb_0.h      = 200.0
mpm.lset_temp_bc.rb_0.T_inf  = 350.0
mpm.lset_temp_bc.rb_1.type   = 1
mpm.lset_temp_bc.rb_1.T_wall = 300.0
```

**Legacy format** (`mpm.bc_lower = 2 2 0` etc.) still works — new format overrides if present.

**Face names:** `x_lo`, `x_hi`, `y_lo`, `y_hi`, `z_lo`, `z_hi`  
**Body names:** `rb_0`, `rb_1`, `rb_2`, ...

---

### 6. Multi-Body Static Rigid Bodies

**Modified:** `Source/mpm_eb.cpp`, `Source/mpm_eb.H`, `Source/mpm_init.cpp`, `Source/nodal_data_ops.cpp`, `Source/nodal_data_ops.H`, `Source/mpm_rigidBody.H`, `Source/utilities.cpp`, `Source/main.cpp`

Each rigid body now has its own `lsphi` MultiFab. The union lsphi (min over all bodies) is used for `removeParticlesInsideEB`. Per-body BCs are applied using each body's own lsphi and BC parameters.

**New globals in `mpm_ebtools` namespace:**
```cpp
std::vector<MultiFab*> lsphi_bodies;   // one per body
int num_lsphi_bodies;
```

**New functions:**
- `fill_body_lsphi(body_id, ...)` — fills one body's lsphi from its geometry
- `init_eb_bodies(geom, ba, dm, num_bodies)` — initialises all bodies, builds union lsphi
- `nodal_levelset_bcs(nodaldata, geom, dt, type, mu, body_lsphi*)` — per-body overload
- `nodal_levelset_bcs_all_bodies(nodaldata, geom, dt, rb_manager)` — loops over all bodies
- `nodal_levelset_bcs_temperature_all_bodies(...)` — thermal equivalent
- `RigidBodyManager::init_geometry(geom, ba, dm)` — call after `init_eb()`

**`main.cpp` addition** (after `init_eb`):
```cpp
specs.rb_manager.init_geometry(geom, ba, dm);
```

**Supported geometry types per body:**
- `udf_cpp` — user `.so` file
- `stl` — surface mesh
- analytic EB2 shapes (sphere, plane, etc.)
- `wedge_hopper` (3D only)

**Input:**
```
mpm.rb_0.geom_type   = udf_cpp
mpm.rb_0.udf_so_file = /path/to/circle.dylib
mpm.rb_1.geom_type   = stl
mpm.rb_1.stl_file    = /path/to/floor.stl
```

**Plotfiles written:** `ebplt_rb0/`, `ebplt_rb1/`, `ebplt_union/`

---

## Complete File Change List

| File | Status | What changed |
|---|---|---|
| `Source/mpm_eb.cpp` | Modified | UDF path, STL path, multi-body `init_eb_bodies`, `fill_body_lsphi`, union lsphi |
| `Source/mpm_eb.H` | Modified | Added `lsphi_bodies`, `num_lsphi_bodies`, `init_eb_bodies` declaration |
| `Source/mpm_udf_loader.H` | New | Cross-platform dlopen wrapper, `UDFLoader`, `UDFImplicitFunction` |
| `Source/mpm_thermal_udf_loader.H` | New | Runtime loader for thermal BC UDFs (`thermal_h`, `thermal_Tinf`) |
| `Source/mpm_rigidBody.H` | New | `RigidBodyBC`, `RigidBodyManager`, clean input format parser |
| `Source/mpm_specs.H` | Modified | New thermal BC members, `rb_manager`, new ParmParse block |
| `Source/nodal_data_ops.cpp` | Modified | Fixed `nodal_levelset_bcs`, added temperature LS BCs, multi-body overloads |
| `Source/nodal_data_ops.H` | Modified | New declarations for all new functions |
| `Source/mpm_init.cpp` | Modified | Fixed `removeParticlesInsideEB` (average_down_nodal fix) |
| `Source/utilities.cpp` | Modified | Updated BC calls to use multi-body functions, `pre_update` split |
| `Source/utilities.H` | Modified | Added `bool pre_update` to `Apply_Nodal_BCs_Temperature` |
| `Source/main.cpp` | Modified | Added `rb_manager.init_geometry()` after `init_eb()` |
| `CMakeLists.txt` | Modified | UDF install rules, STLGeom detection, dl linkage, chmod |
| `CMake/udf_builder/CMakeLists.txt` | New | Satellite CMake for building user UDF `.so` files |
| `Tools/exagoop-build-udf.in` | New | Linux/macOS helper script (configured by CMake) |
| `Tools/exagoop-build-udf.bat.in` | New | Windows equivalent |
| `Tools/GNUmakefile.udf` | New | GNUmake UDF build helper |
| `Build_Gnumake/GNUmakefile` | Modified | Added `udf` and `udf-help` targets |
| `Exec/udf_templates/levelset_simple_template.cpp` | New | Simple φ formula template |
| `Exec/udf_templates/levelset_advanced_template.cpp` | New | Full AMReX EB2 compositing template |
| `Exec/udf_templates/thermal_udf_template.cpp` | New | Constant h and T_inf thermal UDF template |
| `Exec/udf_templates/thermal_udf_varying_template.cpp` | New | Spatially varying thermal UDF template |

---

## Known Issues / Outstanding Work

1. **`utilities.H`** — `Apply_Nodal_BCs_Temperature` declaration must have `bool pre_update` parameter added manually
2. **Moving rigid bodies** — `RigidBodyManager` is Phase 1 (static only). Phase 2 (dynamics: free motion, spring-damper, imposed velocity) not yet implemented
3. **Analytic multi-body** — for multiple analytic EB2 shapes (sphere+plane etc.) as separate bodies, users should use the advanced UDF template rather than separate `geom_type` entries, since EB2 analytic shapes read from `eb2.*` namespace which can only hold one geometry at a time
4. **`mpm.use_levelset` flag** — declared in `RigidBodyManager` but the decision of whether it replaces or sits alongside `eb2.geom_type` is deferred
5. **GPU + moving bodies** — when bodies move, `lsphi` must be refilled each step; `LoopOnCpu` is correct but may be a bottleneck on GPU for many bodies

---

## Architecture Diagram

```
Input file (mpm.* keys)
        │
        ▼
MPMspecs::read_mpm_specs()
        │
        ▼
RigidBodyManager::init()          ← parses rb_N BCs, writes back to legacy arrays
        │
        ▼
init_eb(geom, ba, dm)             ← legacy single-body eb2.* path
        │
        ▼
rb_manager.init_geometry()        ← multi-body: fills lsphi_bodies, union lsphi
        │
        ▼
removeParticlesInsideEB()         ← uses union lsphi (min over all bodies)

─── Time step loop ───────────────────────────────────────────────────────────
P2G → Nodal time update → Apply_Nodal_BCs (velocity + levelset_all_bodies)
    → P2G_Temperature → Apply_Nodal_BCs_Temperature(pre=true)
    → Nodal_Time_Update_Temperature
    → Apply_Nodal_BCs_Temperature(pre=false)
    → G2P → Update positions → Constitutive model
```

---

## Quick Reference: BC Type Integers

### Velocity BCs (nodal and level set)
| Type | Meaning |
|---|---|
| 0 | No BC |
| 1 | No-slip |
| 2 | Free-slip |
| 3 | Coulomb friction |

### Temperature BCs (nodal and level set)
| Type | Meaning | Key parameter |
|---|---|---|
| 0 | No BC | — |
| 1 | Dirichlet | `T_wall` [K] |
| 2 | Adiabatic | — |
| 3 | Heat flux | `flux` [W/m²] |
| 4 | Convective | `h` [W/m²/K] + `T_inf` [K] |
| 5 | Convective UDF | `udf` = path to .so |
