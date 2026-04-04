# ExaGOOP Development Context
## Single source of truth for continuing development

---

## Project Overview

- **Project**: ExaGOOP MPM (Material Point Method) solver — AMReX-based
- **Mac location**: `/Users/sreejith/Documents/01_Research/Code_Developments/ExaGOOP_Dev`
- **Windows location**: `E:\02_CODE_DEVELOPMENTS\02_EXAGOOP`
- **Build systems**: CMake (`Build_Cmake/`) and GNUmake (`Build_Gnumake/`)
- **Key source directory**: `Source/`
- **AMReX location**: `Submodules/amrex/`
- **UDF templates**: `Exec/udf_templates/`
- **Windows compiler**: MSVC (cl.exe) + NVCC (CUDA 13.2), VS Community 2022 v18 (MSVC 14.50)
- **GPU**: CUDA compute capability 86 (sm_86)
- **Build configs**: `cpu-debug`, `cuda86` under `Build_Cmake\`

---

## Git State

| Commit | Description |
|---|---|
| `5e7a993` | Production baseline — no level set BC, known good |
| `235793b` | First levelset implementation — partially working, has bugs (do not use) |
| `levelset-clean` | Current working branch — reimplement levelset cleanly here |

**Always work on `levelset-clean`. Do not merge from `235793b` — it has confirmed bugs documented below.**

---

## What Has Been Built

### 1. Windows/CUDA Porting Fixes

#### `CMake/BuildExaGOOPExe.cmake`
```cmake
# Before add_executable:
if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")
endif()

# After add_executable and target_include_directories:
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)
    target_compile_options(${EXAGOOP_EXE_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244 /wd4005 /Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4244 -Xcompiler=/wd4005>
    )
    set_source_files_properties(
        ${SRC_DIR}/mpm_eb_udf_build.cpp
        PROPERTIES LANGUAGE CXX
    )
endif()
```
Also add `mpm_eb_udf_build.cpp` to `target_sources`.

#### `Source/aesthetics.cpp` — Winsock fix
```cpp
#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <winsock2.h>
    #pragma comment(lib, "Ws2_32.lib")
#else
    #include <unistd.h>
#endif
```

#### `Source/mpm_eb.cpp` — min/max macro fix
```cpp
#ifdef _WIN32
    #undef min
    #undef max
#endif
```

#### `Source/mpm_eb.H` — Host/Device function chain
Add `AMREX_GPU_HOST_DEVICE` (were `AMREX_GPU_DEVICE` only) to:
- `get_levelset_value`
- `basisval`
- `quadraticspline_1d`
- `cubicspline_1d`

These are pure math functions safe on both host and device. Needed because
`get_levelset_value` is called from `amrex::LoopOnCpu` (CPU) in one location.

#### `Source/interpolants.H` — `__constant__` memory fix
```cpp
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
interval_idx_quadratic(int shapefunctiontype, amrex::Real zi)
{
    amrex::Real shift = (shapefunctiontype == 2) ? 1.0 : 1.5;
    int raw = static_cast<int>(std::floor((zi + shift) * 2));
    if (raw < 0 || raw >= 6) return -1;
#ifdef __CUDA_ARCH__
    return interval_map_quadbspline[shapefunctiontype][raw];
#else
    constexpr int interval_map_quadbspline_cpu[5][6] = {
        {-1, -1, -1, -1, -1, -1},
        {0, 0, 1, 2, 3, 3},
        {0, 1, 1, 2, 2, -1},
        {0, 0, 1, 1, 2, 2},
        {0, 0, 1, 1, 2, -1}
    };
    return interval_map_quadbspline_cpu[shapefunctiontype][raw];
#endif
}
```

#### New file: `Source/mpm_eb_udf_build.cpp`
Compiled as CXX only (never by nvcc) to avoid nvcc stub error with `EB2::Build` + UDF lambda.
```cpp
#include <mpm_eb.H>
#include <AMReX_EB2.H>
#include <functional>

void build_udf_eb_only(
    std::function<amrex::Real(const amrex::RealArray&)> udf_if,
    const amrex::Geometry& geom,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm,
    int nghost,
    int ls_refinement_in,
    amrex::MultiFab*& lsphi_out,
    amrex::EBFArrayBoxFactory*& ebfactory_out)
{
    auto gshop = amrex::EB2::makeShop(udf_if);
    Box dom_ls = geom.Domain();
    dom_ls.refine(ls_refinement_in);
    amrex::Geometry geom_ls(dom_ls);
    int required_coarsening_level = 0;
    if (ls_refinement_in > 1) {
        int tmp = ls_refinement_in;
        while (tmp >>= 1) ++required_coarsening_level;
    }
    amrex::EB2::Build(gshop, geom_ls, required_coarsening_level, 10);
    const amrex::EB2::IndexSpace& ebis  = amrex::EB2::IndexSpace::top();
    const amrex::EB2::Level&      eblev = ebis.getLevel(geom);
    ebfactory_out = new amrex::EBFArrayBoxFactory(
        eblev, geom, ba, dm, {nghost, nghost, nghost}, amrex::EBSupport::full);
    amrex::BoxArray ls_ba = amrex::convert(ba, amrex::IntVect::TheNodeVector());
    ls_ba.refine(ls_refinement_in);
    lsphi_out = new amrex::MultiFab;
    lsphi_out->define(ls_ba, dm, 1, nghost);
}
```

Declaration in `Source/mpm_eb.H`:
```cpp
#include <functional>
namespace amrex { class EBFArrayBoxFactory; }
void build_udf_eb_only(
    std::function<amrex::Real(const amrex::RealArray&)> udf_if,
    const amrex::Geometry& geom, const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm, int nghost, int ls_refinement_in,
    amrex::MultiFab*& lsphi_out, amrex::EBFArrayBoxFactory*& ebfactory_out);
```

#### `Source/Make.package` — GNUmake (Linux only)
```makefile
CEXE_sources += mpm_eb_udf_build.cpp
```

---

### 2. UDF Level Set System

**New files:** `Source/mpm_udf_loader.H`, `CMake/udf_builder/CMakeLists.txt`,
`Tools/exagoop-build-udf.in`, `Tools/exagoop-build-udf.bat.in`, `Tools/GNUmakefile.udf`

Users write a `.cpp` exporting `extern "C" double levelset_phi(double x, double y, double z)`,
compile with `exagoop-build-udf`, point input file at the `.so`/`.dylib`. No recompilation needed.

`mpm_udf_loader.H`: Cross-platform `dlopen`/`LoadLibrary` wrapper. `UDFLoader` owns the handle,
`UDFImplicitFunction` adapts it to AMReX's ImplicitFunction concept.

**GPU safety:** UDF function pointers are CPU-only. Use `amrex::LoopOnCpu`, never `amrex::ParallelFor`.

**Building UDFs:**
```bash
# CMake:
exagoop-build-udf Exec/my_case/my_geometry.cpp Exec/my_case

# GNUmake (from Build_Gnumake/):
make udf UDF_SRC=../Exec/my_case/circle.cpp OUT=../Exec/my_case/udf_build/liblevelset_udf.dylib CXX=g++ -B
```

**Note:** The `.dylib` from CMake works with GNUmake executables and vice versa on the same machine.

**`Tools/GNUmakefile.udf` fix** — `COMP=gnu` was being passed as compiler binary directly.
Fix: map `COMP` to real compiler binary:
```makefile
ifeq ($(COMP),gnu)
    CXX_UDF := g++
else ifeq ($(COMP),llvm)
    CXX_UDF := clang++
else
    CXX_UDF := $(COMP)
endif

UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
    LIB_EXT := dylib
    LIB_FLAGS := -dynamiclib
else
    LIB_EXT := so
    LIB_FLAGS := -shared
endif

UDF_LIB := $(OUT)/liblevelset_udf.$(LIB_EXT)
# Use $(CXX_UDF) and $(UDF_LIB) in compile rule, add mkdir -p $(OUT) before compile
```

**Input:**
```
eb2.geom_type   = udf_cpp
eb2.udf_so_file = /path/to/liblevelset_udf.dylib
eb2.ls_refinement = 2
```

---

### 3. Level Set BC System (velocity)

**Files:** `nodal_data_ops.cpp`, `nodal_data_ops.H`, `mpm_specs.H`

`nodal_levelset_bcs()` enforces velocity BCs on nodes where φ < 0.

**BC types:**
| Type | Meaning |
|---|---|
| 0 | No BC |
| 1 | No-slip |
| 2 | Free-slip |
| 3 | Coulomb friction (`mu` parameter) |

**Input:**
```
mpm.levelset_bc      = 2
mpm.levelset_wall_mu = 0.3
```

---

### 4. Nodal Face Thermal BCs (types 0–5)

**Files:** `nodal_data_ops.cpp`, `nodal_data_ops.H`, `mpm_specs.H`, `utilities.cpp`
**New files:** `Source/mpm_thermal_udf_loader.H`, `Exec/udf_templates/thermal_udf_template.cpp`

`nodal_bcs_temperature_extended()` handles five BC types per domain face independently:

| Type | Meaning | Key parameter |
|---|---|---|
| 0 | No BC | — |
| 1 | Dirichlet | `T_wall` [K] — applied **after** time update |
| 2 | Adiabatic | — |
| 3 | Heat flux | `flux` [W/m²] — added to source **before** time update |
| 4 | Convective | `h` [W/m²/K] + `T_inf` [K] — added to source **before** time update |
| 5 | Convective UDF | `h(x,y,z)` and `T_inf(x,y,z)` from shared library |

**Call order in time step (critical):**
```
1. Apply_Nodal_BCs_Temperature(..., pre_update=true)   — types 3,4,5: add to source
2. Nodal_Time_Update_Temperature                        — integrate T
3. Apply_Nodal_BCs_Temperature(..., pre_update=false)   — type 1: override T
```

`utilities.H` — `Apply_Nodal_BCs_Temperature` declaration must have `bool pre_update` parameter.

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

### 5. Level Set Thermal BCs (on EB surface)

**Files:** `nodal_data_ops.cpp`, `nodal_data_ops.H`, `mpm_specs.H`, `utilities.cpp`

`nodal_levelset_bcs_temperature()` — same six types as nodal face thermal BCs,
applied to nodes where φ < 0. Same `average_down_nodal` + `FillBoundary` pattern required.

**`mpm_specs.H` new members:**
```cpp
int         levelset_bc_temp     = 0;
amrex::Real levelset_bc_temp_val = 0.0;
amrex::Real levelset_Tinf        = 0.0;
std::string levelset_temp_udf    = "";
```

---

### 6. Clean Input Format + RigidBodyManager

**New file:** `Source/mpm_rigidBody.H`

`RigidBodyManager` reads clean human-readable input and writes back into legacy arrays.

**New input format:**
```
mpm.use_levelset    = true
mpm.num_rigidbodies = 2

# Nodal velocity BCs (domain faces)
mpm.nodal_vel_bc.x_lo.type     = 2
mpm.nodal_vel_bc.y_lo.type     = 3
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

Legacy format (`mpm.bc_lower = 2 2 0` etc.) still works — new format overrides if present.
**Do not mix both formats for the same parameter.**

**Face names:** `x_lo`, `x_hi`, `y_lo`, `y_hi`, `z_lo`, `z_hi`
**Body names:** `rb_0`, `rb_1`, `rb_2`, ...

---

### 7. Multi-Body Static Rigid Bodies

**Modified:** `mpm_eb.cpp`, `mpm_eb.H`, `mpm_init.cpp`, `nodal_data_ops.cpp`,
`nodal_data_ops.H`, `mpm_rigidBody.H`, `utilities.cpp`, `main.cpp`

Each rigid body has its own `lsphi` MultiFab. Union lsphi (min over all bodies) used for
`removeParticlesInsideEB`. Per-body BCs applied using each body's own lsphi.

**New globals in `mpm_ebtools` namespace:**
```cpp
std::vector<MultiFab*> lsphi_bodies;
int num_lsphi_bodies;
```

**`main.cpp` addition** (after `init_eb`):
```cpp
specs.rb_manager.init_geometry(geom, ba, dm);
```

**Input:**
```
mpm.rb_0.geom_type   = udf_cpp
mpm.rb_0.udf_so_file = /path/to/circle.dylib
mpm.rb_1.geom_type   = stl
mpm.rb_1.stl_file    = /path/to/floor.stl
```

---

## Level Set Bug Fixes — CRITICAL

These bugs existed in `235793b` and must not be repeated in the clean reimplementation.

### Bug 1 — Missing `FillBoundary` on `lsphi_coarse` (phantom obstacles)

`average_down_nodal` fills only the valid region of `lsphi_coarse`. Ghost cells are
uninitialized. `get_levelset_grad` reads the ghost layer → NaN or garbage normals at every
tile boundary → phantom obstacles everywhere with small `max_grid_size`.

**Fix — required at EVERY `average_down_nodal` call site:**
```cpp
MultiFab lsphi_coarse(nodaldata.boxArray(), nodaldata.DistributionMap(), 1, /*nghost=*/1);
amrex::average_down_nodal(*mpm_ebtools::lsphi, lsphi_coarse, amrex::IntVect(lsref));
lsphi_coarse.FillBoundary(geom.periodicity());  // ← MUST NOT BE OMITTED
```

Required at all these sites:
- `nodal_data_ops.cpp` — single-body velocity BC
- `nodal_data_ops.cpp` — multi-body velocity BC
- `nodal_data_ops.cpp` — thermal BC
- `mpm_init.cpp` — particle removal
- `mpm_particle_timestep.cpp` — particle timestep constraint

**Why it worked before:** Original code used `FillSignedDistance` which handles ghost cells
internally. Switch to `LoopOnCpu` exposed the missing `FillBoundary`.

### Bug 2 — Missing `FillBoundary` on `lsphi` after `LoopOnCpu`

After filling `lsphi` via `LoopOnCpu`, `FillBoundary` was not called.

**Fix — after `LoopOnCpu` in `init_eb` / `fill_body_lsphi`:**
```cpp
Box dom_ls = geom.Domain();
dom_ls.refine(ls_refinement);
Geometry geom_ls(dom_ls);
lsphi->FillBoundary(geom_ls.periodicity());  // use REFINED geometry periodicity
```

### Bug 3 — TINYVAL division producing ~1e20 normals (velocity blowup / NaN particles)

```cpp
// OLD — dangerous:
normaldir[d] /= (gradmag + TINYVAL);  // TINYVAL ≈ 1e-20
```
Nodes deep inside the obstacle have zero gradient. Division by TINYVAL → ~1e20 normal →
velocity blowup → NaN particles → `locateParticle` crash.

**Fix — in EVERY `nodal_levelset_bcs` overload (single-body AND multi-body):**
```cpp
if (gradmag < 1.0e-10) return;  // skip degenerate nodes inside obstacle
for (int d = 0; d < AMREX_SPACEDIM; d++)
    normaldir[d] /= gradmag;
```

### Bug 4 — Coarse MFIter with refined `lsphi` in `mpm_particle_timestep.cpp`
```cpp
// WRONG:
lsetarr = mpm_ebtools::lsphi->array(mfi);  // mfi is coarse, lsphi is refined — wrong tile
```
**Fix:** `average_down_nodal` + `FillBoundary` + `lsphi_coarse.array(mfi)`.

---

## Correct `lsphi` Access Pattern

Every function reading `lsphi` on a coarse MFIter must follow this pattern exactly:

```cpp
void nodal_levelset_bcs(MultiFab& nodaldata, const Geometry& geom, ...)
{
    int lsref = mpm_ebtools::ls_refinement;

    MultiFab lsphi_coarse(nodaldata.boxArray(),
                          nodaldata.DistributionMap(),
                          1,    // ncomp
                          1);   // nghost — MUST be >= 1
    amrex::average_down_nodal(*mpm_ebtools::lsphi,
                               lsphi_coarse,
                               amrex::IntVect(lsref));
    lsphi_coarse.FillBoundary(geom.periodicity());  // coarse geom

    for (MFIter mfi(nodaldata); mfi.isValid(); ++mfi)
    {
        Array4<Real> lsarr = lsphi_coarse.array(mfi);  // NOT lsphi->array(mfi)

        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Real lsval = get_levelset_value(lsarr, plo, dx, xp, /*lsref=*/1);

            if (lsval >= 0.0 || nodal_data_arr(nodeid, MASS_INDEX) <= shunya)
                return;

            amrex::Real normaldir[AMREX_SPACEDIM];
            get_levelset_grad(lsarr, plo, dx, xp, /*lsref=*/1, normaldir);

            amrex::Real gradmag = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; d++)
                gradmag += normaldir[d] * normaldir[d];
            gradmag = std::sqrt(gradmag);

            if (gradmag < 1.0e-10) return;  // degenerate — do NOT divide

            for (int d = 0; d < AMREX_SPACEDIM; d++)
                normaldir[d] /= gradmag;

            amrex::Real relvel_in[AMREX_SPACEDIM], relvel_out[AMREX_SPACEDIM];
            for (int d = 0; d < AMREX_SPACEDIM; d++)
                relvel_in[d] = nodal_data_arr(nodeid, VELX_INDEX + d) - wall_vel[d];

            applybc(relvel_in, relvel_out, lset_wall_mu, normaldir, lsetbc);

            for (int d = 0; d < AMREX_SPACEDIM; d++)
                nodal_data_arr(nodeid, VELX_INDEX + d) = relvel_out[d] + wall_vel[d];
        });
    }
}
```

---

## Architecture

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

─── Time step loop ───────────────────────────────────────────────────────
P2G → Nodal_Time_Update_Momentum
    → Apply_Nodal_BCs (velocity + levelset_all_bodies)   ← AFTER momentum update
    → P2G_Temperature
    → Apply_Nodal_BCs_Temperature(pre=true)              ← types 3,4,5: source
    → Nodal_Time_Update_Temperature
    → Apply_Nodal_BCs_Temperature(pre=false)             ← type 1: Dirichlet
    → G2P → Update positions → Constitutive model
```

### Level Set BC Call Chain
```
Apply_Nodal_BCs(geom, nodaldata, specs, dt)         [utilities.cpp]
  → nodal_bcs(...)                                   domain wall BCs
  → if (using_levelset_geometry):
      nodal_levelset_bcs_all_bodies(nodaldata, geom, dt, specs.rb_manager)
          → if (lsphi_bodies.empty()):               FALLBACK single-body
              nodal_levelset_bcs(nodaldata, geom, dt,
                  rb_manager.bodies[0].vel_bc.type,
                  rb_manager.bodies[0].vel_bc.mu)
          → for each body b:                         MULTI-BODY
              nodal_levelset_bcs(nodaldata, geom, dt,
                  vbc.type, vbc.mu, lsphi_bodies[b])
```

---

## UDF File Format

```cpp
#include <cmath>

#ifdef _WIN32
  #define EXAGOOP_API __declspec(dllexport)
#else
  #define EXAGOOP_API __attribute__((visibility("default")))
#endif

extern "C" EXAGOOP_API double levelset_phi(double x, double y, double z)
{
    // φ < 0 INSIDE obstacle, φ > 0 outside, φ = 0 on surface
    const double cx = 0.25;
    const double cy = 0.07;
    const double r  = 0.05;
    return std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)) - r;
}
```

---

## Complete File Change List

| File | Status | What changed |
|---|---|---|
| `Source/mpm_eb.cpp` | Modified | UDF path, STL path, multi-body `init_eb_bodies`, `fill_body_lsphi`, union lsphi, `FillBoundary` after `LoopOnCpu` |
| `Source/mpm_eb.H` | Modified | `lsphi_bodies`, `num_lsphi_bodies`, `init_eb_bodies`, `build_udf_eb_only` declaration, `AMREX_GPU_HOST_DEVICE` on spline functions |
| `Source/mpm_eb_udf_build.cpp` | New | CXX-only `build_udf_eb_only` — avoids nvcc stub error |
| `Source/mpm_udf_loader.H` | New | Cross-platform dlopen wrapper |
| `Source/mpm_thermal_udf_loader.H` | New | Runtime loader for thermal BC UDFs |
| `Source/mpm_rigidBody.H` | New | `RigidBodyBC`, `RigidBodyManager`, clean input format parser |
| `Source/mpm_specs.H` | Modified | New thermal BC members, `rb_manager`, new ParmParse block |
| `Source/nodal_data_ops.cpp` | Modified | Fixed `nodal_levelset_bcs` (all bugs), thermal LS BCs, multi-body overloads, `FillBoundary` at all sites, `gradmag` guard |
| `Source/nodal_data_ops.H` | Modified | New declarations |
| `Source/mpm_init.cpp` | Modified | `average_down_nodal` + `FillBoundary` fix for `removeParticlesInsideEB` |
| `Source/mpm_particle_timestep.cpp` | Modified | `average_down_nodal` + `FillBoundary` fix — was using coarse MFIter with refined lsphi |
| `Source/utilities.cpp` | Modified | Multi-body BC calls, `pre_update` split for temperature |
| `Source/utilities.H` | Modified | `bool pre_update` parameter on `Apply_Nodal_BCs_Temperature` |
| `Source/main.cpp` | Modified | `rb_manager.init_geometry()` after `init_eb()` |
| `Source/aesthetics.cpp` | Modified | Winsock fix for Windows |
| `Source/interpolants.H` | Modified | `__CUDA_ARCH__` guard for `__constant__` memory |
| `CMakeLists.txt` | Modified | UDF install rules, STLGeom detection, dl linkage, chmod |
| `CMake/BuildExaGOOPExe.cmake` | Modified | Windows flags, `mpm_eb_udf_build.cpp` as CXX |
| `CMake/udf_builder/CMakeLists.txt` | New | Satellite CMake for UDF `.so` builds |
| `Tools/exagoop-build-udf.in` | New | Linux/macOS UDF build helper |
| `Tools/exagoop-build-udf.bat.in` | New | Windows UDF build helper |
| `Tools/GNUmakefile.udf` | Modified | Fixed `COMP=gnu` → `g++`, `OUT` as directory |
| `Build_Gnumake/GNUmakefile` | Modified | `udf` and `udf-help` targets |
| `Source/Make.package` | Modified | `mpm_eb_udf_build.cpp` for GNUmake (Linux only) |
| `Exec/udf_templates/` | New | levelset and thermal UDF templates |

---

## Known Issues / Outstanding Work

1. **Moving rigid bodies** — `RigidBodyManager` is Phase 1 (static only). Phase 2 (dynamics) not yet implemented
2. **Analytic multi-body** — for multiple analytic EB2 shapes as separate bodies, use the advanced UDF template — `eb2.*` namespace can only hold one geometry at a time
3. **`mpm.use_levelset` flag** — declared in `RigidBodyManager` but relationship with `eb2.geom_type` is deferred
4. **GPU + moving bodies** — when bodies move, `lsphi` must be refilled each step; `LoopOnCpu` is correct but may bottleneck on GPU for many bodies
5. **Level set BC reimplementation** — currently on branch `levelset-clean`; `nodal_levelset_bcs` needs clean rewrite applying all four bug fixes above before merging

---

## Windows Porting Key Lessons

| Problem | Root Cause | Fix |
|---|---|---|
| `unistd.h` not found | POSIX header | Use `winsock2.h` on Windows |
| `gethostname` not found | Lives in Winsock2 on Windows | `#include <winsock2.h>` + link `Ws2_32.lib` |
| `min`/`max` macro conflicts | `windows.h` defines them as macros | `NOMINMAX` + `#undef min/max` |
| `sockaddr` redefinition | `winsock.h` before `winsock2.h` | `WIN32_LEAN_AND_MEAN` |
| C4244 warning | `long` is 32-bit on MSVC | `/wd4244` suppress |
| C4005 warning | AMReX lexer header conflicts | `/wd4005` suppress |
| CCCL preprocessor error | CUDA 13.2 requires conforming preprocessor | `CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING` |
| `__device__` called from `__host__` | Function chain not marked `HOST_DEVICE` | Add `AMREX_GPU_HOST_DEVICE` to call chain |
| `__constant__` memory in host code | GPU constant memory not readable from CPU | `__CUDA_ARCH__` guard with CPU fallback |
| nvcc stub.c error with `EB2::Build` | UDF lambda with dlopen pointer | Move `EB2::Build` to CXX-only file |
| LNK2005 multiply defined symbol | Function defined in header in multiple TUs | Add `inline` keyword |

---

## Build Status

- ✅ Mac CPU (GNUmake) — working
- ✅ Windows CPU debug (CMake) — working
- ✅ Windows CUDA release (CMake) — working
- ⚠️ Level set BCs — on `levelset-clean` branch, reimplementation in progress
