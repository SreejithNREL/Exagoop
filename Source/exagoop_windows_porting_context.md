# ExaGOOP Windows Porting Context
## For continuing the thread in a new chat

---

## Project Overview
- **Project**: ExaGOOP MPM (Material Point Method) solver
- **Location**: `E:\02_CODE_DEVELOPMENTS\02_EXAGOOP`
- **Build system**: CMake (primary, used for Windows) + GNUmake (Linux/HPC)
- **Submodule**: AMReX at `E:\02_CODE_DEVELOPMENTS\02_EXAGOOP\Submodules\amrex`
- **Compiler**: MSVC (cl.exe) + NVCC (CUDA 13.2) on Windows
- **VS Version**: Visual Studio Community 2022 v18 (MSVC 14.50)
- **GPU**: CUDA compute capability 86 (sm_86)
- **Build configs**: `cpu-debug`, `cuda86` under `Build_Cmake\`

---

## All Fixes Applied So Far

### 1. CMake Windows Flags (`CMake/BuildExaGOOPExe.cmake`)
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

### 2. `Source/aesthetics.cpp` — Winsock fix
```cpp
// At the very top, before any includes:
#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <winsock2.h>
    #pragma comment(lib, "Ws2_32.lib")
#else
    #include <unistd.h>
#endif
```

### 3. `Source/mpm_eb.cpp` — min/max macro fix
```cpp
// After all includes:
#ifdef _WIN32
    #undef min
    #undef max
#endif
```

### 4. `Source/mpm_eb.H` — Host/Device function chain
The following functions needed `AMREX_GPU_HOST_DEVICE` added (were previously `AMREX_GPU_DEVICE` only):
- `get_levelset_value`
- `basisval`
- `quadraticspline_1d`
- `cubicspline_1d`

These are pure math functions safe to run on both host and device. They needed `HOST_DEVICE` because `get_levelset_value` is called from `amrex::LoopOnCpu` (CPU) in one location, and from `amrex::ParallelFor` (GPU) everywhere else.

### 5. `Source/interpolants.H` — `__constant__` memory fix
`interval_map_quadbspline` is declared as `__constant__` (GPU memory) on CUDA builds, which can't be read from host code. Fixed with `__CUDA_ARCH__` guard in `interval_idx_quadratic`:
```cpp
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
interval_idx_quadratic(int shapefunctiontype, amrex::Real zi)
{
    amrex::Real shift = (shapefunctiontype == 2) ? 1.0 : 1.5;
    int raw = static_cast<int>(std::floor((zi + shift) * 2));
    if (raw < 0 || raw >= 6)
        return -1;
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

### 6. New file: `Source/mpm_eb_udf_build.cpp`
Created to avoid nvcc stub compilation error caused by `EB2::Build(gshop,...)` with a UDF lambda. The file is forced to compile as CXX (not CUDA) via CMake. 

**Root cause**: When `EB2::Build` is called with a lambda-based `gshop` that captures a `dlopen`-loaded function pointer, nvcc's stub generator produces malformed C code that MSVC can't compile. Moving this to a CXX-only file fixes it.

```cpp
// Compiled as CXX only — never by nvcc
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

    amrex::Box dom_ls = geom.Domain();
    dom_ls.refine(ls_refinement_in);
    amrex::Geometry geom_ls(dom_ls);

    int required_coarsening_level = 0;
    if (ls_refinement_in > 1)
    {
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

### 7. `Source/mpm_eb.H` — Declaration for `build_udf_eb_only`
```cpp
#include <functional>
namespace amrex { class EBFArrayBoxFactory; }  // forward declare

void build_udf_eb_only(
    std::function<amrex::Real(const amrex::RealArray&)> udf_if,
    const amrex::Geometry& geom,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm,
    int nghost,
    int ls_refinement_in,
    amrex::MultiFab*& lsphi_out,
    amrex::EBFArrayBoxFactory*& ebfactory_out);
```

### 8. `Source/mpm_eb.cpp` — UDF geometry block
The `udf_cpp` geometry type block was refactored to:
- Call `build_udf_eb_only` for EB index space + ebfactory + lsphi allocation
- Then overwrite lsphi via `LoopOnCpu` with exact UDF values (instead of `FillSignedDistance` which is inaccurate for small boxes)

```cpp
else if (geom_type == "udf_cpp")
{
    using_levelset_geometry = true;

    std::string so_path;
    if (!pp.query("udf_so_file", so_path))
        amrex::Abort("[UDF] eb2.udf_so_file must be set when eb2.geom_type = udf_cpp");

    g_udf_loaders[0].load(so_path);
    auto phi_fn = g_udf_loaders[0].get_fn();

    auto udf_if_fn = [phi_fn](const amrex::RealArray &p) -> amrex::Real
    { return static_cast<amrex::Real>(phi_fn(p[0], p[1], p[2])); };

    build_udf_eb_only(udf_if_fn, geom, ba, dm, nghost, ls_refinement, lsphi, ebfactory);

    const auto plo = geom.ProbLoArray();
    const auto dx  = geom.CellSizeArray();
    const int  lsr = ls_refinement;

    for (MFIter mfi(*lsphi); mfi.isValid(); ++mfi)
    {
        const Box&   bx  = mfi.fabbox();
        Array4<Real> phi = lsphi->array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k)
        {
            amrex::Real x = plo[0] + i * dx[0] / lsr;
            amrex::Real y = plo[1] + j * dx[1] / lsr;
            amrex::Real z = (AMREX_SPACEDIM == 3) ? plo[2] + k * dx[2] / lsr : 0.0;
            phi(i, j, k) = static_cast<amrex::Real>(phi_fn(x, y, z));
        });
    }
    amrex::Print() << "[UDF] lsphi filled on CPU (GPU-safe). "
                   << "min=" << lsphi->min(0) << " max=" << lsphi->max(0) << "\n";
}
```

### 9. `Source/Make.package` — GNUmake (Linux only)
```makefile
CEXE_sources += mpm_eb_udf_build.cpp
```

---

## Key Lessons Learned

| Problem | Root Cause | Fix |
|---|---|---|
| `unistd.h` not found | POSIX header, doesn't exist on Windows | Use `winsock2.h` on Windows |
| `gethostname` not found | Lives in Winsock2 on Windows | `#include <winsock2.h>` + link `Ws2_32.lib` |
| `min`/`max` macro conflicts | `windows.h` defines them as macros | `NOMINMAX` define + `#undef min/max` after includes |
| `sockaddr` redefinition | `winsock.h` included before `winsock2.h` | `WIN32_LEAN_AND_MEAN` define |
| C4244 warning (`__int64` to `long`) | `long` is 32-bit on MSVC, 64-bit on Linux | `/wd4244` suppress (AMReX internal issue) |
| C4005 warning (`INT8_MIN` redefinition) | AMReX lexer header conflicts with `stdint.h` | `/wd4005` suppress |
| CCCL preprocessor error | CUDA 13.2 CCCL requires conforming preprocessor | `CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING` or `/Zc:preprocessor` |
| `__device__` called from `__host__` | Function chain not marked `HOST_DEVICE` | Add `AMREX_GPU_HOST_DEVICE` to call chain |
| `__constant__` memory in host code | GPU constant memory not readable from CPU | `__CUDA_ARCH__` guard with CPU fallback |
| nvcc stub.c error with EB2::Build | UDF lambda with dlopen pointer trips stub generator | Move `EB2::Build` to CXX-only file using `std::function` type erasure |
| LNK2005 multiply defined symbol | Function defined in header included by multiple TUs | Add `inline` keyword |

---

## Current Status
- ✅ CPU debug build works
- ✅ CUDA release build works  
- GNUmake is Linux/WSL only — no changes needed for Windows

## Remaining Items (if any)
- Runtime testing of UDF geometry on Windows+CUDA
- Verify `lsphi` values are correct when using UDF geometry
