#ifndef MPM_THERMAL_UDF_LOADER_H_
#define MPM_THERMAL_UDF_LOADER_H_
// clang-format off
/**
 * @file mpm_thermal_udf_loader.H
 * @brief Runtime loader for user-defined spatially varying thermal BC functions.
 *
 * Follows the same dlopen/LoadLibrary pattern as mpm_udf_loader.H.
 * The user compiles a .cpp into a shared library exporting two C functions:
 *
 *     // Heat transfer coefficient at point (x,y,z)
 *     extern "C" double thermal_h(double x, double y, double z);
 *
 *     // Ambient temperature at point (x,y,z)
 *     extern "C" double thermal_Tinf(double x, double y, double z);
 *
 * For a prescribed heat flux BC, only thermal_h is used (as the flux value q).
 * For a convective BC, both are used: Q = h(x)*(Tinf(x) - T_node).
 *
 * Build with:
 *     exagoop-build-udf my_thermal.cpp ./build     (CMake)
 *     make udf UDF_SRC=my_thermal.cpp              (GNUmake)
 *
 * Input file:
 *     mpm.bc_lower_temp      = 0  4     # face 0: none, face 1: convective UDF
 *     mpm.thermal_udf_lo_1   = /path/to/build/liblevelset_udf.dylib
 */

#include <AMReX_Print.H>
#include <string>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
   using DylibHandle_T = HMODULE;
#  define DYLIB_T_OPEN(path)    LoadLibraryA(path)
#  define DYLIB_T_SYM(h, sym)   reinterpret_cast<void*>(GetProcAddress(h, sym))
#  define DYLIB_T_CLOSE(h)      FreeLibrary(h)
   inline std::string dylib_t_error() {
       char buf[512];
       FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_IGNORE_INSERTS,
                      nullptr, GetLastError(), 0, buf, sizeof(buf), nullptr);
       return std::string(buf);
   }
#else
#  include <dlfcn.h>
   using DylibHandle_T = void*;
#  define DYLIB_T_OPEN(path)    dlopen(path, RTLD_NOW | RTLD_LOCAL)
#  define DYLIB_T_SYM(h, sym)   dlsym(h, sym)
#  define DYLIB_T_CLOSE(h)      dlclose(h)
   inline std::string dylib_t_error() { return std::string(dlerror()); }
#endif

// Function pointer types
using ThermalHFn    = double (*)(double, double, double);  // h(x,y,z) or q(x,y,z)
using ThermalTinfFn = double (*)(double, double, double);  // T_inf(x,y,z)

// ── ThermalUDFLoader ──────────────────────────────────────────────────────────
/**
 * @brief Loads a user thermal BC shared library and resolves thermal_h and
 *        thermal_Tinf symbols.
 *
 * thermal_h    is required (used for both flux and convective BCs).
 * thermal_Tinf is optional — only needed for convective BCs (type 4).
 *              If absent from the .so, Tinf queries return 0.0.
 */
class ThermalUDFLoader
{
public:
    ThermalUDFLoader() = default;

    void load(const std::string& so_path)
    {
        handle_ = DYLIB_T_OPEN(so_path.c_str());
        if (!handle_)
            amrex::Abort("[ThermalUDF] Failed to load '" + so_path +
                         "': " + dylib_t_error());

        // thermal_h is required
        void* sym_h = DYLIB_T_SYM(handle_, "thermal_h");
        if (!sym_h)
            amrex::Abort("[ThermalUDF] Symbol 'thermal_h' not found in: "
                         + so_path);
        h_fn_ = reinterpret_cast<ThermalHFn>(sym_h);

        // thermal_Tinf is optional
        void* sym_t = DYLIB_T_SYM(handle_, "thermal_Tinf");
        tinf_fn_ = sym_t ? reinterpret_cast<ThermalTinfFn>(sym_t) : nullptr;

        amrex::Print() << "[ThermalUDF] Loaded: " << so_path << "\n";
        if (!tinf_fn_)
            amrex::Print() << "[ThermalUDF] Note: thermal_Tinf not found "
                           << "— T_inf will be 0.0 everywhere.\n";
    }

    // h(x,y,z) — heat transfer coefficient or flux value
    double h(double x, double y, double z) const { return h_fn_(x, y, z); }

    // T_inf(x,y,z) — ambient temperature (returns 0 if not loaded)
    double Tinf(double x, double y, double z) const
    {
        return tinf_fn_ ? tinf_fn_(x, y, z) : 0.0;
    }

    ThermalHFn    get_h_fn()    const { return h_fn_;    }
    ThermalTinfFn get_tinf_fn() const { return tinf_fn_; }

    bool is_loaded()      const { return handle_ != nullptr; }
    bool has_tinf()       const { return tinf_fn_ != nullptr; }

    ThermalUDFLoader(const ThermalUDFLoader&)            = delete;
    ThermalUDFLoader& operator=(const ThermalUDFLoader&) = delete;

    ~ThermalUDFLoader()
    {
        if (handle_) { DYLIB_T_CLOSE(handle_); handle_ = nullptr; }
    }

private:
    DylibHandle_T  handle_  = nullptr;
    ThermalHFn     h_fn_    = nullptr;
    ThermalTinfFn  tinf_fn_ = nullptr;
};
// clang-format on
#endif // MPM_THERMAL_UDF_LOADER_H_
