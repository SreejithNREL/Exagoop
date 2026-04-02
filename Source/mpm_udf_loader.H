#ifndef MPM_UDF_LOADER_H_
#define MPM_UDF_LOADER_H_
// clang-format off
/**
 * @file mpm_udf_loader.H
 * @brief Cross-platform runtime loader for user-defined level-set shared libraries.
 *
 * Wraps dlopen/dlsym (Linux, macOS) and LoadLibrary/GetProcAddress (Windows)
 * behind a single UDFLoader class. ExaGOOP uses this to load a user-compiled
 * .so/.dylib/.dll at startup without recompiling ExaGOOP itself.
 *
 * The loaded library must export exactly one C symbol:
 *
 *     extern "C" double levelset_phi(double x, double y, double z);
 *
 * where phi < 0 inside the solid, phi > 0 outside (material region).
 *
 * Usage:
 *     UDFLoader loader;
 *     loader.load("/path/to/levelset_udf.so");
 *     double phi = loader(x, y, z);
 */

#include <AMReX_Print.H>
#include <AMReX_GpuQualifiers.H>
#include <string>

// ── Platform detection ────────────────────────────────────────────────────────
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
   using DylibHandle = HMODULE;
#  define DYLIB_OPEN(path)    LoadLibraryA(path)
#  define DYLIB_SYM(h, sym)   reinterpret_cast<void*>(GetProcAddress(h, sym))
#  define DYLIB_CLOSE(h)      FreeLibrary(h)
#  define DYLIB_EXT           ".dll"
   inline std::string dylib_error()
   {
       char buf[512];
       FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                      nullptr, GetLastError(), 0, buf, sizeof(buf), nullptr);
       return std::string(buf);
   }
#else
#  include <dlfcn.h>
   using DylibHandle = void*;
#  define DYLIB_OPEN(path)    dlopen(path, RTLD_NOW | RTLD_LOCAL)
#  define DYLIB_SYM(h, sym)   dlsym(h, sym)
#  define DYLIB_CLOSE(h)      dlclose(h)
#  ifdef __APPLE__
#    define DYLIB_EXT         ".dylib"
#  else
#    define DYLIB_EXT         ".so"
#  endif
   inline std::string dylib_error() { return std::string(dlerror()); }
#endif

// ── Function pointer type ─────────────────────────────────────────────────────
using LevelSetPhiFn = double (*)(double, double, double);

// ── UDFLoader class ───────────────────────────────────────────────────────────
class UDFLoader
{
public:
    UDFLoader()  = default;

    /**
     * @brief Loads a compiled UDF shared library and resolves levelset_phi.
     *
     * Call this once at startup (inside init_eb) before EB2::Build.
     * The library must remain loaded for the duration of the simulation,
     * so keep the UDFLoader instance alive at file scope.
     *
     * @param so_path  Full path to the .so / .dylib / .dll
     */
    void load(const std::string& so_path)
    {
        handle_ = DYLIB_OPEN(so_path.c_str());
        if (!handle_)
        {
            amrex::Abort("[UDF] Failed to load '" + so_path + "': " + dylib_error());
        }

        void* sym = DYLIB_SYM(handle_, "levelset_phi");
        if (!sym)
        {
            amrex::Abort("[UDF] Symbol 'levelset_phi' not found in '" +
                         so_path + "': " + dylib_error());
        }

        phi_fn_ = reinterpret_cast<LevelSetPhiFn>(sym);
        amrex::Print() << "[UDF] Loaded successfully: " << so_path << "\n";
    }
	
	// Add inside the UDFLoader class, after load():
	LevelSetPhiFn get_fn() const { return phi_fn_; }

    /**
     * @brief Evaluates phi at a physical point.
     *
     * Called from UDFImplicitFunction::operator() which is invoked during
     * EB2::Build and FillSignedDistance. Must NOT be called before load().
     */
    double operator()(double x, double y, double z) const
    {
        return phi_fn_(x, y, z);
    }

    bool is_loaded() const { return handle_ != nullptr; }

    // Non-copyable: owns an OS handle
    UDFLoader(const UDFLoader&)            = delete;
    UDFLoader& operator=(const UDFLoader&) = delete;

    ~UDFLoader()
    {
        if (handle_)
        {
            DYLIB_CLOSE(handle_);
            handle_ = nullptr;
        }
    }

private:
    DylibHandle   handle_ = nullptr;
    LevelSetPhiFn phi_fn_ = nullptr;
};

// ── UDFImplicitFunction ───────────────────────────────────────────────────────
/**
 * @brief Adapts UDFLoader to the AMReX ImplicitFunction concept.
 *
 * AMReX EB2::makeShop() requires an object with:
 *     amrex::Real operator()(const amrex::RealArray&) const
 *
 * This thin wrapper forwards calls to the loaded UDFLoader.
 * Holds a const reference — UDFLoader must outlive this object.
 */
class UDFImplicitFunction
{
public:
    explicit UDFImplicitFunction(const UDFLoader& loader) : loader_(loader) {}

    amrex::Real operator()(const amrex::RealArray& p) const
    {
        return static_cast<amrex::Real>(loader_(p[0], p[1], p[2]));
    }

    // Some AMReX EB2 paths also use a 3-arg overload
    amrex::Real operator()(amrex::Real x, amrex::Real y, amrex::Real z) const
    {
        return static_cast<amrex::Real>(loader_(x, y, z));
    }

private:
    const UDFLoader& loader_;
};
// clang-format on
#endif // MPM_UDF_LOADER_H_
