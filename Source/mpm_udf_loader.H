#ifndef MPM_UDF_LOADER_H
#define MPM_UDF_LOADER_H

// clang-format off
#include <AMReX.H>
#include <AMReX_Array.H>
#include <string>
#include <stdexcept>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif
// clang-format on

/**
 * @brief Cross-platform shared-library handle owner.
 *
 * Loads a shared library (.so / .dylib / .dll) at construction and
 * unloads it at destruction.  Non-copyable; move-only.
 *
 * Usage:
 *   UDFLoader loader("/path/to/liblevelset_udf.so");
 *   auto fn = loader.symbol<double(*)(double,double,double)>("levelset_phi");
 */
class UDFLoader
{
  public:
    explicit UDFLoader(const std::string &path)
    {
#ifdef _WIN32
        m_handle = LoadLibraryA(path.c_str());
        if (!m_handle)
            amrex::Abort("UDFLoader: cannot open library '" + path +
                         "' (LoadLibrary returned NULL)");
#else
        m_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!m_handle)
            amrex::Abort("UDFLoader: cannot open library '" + path +
                         "': " + std::string(dlerror()));
#endif
    }

    ~UDFLoader()
    {
        if (m_handle)
        {
#ifdef _WIN32
            FreeLibrary(static_cast<HMODULE>(m_handle));
#else
            dlclose(m_handle);
#endif
        }
    }

    // Non-copyable
    UDFLoader(const UDFLoader &) = delete;
    UDFLoader &operator=(const UDFLoader &) = delete;

    // Move-constructible
    UDFLoader(UDFLoader &&other) noexcept : m_handle(other.m_handle)
    {
        other.m_handle = nullptr;
    }

    /**
     * @brief Look up a symbol by name and cast to FuncPtr.
     *
     * Example:
     *   auto f =
     * loader.symbol<double(*)(double,double,double)>("levelset_phi");
     */
    template <typename FuncPtr> FuncPtr symbol(const std::string &name) const
    {
#ifdef _WIN32
        FARPROC raw =
            GetProcAddress(static_cast<HMODULE>(m_handle), name.c_str());
        if (!raw)
            amrex::Abort("UDFLoader: symbol '" + name + "' not found");
#else
        dlerror(); // clear
        void *raw = dlsym(m_handle, name.c_str());
        const char *err = dlerror();
        if (err)
            amrex::Abort("UDFLoader: symbol '" + name +
                         "' not found: " + std::string(err));
#endif
        // Cast through void* to suppress -Wpedantic on function pointer casts
        FuncPtr fp;
        std::memcpy(&fp, &raw, sizeof(fp));
        return fp;
    }

  private:
    void *m_handle = nullptr;
};

/**
 * @brief Adapts a C-linkage levelset UDF to AMReX's ImplicitFunction concept.
 *
 * The UDF must export:
 *   extern "C" double levelset_phi(double x, double y, double z);
 *
 * Convention (same as AMReX EB2):
 *   phi < 0  →  inside obstacle (covered)
 *   phi > 0  →  outside obstacle (regular)
 *   phi = 0  →  on the surface
 *
 * IMPORTANT: This class holds ONLY the raw function pointer — it does NOT
 * own the shared-library handle.  The caller must keep a UDFLoader alive
 * for the entire lifetime of this object (and of any EB2 index space built
 * from it).
 *
 * This design is intentional: AMReX's EB2::makeShop / EB2::Build internally
 * copy the implicit-function object, so it must be copyable.  UDFLoader is
 * non-copyable (it owns a dlopen handle), so it cannot be a member here.
 *
 * Typical usage:
 *   UDFLoader           loader(so_path);               // owns dlopen handle
 *   UDFImplicitFunction udf_if(loader);                // borrows fn pointer
 *   auto shop = amrex::EB2::makeShop(udf_if);          // copies udf_if — ok
 *   amrex::EB2::Build(shop, ...);
 *   // loader must remain alive until EB2 index space is no longer needed
 *
 * GPU safety: function pointers are CPU-only.
 * Use amrex::LoopOnCpu (never amrex::ParallelFor) when calling this.
 */
class UDFImplicitFunction
{
  public:
    using PhiFn = double (*)(double, double, double);

    /// Borrow the function pointer from an already-open UDFLoader.
    /// The loader must outlive this object.
    explicit UDFImplicitFunction(const UDFLoader &loader)
    {
        m_phi_fn = loader.symbol<PhiFn>("levelset_phi");
    }

    /// Copyable — AMReX EB2 requires this.
    UDFImplicitFunction(const UDFImplicitFunction &) = default;
    UDFImplicitFunction &operator=(const UDFImplicitFunction &) = default;

    AMREX_FORCE_INLINE amrex::Real operator()(const amrex::RealArray &p) const
    {
#if (AMREX_SPACEDIM == 3)
        return static_cast<amrex::Real>(m_phi_fn(p[0], p[1], p[2]));
#elif (AMREX_SPACEDIM == 2)
        return static_cast<amrex::Real>(m_phi_fn(p[0], p[1], 0.0));
#else
        return static_cast<amrex::Real>(m_phi_fn(p[0], 0.0, 0.0));
#endif
    }

  private:
    PhiFn m_phi_fn = nullptr;
};

#endif // MPM_UDF_LOADER_H
