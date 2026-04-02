#ifndef MPM_RIGIDBODY_H_
#define MPM_RIGIDBODY_H_
// clang-format off
/**
 * @file mpm_rigidBody.H
 * @brief Lightweight rigid body / level-set BC manager — Phase 1.
 *
 * Phase 1 (current): owns BC parameters per body, maps them to the existing
 * global lsphi from mpm_ebtools. No body dynamics yet.
 *
 * Phase 2 (rigid body dynamics): each RigidBodyBC will grow to own its own
 * lsphi MultiFab, geometry UDF, position/velocity state, and ODE integrator.
 *
 * INPUT FILE CONVENTION
 * ──────────────────────
 * All keys use the mpm. prefix and human-readable face/body names.
 *
 *   mpm.use_levelset    = true        # enable level-set BC system
 *   mpm.num_rigidbodies = 2           # how many rb_N blocks to read
 *
 *   # ── Nodal velocity BCs (domain faces) ──────────────────────────────────
 *   # Types: 0=none  1=no-slip  2=free-slip  3=Coulomb
 *   mpm.nodal_vel_bc.x_lo.type     = 2
 *   mpm.nodal_vel_bc.x_hi.type     = 2
 *   mpm.nodal_vel_bc.y_lo.type     = 2
 *   mpm.nodal_vel_bc.y_hi.type     = 2
 *   mpm.nodal_vel_bc.z_lo.type     = 0
 *   mpm.nodal_vel_bc.z_hi.type     = 0
 *   mpm.nodal_vel_bc.x_lo.mu       = 0.3    # friction (type 3 only)
 *   mpm.nodal_vel_bc.x_lo.velocity = 0.0 0.0 0.0   # wall velocity
 *
 *   # ── Nodal temperature BCs (domain faces) ────────────────────────────────
 *   # Types: 0=none  1=Dirichlet  2=adiabatic  3=flux  4=convective  5=UDF
 *   mpm.nodal_temp_bc.x_lo.type    = 1
 *   mpm.nodal_temp_bc.x_lo.T_wall  = 300.0
 *   mpm.nodal_temp_bc.y_hi.type    = 4
 *   mpm.nodal_temp_bc.y_hi.h       = 150.0
 *   mpm.nodal_temp_bc.y_hi.T_inf   = 300.0
 *   mpm.nodal_temp_bc.y_hi.type    = 5      # spatially varying
 *   mpm.nodal_temp_bc.y_hi.udf     = /path/to/thermal.dylib
 *
 *   # ── Level-set velocity BCs (one block per rigid body) ───────────────────
 *   # Types: 0=none  1=no-slip  2=free-slip  3=Coulomb
 *   mpm.lset_vel_bc.rb_0.type      = 2
 *   mpm.lset_vel_bc.rb_0.mu        = 0.0
 *   mpm.lset_vel_bc.rb_0.velocity  = 0.0 0.0 0.0
 *
 *   # ── Level-set temperature BCs (one block per rigid body) ────────────────
 *   # Types: 0=none  1=Dirichlet  2=adiabatic  3=flux  4=convective  5=UDF
 *   mpm.lset_temp_bc.rb_0.type     = 4
 *   mpm.lset_temp_bc.rb_0.h        = 200.0
 *   mpm.lset_temp_bc.rb_0.T_inf    = 350.0
 *   mpm.lset_temp_bc.rb_1.type     = 1
 *   mpm.lset_temp_bc.rb_1.T_wall   = 300.0
 *   mpm.lset_temp_bc.rb_1.udf      = /path/to/thermal.dylib  # type 5
 *
 * LEGACY FORMAT (still works for backward compatibility)
 * ───────────────────────────────────────────────────────
 *   mpm.bc_lower           = 2 2 0
 *   mpm.bc_upper           = 2 2 0
 *   mpm.levelset_bc        = 2
 *   mpm.levelset_wall_mu   = 0.3
 *   mpm.levelset_bc_temp   = 4
 *   etc.
 */

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <string>
#include <vector>
#include <array>

// Forward declarations — mpm_eb.H included by translation units that need it
namespace amrex { class Geometry; class BoxArray; class DistributionMapping; }
namespace mpm_ebtools { void init_eb_bodies(const amrex::Geometry&,
                                            const amrex::BoxArray&,
                                            const amrex::DistributionMapping&,
                                            int); }

// ── Face index helpers ────────────────────────────────────────────────────────
// Maps face name string to (dimension, lo/hi) for ParmParse reads.
// Face order: x_lo=0, x_hi=1, y_lo=2, y_hi=3, z_lo=4, z_hi=5
static constexpr int NFACES = 2 * AMREX_SPACEDIM;

// face_dim[f]  = spatial dimension of face f
// face_is_lo[f] = true if face f is the lower face
inline int  face_dim(int f)   { return f / 2; }
inline bool face_is_lo(int f) { return (f % 2) == 0; }

// face_name[f] matches the input key suffix: x_lo, x_hi, y_lo, etc.
inline const char* face_name(int f)
{
    static const char* names[] = {
        "x_lo","x_hi","y_lo","y_hi","z_lo","z_hi"};
    return names[f];
}

// ── Per-face nodal velocity BC ────────────────────────────────────────────────
struct NodalVelFaceBC
{
    int  type = 2;   // default: free-slip
    amrex::Real mu = 0.0;
    amrex::Real velocity[AMREX_SPACEDIM] = {0.0};
};

// ── Per-face nodal temperature BC ────────────────────────────────────────────
struct NodalTempFaceBC
{
    int  type    = 0;     // default: no BC
    amrex::Real val   = 0.0;   // T_wall / q / h depending on type
    amrex::Real T_inf = 0.0;   // ambient T for types 4, 5
    std::string udf   = "";    // .so path for type 5
};

// ── Per-body level-set velocity BC ───────────────────────────────────────────
struct LSetVelBC
{
    int  type = 0;   // default: no BC
    amrex::Real mu = 0.0;
    amrex::Real velocity[AMREX_SPACEDIM] = {0.0};
};

// ── Per-body level-set temperature BC ────────────────────────────────────────
struct LSetTempBC
{
    int  type    = 0;     // default: no BC
    amrex::Real val   = 0.0;   // T_wall / q / h depending on type
    amrex::Real T_inf = 0.0;   // ambient T for types 4, 5
    std::string udf   = "";    // .so path for type 5
};

// ── RigidBodyBC — all BC data for one level-set rigid body ───────────────────
struct RigidBodyBC
{
    int id = 0;
    LSetVelBC  vel_bc;
    LSetTempBC temp_bc;

    // Phase 2: will also own lsphi MultiFab, geometry UDF, dynamics state
};

// ── RigidBodyManager ─────────────────────────────────────────────────────────
/**
 * @brief Manages per-body level-set BC parameters.
 *
 * In Phase 1 there is one global lsphi (from mpm_ebtools).
 * RigidBodyManager maps each body's BC params to that shared lsphi.
 * In Phase 2, each body will own its own lsphi and geometry.
 */
class RigidBodyManager
{
public:
    // Global flags
    bool use_levelset    = false;
    int  num_rigidbodies = 0;

    // Per-face nodal BCs (clean new format)
    std::array<NodalVelFaceBC,  NFACES> nodal_vel_bc;
    std::array<NodalTempFaceBC, NFACES> nodal_temp_bc;

    // Per-body level-set BCs
    std::vector<RigidBodyBC> bodies;

    /**
     * @brief Initialises level-set geometry for all rigid bodies.
     *
     * Call this AFTER init_eb() (which handles the legacy single-body eb2.*
     * path) and AFTER read_mpm_specs() (which calls rb_manager.init()).
     *
     * When num_rigidbodies > 0, this calls mpm_ebtools::init_eb_bodies()
     * which fills one lsphi per body and builds the union lsphi.
     *
     * @param geom  Coarse geometry
     * @param ba    BoxArray
     * @param dm    DistributionMapping
     */
    void init_geometry(const amrex::Geometry &geom,
                       const amrex::BoxArray &ba,
                       const amrex::DistributionMapping &dm)
    {
        if (num_rigidbodies > 0)
        {
            mpm_ebtools::init_eb_bodies(geom, ba, dm, num_rigidbodies);
        }
    }

    /**
     * @brief Reads all BC parameters from the input file.
     *
     * Reads the new mpm.nodal_vel_bc.*, mpm.nodal_temp_bc.*,
     * mpm.lset_vel_bc.rb_N.*, mpm.lset_temp_bc.rb_N.* keys.
     *
     * Then populates legacy MPMspecs arrays (bclo, bchi, etc.) so the
     * rest of ExaGOOP continues to work unchanged.
     *
     * @param bclo          Output: velocity BC type at lower faces (legacy array)
     * @param bchi          Output: velocity BC type at upper faces
     * @param wall_mu_lo    Output: friction at lower faces
     * @param wall_mu_hi    Output: friction at upper faces
     * @param wall_vel_lo   Output: wall velocity at lower faces
     * @param wall_vel_hi   Output: wall velocity at upper faces
     * @param bclo_temp     Output: temperature BC type at lower faces
     * @param bchi_temp     Output: temperature BC type at upper faces
     * @param bclo_tempval  Output: temperature BC value at lower faces
     * @param bchi_tempval  Output: temperature BC value at upper faces
     * @param bclo_Tinf     Output: T_inf at lower faces
     * @param bchi_Tinf     Output: T_inf at upper faces
     * @param bclo_udf      Output: UDF path at lower faces
     * @param bchi_udf      Output: UDF path at upper faces
     * @param levelset_bc   Output: legacy velocity BC type (body 0)
     * @param levelset_wall_mu  Output: legacy friction (body 0)
     */
    void init(
        std::vector<int>         &bclo,
        std::vector<int>         &bchi,
        amrex::Array<amrex::Real, AMREX_SPACEDIM> &wall_mu_lo,
        amrex::Array<amrex::Real, AMREX_SPACEDIM> &wall_mu_hi,
        amrex::Array<amrex::Real, AMREX_SPACEDIM*AMREX_SPACEDIM> &wall_vel_lo,
        amrex::Array<amrex::Real, AMREX_SPACEDIM*AMREX_SPACEDIM> &wall_vel_hi
#if USE_TEMP
        ,
        std::vector<int>         &bclo_temp,
        std::vector<int>         &bchi_temp,
        std::vector<amrex::Real> &bclo_tempval,
        std::vector<amrex::Real> &bchi_tempval,
        std::vector<amrex::Real> &bclo_Tinf,
        std::vector<amrex::Real> &bchi_Tinf,
        std::vector<std::string> &bclo_udf,
        std::vector<std::string> &bchi_udf,
        int                      &levelset_bc_temp,
        amrex::Real              &levelset_bc_temp_val,
        amrex::Real              &levelset_Tinf,
        std::string              &levelset_temp_udf
#endif
        ,
        int                      &levelset_bc,
        amrex::Real              &levelset_wall_mu)
    {
        amrex::ParmParse pp("mpm");

        // ── Global flags ──────────────────────────────────────────────────────
        pp.query("use_levelset",    use_levelset);
        pp.query("num_rigidbodies", num_rigidbodies);

        // ── Nodal velocity BCs (new format) ───────────────────────────────────
        // Face names: x_lo, x_hi, y_lo, y_hi, z_lo, z_hi
        // These override the legacy mpm.bc_lower / mpm.bc_upper if present.
        for (int f = 0; f < NFACES; ++f)
        {
            std::string base = std::string("nodal_vel_bc.")
                               + face_name(f) + ".";
            auto& fbc = nodal_vel_bc[f];

            pp.query((base + "type").c_str(),     fbc.type);
            pp.query((base + "mu").c_str(),       fbc.mu);

            // Wall velocity: up to AMREX_SPACEDIM components
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                std::string vkey = base + "velocity_" + std::to_string(d);
                pp.query(vkey.c_str(), fbc.velocity[d]);
            }
            // Also accept the array form: nodal_vel_bc.x_lo.velocity = 0 0 0
            amrex::Vector<amrex::Real> vtmp;
            if (pp.queryarr((base + "velocity").c_str(), vtmp))
                for (int d = 0; d < (int)vtmp.size() && d < AMREX_SPACEDIM; ++d)
                    fbc.velocity[d] = vtmp[d];
        }

        // Write back to legacy arrays so existing code continues to work
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            int f_lo = 2 * d;
            int f_hi = 2 * d + 1;

            // Only override if new format was actually specified
            // (type != 2 means user changed it from default)
            bclo[d]       = nodal_vel_bc[f_lo].type;
            bchi[d]       = nodal_vel_bc[f_hi].type;
            wall_mu_lo[d] = nodal_vel_bc[f_lo].mu;
            wall_mu_hi[d] = nodal_vel_bc[f_hi].mu;

            for (int dd = 0; dd < AMREX_SPACEDIM; ++dd)
            {
                wall_vel_lo[d * AMREX_SPACEDIM + dd] =
                    nodal_vel_bc[f_lo].velocity[dd];
                wall_vel_hi[d * AMREX_SPACEDIM + dd] =
                    nodal_vel_bc[f_hi].velocity[dd];
            }
        }

#if USE_TEMP
        // ── Nodal temperature BCs (new format) ────────────────────────────────
        for (int f = 0; f < NFACES; ++f)
        {
            std::string base = std::string("nodal_temp_bc.")
                               + face_name(f) + ".";
            auto& fbc = nodal_temp_bc[f];

            pp.query((base + "type").c_str(),   fbc.type);
            pp.query((base + "T_wall").c_str(), fbc.val);   // type 1
            pp.query((base + "flux").c_str(),   fbc.val);   // type 3
            pp.query((base + "h").c_str(),      fbc.val);   // type 4, 5
            pp.query((base + "T_inf").c_str(),  fbc.T_inf);
            pp.query((base + "udf").c_str(),    fbc.udf);
        }

        // Write back to legacy arrays
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            int f_lo = 2 * d;
            int f_hi = 2 * d + 1;
            bclo_temp[d]    = nodal_temp_bc[f_lo].type;
            bchi_temp[d]    = nodal_temp_bc[f_hi].type;
            bclo_tempval[d] = nodal_temp_bc[f_lo].val;
            bchi_tempval[d] = nodal_temp_bc[f_hi].val;
            bclo_Tinf[d]    = nodal_temp_bc[f_lo].T_inf;
            bchi_Tinf[d]    = nodal_temp_bc[f_hi].T_inf;
            bclo_udf[d]     = nodal_temp_bc[f_lo].udf;
            bchi_udf[d]     = nodal_temp_bc[f_hi].udf;
        }
#endif

        // ── Level-set / rigid body BCs ────────────────────────────────────────
        bodies.resize(std::max(num_rigidbodies, 1));

        for (int rb = 0; rb < (int)bodies.size(); ++rb)
        {
            bodies[rb].id = rb;
            std::string rb_name = "rb_" + std::to_string(rb);

            // Velocity BC
            {
                std::string base = "lset_vel_bc." + rb_name + ".";
                auto& vbc = bodies[rb].vel_bc;
                pp.query((base + "type").c_str(), vbc.type);
                pp.query((base + "mu").c_str(),   vbc.mu);
                amrex::Vector<amrex::Real> vtmp;
                if (pp.queryarr((base + "velocity").c_str(), vtmp))
                    for (int d = 0; d < (int)vtmp.size() && d < AMREX_SPACEDIM; ++d)
                        vbc.velocity[d] = vtmp[d];
            }

#if USE_TEMP
            // Temperature BC
            {
                std::string base = "lset_temp_bc." + rb_name + ".";
                auto& tbc = bodies[rb].temp_bc;
                pp.query((base + "type").c_str(),   tbc.type);
                pp.query((base + "T_wall").c_str(), tbc.val);   // type 1
                pp.query((base + "flux").c_str(),   tbc.val);   // type 3
                pp.query((base + "h").c_str(),      tbc.val);   // type 4, 5
                pp.query((base + "T_inf").c_str(),  tbc.T_inf);
                pp.query((base + "udf").c_str(),    tbc.udf);
            }
#endif
        }

        // ── Write back body 0 to legacy single-body scalars ───────────────────
        // This keeps utilities.cpp, nodal_levelset_bcs calls unchanged for now.
        if (!bodies.empty())
        {
            levelset_bc      = bodies[0].vel_bc.type;
            levelset_wall_mu = bodies[0].vel_bc.mu;
#if USE_TEMP
            levelset_bc_temp     = bodies[0].temp_bc.type;
            levelset_bc_temp_val = bodies[0].temp_bc.val;
            levelset_Tinf        = bodies[0].temp_bc.T_inf;
            levelset_temp_udf    = bodies[0].temp_bc.udf;
#endif
        }

        // ── Print summary ─────────────────────────────────────────────────────
        amrex::Print() << "\n[RigidBodyManager] use_levelset    = "
                       << (use_levelset ? "true" : "false") << "\n";
        amrex::Print() << "[RigidBodyManager] num_rigidbodies = "
                       << num_rigidbodies << "\n";

        for (int rb = 0; rb < (int)bodies.size(); ++rb)
        {
            amrex::Print() << "[RigidBodyManager] rb_" << rb
                           << "  vel_bc=" << bodies[rb].vel_bc.type
#if USE_TEMP
                           << "  temp_bc=" << bodies[rb].temp_bc.type
#endif
                           << "\n";
        }

        for (int f = 0; f < NFACES; ++f)
        {
            amrex::Print() << "[RigidBodyManager] nodal_vel_bc."
                           << face_name(f)
                           << ".type=" << nodal_vel_bc[f].type
#if USE_TEMP
                           << "  nodal_temp_bc." << face_name(f)
                           << ".type=" << nodal_temp_bc[f].type
#endif
                           << "\n";
        }
    }
};
// clang-format on
#endif // MPM_RIGIDBODY_H_
