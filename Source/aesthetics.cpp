#include <AMReX.H> // for amrex::Print and amrex::Real
#include <aesthetics.H>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <AMReX_ParallelDescriptor.H>
#include <ctime>
#include <unistd.h>

using namespace amrex;

#ifndef EXAGOOP_GIT_HASH
#define EXAGOOP_GIT_HASH "unknown"
#endif

#ifndef AMREX_GIT_HASH
#define AMREX_GIT_HASH "unknown"
#endif

/**
 * @brief Prints a detailed startup banner for an ExaGOOP run.
 *
 * This function:
 *  - Executes only on the I/O processor
 *  - Gathers system metadata (hostname, compiler, GPU backend, MPI/OpenMP
 * status)
 *  - Prints git hashes for ExaGOOP and AMReX
 *  - Prints build configuration and timestamp
 *
 * The banner helps ensure reproducibility and provides useful diagnostics
 * for debugging and provenance tracking.
 */

void PrintWelcomeMessage()
{
    if (!amrex::ParallelDescriptor::IOProcessor())
        return;

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    std::time_t now = std::time(nullptr);
    std::tm *local = std::localtime(&now);
    char timebuf[64];
    std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S %Z", local);

    std::string compiler;
#if defined(__clang__)
    compiler = "Clang " + std::to_string(__clang_major__) + "." +
               std::to_string(__clang_minor__) + "." +
               std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
    compiler = "GCC " + std::to_string(__GNUC__) + "." +
               std::to_string(__GNUC_MINOR__) + "." +
               std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    compiler = "MSVC " + std::to_string(_MSC_VER);
#else
    compiler = "Unknown compiler";
#endif

    std::string gpu_backend = "None";
#if defined(AMREX_USE_CUDA)
    gpu_backend = "CUDA";
#elif defined(AMREX_USE_HIP)
    gpu_backend = "HIP";
#elif defined(AMREX_USE_SYCL)
    gpu_backend = "SYCL";
#endif

    std::string build_type =
#if defined(NDEBUG)
        "Release";
#else
        "Debug";
#endif

    std::string mpi_status =
#if defined(AMREX_USE_MPI)
        "ON";
#else
        "OFF";
#endif

    std::string omp_status =
#if defined(_OPENMP)
        "ON";
#else
        "OFF";
#endif

    std::string precision =
#if defined(AMREX_USE_FLOAT)
        "float";
#elif defined(AMREX_USE_DOUBLE)
        "double";
#else
                    "unknown";
#endif

    // -----------------------------
    // Print Banner
    // -----------------------------
    amrex::Print() << "\n ======================================================"
                      "=========\n";
    amrex::Print() << "                  Welcome to EXAGOOP MPM Solver\n";
    amrex::Print()
        << "           Developed by SAMSers: Hari, Sree and Marc at NREL\n";
    amrex::Print() << " --------------------------------------------------------"
                      "---------\n";
    amrex::Print() << " ExaGOOP Git commit: " << EXAGOOP_GIT_HASH << "\n";
    amrex::Print() << " AMReX   Git commit: " << AMREX_GIT_HASH << "\n";
    amrex::Print() << " Build: " << build_type << " | GPU: " << gpu_backend
                   << "  | MPI: " << mpi_status << " | OpenMP: " << omp_status
                   << "  | Precision: " << precision << "\n";
    amrex::Print() << " Compiler: " << compiler << "\n";
    amrex::Print() << " Hostname: " << hostname << "\n";
    amrex::Print() << " Run started: " << timebuf << "\n";
    amrex::Print() << " ========================================================"
                      "=======\n\n";
}

/**
 * @brief Prints a formatted progress message with a default '-' fill character.
 *
 * @param msg        Base message to print
 * @param print_len  Desired total printed width before the arrow
 * @param begin      If true, prints the starting message; otherwise prints "
 * Done"
 *
 * This helper is used to create consistent progress-line formatting
 * throughout ExaGOOP’s console output.
 */

void PrintMessage(std::string msg, int print_len, bool begin)
{
    if (begin == true)
    {
        msg.append(print_len - msg.length(), '-');
        msg.append(1, '>');
        amrex::Print() << msg;
    }
    else
    {
        msg = " Done";
        amrex::Print() << msg;
    }
}

void PrintMultiLineMessage(std::string msg, int print_len, bool begin)
{
    if (begin == true)
    {
        msg.append(print_len - msg.length(), '.');
        //msg.append(1, '>');
        amrex::Print() << msg;
    }
    else
    {
        msg = " Done";
        amrex::Print() << msg;
    }
}

/**
 * @brief Prints a formatted progress message using a custom fill character.
 *
 * @param msg        Base message to print
 * @param print_len  Desired total printed width before the arrow
 * @param begin      If true, prints the starting message; otherwise prints "
 * Done"
 * @param c          Character used to fill the spacing region
 *
 * This overload allows decorative separators (e.g., '*') for section headers.
 */

void PrintMessage(std::string msg, int print_len, bool begin, char c)
{
    if (begin == true)
    {
        msg.append(print_len - msg.length(), c);
        amrex::Print() << msg;
    }
    else
    {
        msg = " Done";
        amrex::Print() << msg;
    }
}

/**
 * @brief Prints key simulation parameters for the current MPM setup.
 *
 * This function prints:
 *  - Total number of material points
 *  - Total mass of material points
 *  - Total volume of material points
 *  - Rigid-body particle counts and masses for each rigid body
 *
 * It queries the MPMParticleContainer and MPMspecs structures to extract
 * physically meaningful summary information for the user.
 */

void PrintSimParams(MPMParticleContainer *mpm_pc, MPMspecs *specs)
{

    std::string msg = "";
    int tmpi;
    amrex::Real tmpr;

    msg = "\n     ";
    PrintMessage(msg, print_length, true, '*'); //* line

    msg = "\n     Total number of material points:";
    PrintMessage(msg, print_length, true);

    mpm_pc->Calculate_Total_Number_of_MaterialParticles(tmpi);
    amrex::Print() << " " << tmpi;

    msg = "\n     Total mass of material points:";
    PrintMessage(msg, print_length, true);

    mpm_pc->Calculate_Total_Mass_MaterialPoints(tmpr);
    amrex::Print() << " " << std::setprecision(4) << tmpr;

    msg = "\n     Total volume of material points:";
    PrintMessage(msg, print_length, true);

    mpm_pc->Calculate_Total_Vol_MaterialPoints(tmpr);
    amrex::Print() << " " << std::setprecision(4) << tmpr;

    msg = "\n     Rigid particle details:";
    PrintMessage(msg, print_length, true);

    for (int i = 0; i < specs->no_of_rigidbodies_present; i++)
    {
        msg = "\n        Total number of rigid body particles in body " +
              std::to_string(i) + ":";
        PrintMessage(msg, print_length, true);
        amrex::Print() << " " << specs->Rb[i].num_of_mp << "\n";

        msg = "\n        Total mass of rigid body " + std::to_string(i) + ":";
        PrintMessage(msg, print_length, true);
        amrex::Print() << " " << specs->Rb[i].total_mass << "\n";
    }

    msg = "\n     ";
    PrintMessage(msg, print_length, true, '*'); //* line
}



std::string FormatParticleCount(long long npart)
{
    std::ostringstream oss;

    // Format the number depending on magnitude
    if (npart < 1'000'000) {
        oss << npart;  // normal integer
    } else {
        oss << std::scientific << std::uppercase
            << std::setprecision(3) << static_cast<double>(npart);
    }

    std::string num_str = oss.str();

    // Desired total width of the message (adjust as needed)
    const int total_width = 15;

    std::ostringstream final_msg;
    final_msg << "\n    Read "
              << std::setw(total_width - 10)  // ensures same total length
              << std::right << num_str
              << " material points";

    return final_msg.str();
}

std::string FormatElapsedTime(double seconds)
{
    long long total_ms = static_cast<long long>(seconds * 1000.0);

    long long hours = total_ms / (1000LL * 60 * 60);
    total_ms %= (1000LL * 60 * 60);

    long long minutes = total_ms / (1000LL * 60);
    total_ms %= (1000LL * 60);

    long long secs = total_ms / 1000LL;
    long long ms   = total_ms % 1000LL;

    std::ostringstream oss;
    oss << "\n    Took "
        << std::setw(2) << std::setfill('0') << hours   << "H "
        << std::setw(2) << std::setfill('0') << minutes << "M "
        << std::setw(2) << std::setfill('0') << secs    << "S "
        << std::setw(3) << std::setfill('0') << ms      << "ms "
        << "to read";

    return oss.str();
}




