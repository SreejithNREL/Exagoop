cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(ExaGOOP LANGUAGES CXX)

#Setting CMAKE basics
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


#Setting ExaGOOP permanent settings. Do NOT change them
set(EXAGOOP_DIM "3" CACHE STRING "Number of spatial dimensions")
set(EXAGOOP_PRECISION DOUBLE)
set(EXAGOOP_ENABLE_EB ON)
set(EXAGOOP_ENABLE_PARTICLES ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

if(WIN32)
    #add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN VC_EXTRA_LEAN WIN32_EXTRA_LEAN)    
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)
endif()

# ===============================================================
# EB enforcement (AMReX limitation)
# ===============================================================
if(EXAGOOP_DIM EQUAL 1 AND EXAGOOP_ENABLE_EB)
    message(WARNING "EB requested but not available in 1D. Forcing USE_EB=0.")
    add_compile_definitions(USE_EB=0)
elseif(EXAGOOP_ENABLE_EB)
    add_compile_definitions(USE_EB=1)
else()
    add_compile_definitions(USE_EB=0)
endif()


# ===============================================================
# Precision (match GNUmake: SINGLE or DOUBLE)
# ===============================================================
set(EXAGOOP_PRECISION "DOUBLE" CACHE STRING "Precision: SINGLE or DOUBLE")

if(EXAGOOP_PRECISION STREQUAL "SINGLE")
    add_compile_definitions(AMREX_USE_FLOAT)
    add_compile_definitions(EXAGOOP_PRECISION_STR="single")
elseif(EXAGOOP_PRECISION STREQUAL "DOUBLE")
    add_compile_definitions(AMREX_USE_DOUBLE)
    add_compile_definitions(EXAGOOP_PRECISION_STR="double")
else()
    message(FATAL_ERROR "EXAGOOP_PRECISION must be SINGLE or DOUBLE")
endif()


# HPC options
option(EXAGOOP_ENABLE_MPI    "Enable MPI"    OFF)
option(EXAGOOP_ENABLE_OPENMP "Enable OpenMP" OFF)
option(EXAGOOP_ENABLE_CUDA   "Enable CUDA"   OFF)
option(EXAGOOP_ENABLE_HIP    "Enable HIP"    OFF)
option(EXAGOOP_ENABLE_SYCL   "Enable SyCL"   OFF)

# ===============================================================
# GPU backend detection (match GNUmake)
# ===============================================================
if(EXAGOOP_ENABLE_CUDA)
    add_compile_definitions(EXAGOOP_GPU_BACKEND="CUDA")
elseif(EXAGOOP_ENABLE_HIP)
    add_compile_definitions(EXAGOOP_GPU_BACKEND="HIP")
elseif(EXAGOOP_ENABLE_SYCL)
    add_compile_definitions(EXAGOOP_GPU_BACKEND="SYCL")
else()
    add_compile_definitions(EXAGOOP_GPU_BACKEND="None")
endif()


# Enable Temperature Equation
option(EXAGOOP_USE_TEMP "Enable temperature equation" ON)

if(EXAGOOP_USE_TEMP)
    add_compile_definitions(USE_TEMP=1)
else()
    add_compile_definitions(USE_TEMP=0)
endif()

# Enable EB only if requested AND dimension >= 2
if(EXAGOOP_ENABLE_EB AND EXAGOOP_DIM GREATER_EQUAL 2)
    add_compile_definitions(USE_EB=1)
else()
    add_compile_definitions(USE_EB=0)
endif()


# Setting CUDA
if(EXAGOOP_ENABLE_CUDA)
    enable_language(CUDA)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR "Your nvcc version is ${CMAKE_CUDA_COMPILER_VERSION} which is unsupported."
            "Please use CUDA toolkit version 10.0 or newer.")
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 90)
    endif()
endif()

# Setting HIP
if(EXAGOOP_ENABLE_HIP)
    enable_language(HIP)
endif()

# Setting MPI Flags
if(EXAGOOP_ENABLE_MPI)
    message(STATUS "MPI Configure Section")
    find_package(MPI REQUIRED CXX)
endif()

# -----------------------------
# Match AMReX GNUmake naming
# -----------------------------

set(EBASE "ExaGOOP")
set(DIM "${EXAGOOP_DIM}d")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(COMP "gnu")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(COMP "clang")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(COMP "intel")
else()
    set(COMP "unknown")
endif()

if(EXAGOOP_ENABLE_MPI)
    set(PARALLEL ".MPI")
endif()

if(EXAGOOP_ENABLE_OMP)
    set(PARALLEL "${PARALLEL}.OMP")
endif()

if(EXAGOOP_ENABLE_CUDA)
    set(PARALLEL "${PARALLEL}.CUDA")
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(BTYPE "DEBUG")
endif()

set(EXAGOOP_EXE_NAME "${EBASE}${DIM}.${COMP}${PARALLEL}${BTYPE}")
if(NOT WIN32)
    set(EXAGOOP_EXE_NAME "${EXAGOOP_EXE_NAME}.ex")
endif()
message(STATUS "Executable name: ${EXAGOOP_EXE_NAME}")


# ===============================================================
# AMReX configuration
# ===============================================================
message(STATUS "AMReX Configure Section")
option(EXAGOOP_USE_HDF5          "Enable HDF5"          OFF)
option(EXAGOOP_USE_HDF5_PARALLEL "Enable parallel HDF5" OFF)
set(AMReX_HDF5          ${EXAGOOP_USE_HDF5})
set(AMReX_HDF5_PARALLEL ${EXAGOOP_USE_HDF5_PARALLEL})

set(AMREX_SUBMOD_LOCATION "${CMAKE_SOURCE_DIR}/Submodules/amrex")
list(APPEND CMAKE_MODULE_PATH "${AMREX_SUBMOD_LOCATION}/Tools/CMake")
include(SetAmrexOptions)
add_subdirectory(${AMREX_SUBMOD_LOCATION})
include(SetAmrexCompileFlags)

# ── UDF: capture the AMReX cmake directory AFTER add_subdirectory ─────────────
set(EXAGOOP_AMREX_DIR "${CMAKE_BINARY_DIR}/Submodules/amrex/lib/cmake/AMReX")

# ── STL geometry: detect whether this AMReX has EB2::STLGeom ─────────────────
# STLGeom was added in AMReX ~23.05. We check for the header rather than
# relying on version macros, which are unreliable in older releases.
include(CheckIncludeFileCXX)
set(CMAKE_REQUIRED_INCLUDES "${AMREX_SUBMOD_LOCATION}/Src/EB")
check_include_file_cxx("AMReX_EB2_IF_STL.H" EXAGOOP_AMREX_HAS_STLGEOM_HEADER)
unset(CMAKE_REQUIRED_INCLUDES)

if(EXAGOOP_AMREX_HAS_STLGEOM_HEADER)
    add_compile_definitions(EXAGOOP_AMREX_HAS_STLGEOM=1)
    message(STATUS "AMReX EB2::STLGeom    : available")
else()
    add_compile_definitions(EXAGOOP_AMREX_HAS_STLGEOM=0)
    message(STATUS "AMReX EB2::STLGeom    : NOT available (older AMReX — STL will use fallback path)")
endif()

# ===============================================================
# Parallelism macros (match GNUmake)
# ===============================================================
add_compile_definitions(USE_MPI=$<BOOL:${EXAGOOP_ENABLE_MPI}>)
add_compile_definitions(USE_OMP=$<BOOL:${EXAGOOP_ENABLE_OPENMP}>)

# ===============================================================
# Build type string for banner
# ===============================================================
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(EXAGOOP_BUILD_TYPE="Debug")
else()
    add_compile_definitions(EXAGOOP_BUILD_TYPE="Release")
endif()


# ===============================================================
# Git hashes (match GNUmake)
# ===============================================================
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE EXAGOOP_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
add_compile_definitions(EXAGOOP_GIT_HASH="${EXAGOOP_GIT_HASH}")

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${AMREX_SUBMOD_LOCATION}
    OUTPUT_VARIABLE AMREX_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
add_compile_definitions(AMREX_GIT_HASH="${AMREX_GIT_HASH}")


# General information
message(STATUS "${PROJECT_NAME} Information:")
message(STATUS "CMAKE_SYSTEM_NAME         = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID     = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE          = ${CMAKE_BUILD_TYPE}")
message(STATUS "EXAGOOP_PRECISION         = ${EXAGOOP_PRECISION}")

include(BuildExeAndLib)

# ===============================================================
# UDF level-set system
# ===============================================================

# ── Block 1: link the dynamic loader ─────────────────────────────────────────
# dlopen/dlsym on Linux and macOS; LoadLibrary on Windows is in kernel32
# which CMake links automatically, so CMAKE_DL_LIBS is empty there.
if(UNIX)
    target_link_libraries(${EXAGOOP_EXE_NAME} PRIVATE ${CMAKE_DL_LIBS})
endif()

# ── Block 2: generate and install the helper scripts ─────────────────────────
# The .in files contain @VAR@ tokens that configure_file substitutes.
# This must happen after add_subdirectory(amrex) so EXAGOOP_AMREX_DIR is set,
# and after include(BuildExeAndLib) so the install prefix is finalised.

# Where ExaGOOPUDF.cmake lands after `cmake --install`
set(EXAGOOP_CMAKE_DIR "${CMAKE_INSTALL_PREFIX}/cmake")

# Bake EXAGOOP_AMREX_DIR, EXAGOOP_CMAKE_DIR, and CMAKE_INSTALL_PREFIX
# into the helper scripts at configure time
configure_file(
    "${CMAKE_SOURCE_DIR}/tools/exagoop-build-udf.in"
    "${CMAKE_BINARY_DIR}/exagoop-build-udf"
    @ONLY)

file(CHMOD "${CMAKE_BINARY_DIR}/exagoop-build-udf"
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                 GROUP_READ GROUP_EXECUTE
                 WORLD_READ WORLD_EXECUTE)

configure_file(
    "${CMAKE_SOURCE_DIR}/tools/exagoop-build-udf.bat.in"
    "${CMAKE_BINARY_DIR}/exagoop-build-udf.bat"
    @ONLY)

# Install the satellite CMake file (used by exagoop-build-udf internally)
install(FILES "${CMAKE_SOURCE_DIR}/CMake/udf_builder/CMakeLists.txt"
    DESTINATION CMake/udf_builder)

# Install the helper scripts
install(PROGRAMS "${CMAKE_BINARY_DIR}/exagoop-build-udf"
    DESTINATION bin)

install(FILES "${CMAKE_BINARY_DIR}/exagoop-build-udf.bat"
    DESTINATION bin)

# Install the GNUmake alternative
install(FILES "${CMAKE_SOURCE_DIR}/tools/GNUmakefile.udf"
    DESTINATION share/exagoop)

# Install the user-facing UDF templates
install(FILES
    "${CMAKE_SOURCE_DIR}/Exec/udf_templates/levelset_simple_template.cpp"
    "${CMAKE_SOURCE_DIR}/Exec/udf_templates/levelset_advanced_template.cpp"
    DESTINATION share/exagoop/udf_templates)

message(STATUS "UDF level-set system enabled")
message(STATUS "  AMReX dir for UDF builds : ${EXAGOOP_AMREX_DIR}")
message(STATUS "  Helper scripts generated : ${CMAKE_BINARY_DIR}/exagoop-build-udf(.bat)")
