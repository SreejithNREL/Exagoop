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
option(EXAGOOP_ENABLE_MPI "Enable MPI" OFF)
option(EXAGOOP_ENABLE_OPENMP "Enable OpenMP" OFF)
option(EXAGOOP_ENABLE_CUDA "Enable CUDA" OFF)
option(EXAGOOP_ENABLE_HIP "Enable HIP" OFF)
option(EXAGOOP_ENABLE_SYCL "Enable SyCL" OFF)

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


#Enable Temperature Equation
option(EXAGOOP_USE_TEMP "Enable temperature equation" ON)

if (EXAGOOP_USE_TEMP)
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
endif ()

# Setting MPI Flags
if(EXAGOOP_ENABLE_MPI)
  message(STATUS "MPI Configure Section")
  find_package(MPI REQUIRED CXX)
endif()

# -----------------------------
# Match AMReX GNUmake naming
# -----------------------------

# Base name
set(EBASE "ExaGOOP")

# Dimension (AMReX_SPACEDIM is already defined)
set(DIM "${EXAGOOP_DIM}d")

# Compiler name (match AMReX COMP)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(COMP "gnu")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(COMP "clang")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(COMP "intel")
else()
    set(COMP "unknown")
endif()

# Parallel model
if (EXAGOOP_ENABLE_MPI)
    set(PARALLEL ".MPI")
endif()

if (EXAGOOP_ENABLE_OMP)
    set(PARALLEL "${PARALLEL}.OMP")
endif()

if (EXAGOOP_ENABLE_CUDA)
    set(PARALLEL "${PARALLEL}.CUDA")
endif()

# Debug/opt
if (CMAKE_BUILD_TYPE MATCHES ".Debug")
    set(BTYPE "DEBUG")
endif()

# Final executable name
set(EXAGOOP_EXE_NAME "${EBASE}${DIM}.${COMP}${PARALLEL}${BTYPE}") 
# Add .ex only on non-Windows platforms 
if(NOT WIN32) 
    set(EXAGOOP_EXE_NAME "${EXAGOOP_EXE_NAME}.ex") 
endif()
message(STATUS "Executable name: ${EXAGOOP_EXE_NAME}")



# Setting AMReX Configuration
message(STATUS "AMReX Configure Section")
option(EXAGOOP_USE_HDF5 "Enable HDF5" OFF) 
option(EXAGOOP_USE_HDF5_PARALLEL "Enable parallel HDF5" OFF)
set(AMReX_HDF5 ${EXAGOOP_USE_HDF5}) 
set(AMReX_HDF5_PARALLEL ${EXAGOOP_USE_HDF5_PARALLEL})

set(AMREX_SUBMOD_LOCATION "${CMAKE_SOURCE_DIR}/Submodules/amrex")
list(APPEND CMAKE_MODULE_PATH "${AMREX_SUBMOD_LOCATION}/Tools/CMake")
include(SetAmrexOptions)
add_subdirectory(${AMREX_SUBMOD_LOCATION})
include(SetAmrexCompileFlags)

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

# ExaGOOP git hash
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE EXAGOOP_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_compile_definitions(EXAGOOP_GIT_HASH="${EXAGOOP_GIT_HASH}")

# AMReX git hash
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${AMREX_SUBMOD_LOCATION}
    OUTPUT_VARIABLE AMREX_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_compile_definitions(AMREX_GIT_HASH="${AMREX_GIT_HASH}")


# General information about machine, compiler, and build type
message(STATUS "${PROJECT_NAME} Information:")
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "EXAGOOP_PRECISION = ${EXAGOOP_PRECISION}")

include(BuildExeAndLib)
