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

# HPC options
option(EXAGOOP_ENABLE_MPI "Enable MPI" OFF)
option(EXAGOOP_ENABLE_OPENMP "Enable OpenMP" OFF)
option(EXAGOOP_ENABLE_CUDA "Enable CUDA" OFF)
option(EXAGOOP_ENABLE_HIP "Enable HIP" OFF)
option(EXAGOOP_ENABLE_SYCL "Enable SyCL" OFF)

#Enable Temperature Equation
option(EXAGOOP_USE_TEMP "Enable temperature equation" ON)

if (EXAGOOP_USE_TEMP)
    add_compile_definitions(USE_TEMP=1)
else()
    add_compile_definitions(USE_TEMP=0)
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
    set(PARALLEL "MPI")
else()
    set(PARALLEL "SERIAL")
endif()

if (EXAGOOP_ENABLE_OMP)
    set(PARALLEL "${PARALLEL}.OMP")
endif()

if (EXAGOOP_ENABLE_CUDA)
    set(PARALLEL "${PARALLEL}.CUDA")
endif()

# Debug/opt
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(BTYPE "DEBUG")
else()
    set(BTYPE "OPT")
endif()

# Final executable name
set(EXAGOOP_EXE_NAME "${EBASE}${DIM}.${COMP}.${PARALLEL}.${BTYPE}.ex")
message(STATUS "Executable name: ${EXAGOOP_EXE_NAME}")


# Setting AMReX Configuration
message(STATUS "AMReX Configure Section")
set(AMREX_SUBMOD_LOCATION "${CMAKE_SOURCE_DIR}/Submodules/amrex")
include(SetAmrexOptions)
list(APPEND CMAKE_MODULE_PATH "${AMREX_SUBMOD_LOCATION}/Tools/CMake")
add_subdirectory(${AMREX_SUBMOD_LOCATION})
include(SetAmrexCompileFlags)

# General information about machine, compiler, and build type
message(STATUS "${PROJECT_NAME} Information:")
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "EXAGOOP_PRECISION = ${EXAGOOP_PRECISION}")

include(BuildExeAndLib)
