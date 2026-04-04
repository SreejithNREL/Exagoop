cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# =============================================================================
# ExaGOOP UDF builder — satellite CMake project
#
# This is a minimal standalone CMake project invoked by the exagoop-build-udf
# script.  It is NOT part of the main ExaGOOP CMake build.
#
# Variables injected by the calling script (all required):
#   UDF_SRC   — absolute path to the UDF .cpp source file
#   OUT_DIR   — directory where liblevelset_udf.so/.dylib will be written
#
# Usage (via script):
#   exagoop-build-udf /path/to/geometry.cpp /path/to/output_dir
#
# Usage (direct CMake, for debugging):
#   cmake -S /path/to/CMake/udf_builder \
#         -B /tmp/udf_build             \
#         -DUDF_SRC=/abs/path/to/geometry.cpp \
#         -DOUT_DIR=/abs/path/to/output_dir
#   cmake --build /tmp/udf_build
# =============================================================================

project(ExaGOOP_UDF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------
# Validate required inputs
# ---------------------------------------------------------------
if(NOT DEFINED UDF_SRC OR UDF_SRC STREQUAL "")
    message(FATAL_ERROR
        "UDF_SRC is not set.\n"
        "Pass it with -DUDF_SRC=/absolute/path/to/geometry.cpp")
endif()

if(NOT DEFINED OUT_DIR OR OUT_DIR STREQUAL "")
    message(FATAL_ERROR
        "OUT_DIR is not set.\n"
        "Pass it with -DOUT_DIR=/absolute/path/to/output_dir")
endif()

if(NOT EXISTS "${UDF_SRC}")
    message(FATAL_ERROR "UDF_SRC file not found: ${UDF_SRC}")
endif()

# ---------------------------------------------------------------
# Build shared library
# ---------------------------------------------------------------
add_library(levelset_udf SHARED "${UDF_SRC}")

# Hide all symbols except the explicitly exported ones
# (matches -fvisibility=hidden in GNUmakefile.udf)
set_target_properties(levelset_udf PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    # Place output directly in OUT_DIR (no lib/ subdirectory)
    LIBRARY_OUTPUT_DIRECTORY "${OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${OUT_DIR}"   # for Windows .dll
    # Drop the default "lib" prefix on Windows
    PREFIX "lib"
)

target_compile_options(levelset_udf PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang,Intel>:-O2>
)

# ---------------------------------------------------------------
# Status message
# ---------------------------------------------------------------
message(STATUS "ExaGOOP UDF builder")
message(STATUS "  Source : ${UDF_SRC}")
message(STATUS "  Output : ${OUT_DIR}/liblevelset_udf<ext>")
message(STATUS "")
message(STATUS "  Add to your input file:")
message(STATUS "    eb2.geom_type   = udf_cpp")
message(STATUS "    eb2.udf_so_file = ${OUT_DIR}/liblevelset_udf<ext>")
