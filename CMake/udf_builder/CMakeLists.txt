# ExaGOOPUDF.cmake
# ─────────────────────────────────────────────────────────────────────────────
# Satellite CMakeLists for building a user-defined level-set shared library.
#
# This file is installed by ExaGOOP into <install_prefix>/cmake/ and is
# invoked automatically by the exagoop-build-udf helper script.  Users do
# not edit this file.
#
# Variables consumed (all passed via -D on the cmake command line):
#
#   UDF_SOURCE_FILE   — full path to the user's .cpp file          (required)
#   AMReX_DIR         — path to AMReXConfig.cmake                   (required)
#   UDF_OUTPUT_NAME   — base name of the produced library           (default: levelset_udf)
#   UDF_EXTRA_FLAGS   — any additional compiler flags               (optional)
# ─────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.18)
project(ExaGOOP_UDF LANGUAGES C CXX)

# ── Validate inputs ───────────────────────────────────────────────────────────
if(NOT DEFINED UDF_SOURCE_FILE)
    message(FATAL_ERROR
        "[ExaGOOP UDF] UDF_SOURCE_FILE is not set.\n"
        "Pass -DUDF_SOURCE_FILE=/full/path/to/my_geometry.cpp")
endif()

if(NOT EXISTS "${UDF_SOURCE_FILE}")
    message(FATAL_ERROR
        "[ExaGOOP UDF] Source file not found: ${UDF_SOURCE_FILE}")
endif()

if(NOT DEFINED AMReX_DIR)
    message(FATAL_ERROR
        "[ExaGOOP UDF] AMReX_DIR is not set.\n"
        "The exagoop-build-udf script fills this in automatically.\n"
        "If calling cmake directly, pass -DAMReX_DIR=<path/to/AMReXConfig.cmake>")
endif()

# ── Find the same AMReX ExaGOOP was built against ────────────────────────────
find_package(AMReX REQUIRED
    HINTS "${AMReX_DIR}"
    COMPONENTS EB)

# ── Build the shared library ──────────────────────────────────────────────────
if(NOT DEFINED UDF_OUTPUT_NAME)
    set(UDF_OUTPUT_NAME "levelset_udf")
endif()

# Use a fixed target name; the output file name is controlled by OUTPUT_NAME
add_library(exagoop_udf_lib SHARED "${UDF_SOURCE_FILE}")

target_link_libraries(exagoop_udf_lib PRIVATE AMReX::amrex)
target_compile_features(exagoop_udf_lib PRIVATE cxx_std_17)

if(UDF_EXTRA_FLAGS)
    target_compile_options(exagoop_udf_lib PRIVATE ${UDF_EXTRA_FLAGS})
endif()

# ── Windows: export the required symbol ──────────────────────────────────────
# On Linux/macOS extern "C" is sufficient. On Windows DLL exports require
# either __declspec(dllexport) or a .def file.  We pass a preprocessor flag
# so the user template can use the EXAGOOP_UDF_EXPORT macro portably.
if(WIN32)
    target_compile_definitions(exagoop_udf_lib PRIVATE EXAGOOP_UDF_WINDOWS_EXPORT)
endif()

# ── Output location and name ──────────────────────────────────────────────────
# Always produces:
#   Linux   →  <build_dir>/levelset_udf.so
#   macOS   →  <build_dir>/levelset_udf.dylib
#   Windows →  <build_dir>/Release/levelset_udf.dll
set_target_properties(exagoop_udf_lib PROPERTIES
    OUTPUT_NAME                     "${UDF_OUTPUT_NAME}"
    LIBRARY_OUTPUT_DIRECTORY        "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY        "${CMAKE_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY        "${CMAKE_BINARY_DIR}"
    # MSVC multi-config: put Release dll alongside the .so on other platforms
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "[ExaGOOP UDF] Source : ${UDF_SOURCE_FILE}")
message(STATUS "[ExaGOOP UDF] AMReX  : ${AMReX_DIR}")
message(STATUS "[ExaGOOP UDF] Output : ${CMAKE_BINARY_DIR}/${UDF_OUTPUT_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
