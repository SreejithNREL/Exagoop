# ─────────────────────────────────────────────────────────────────────────────
# ExaGOOP CMakeLists.txt — additions for UDF level-set support
# ─────────────────────────────────────────────────────────────────────────────
# Paste these blocks into your existing CMakeLists.txt at the indicated points.
# Do NOT replace your entire CMakeLists.txt with this file.
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. After your add_executable() or add_library() call ─────────────────────

# Link the dynamic loader library.
# On Linux/macOS: libdl (provides dlopen/dlsym).
# On Windows: kernel32 already provides LoadLibrary — CMAKE_DL_LIBS is empty.
target_link_libraries(ExaGOOP PRIVATE ${CMAKE_DL_LIBS})

# Add the Source directory to include path so mpm_udf_loader.H is found.
# (Skip if Source/ is already in your target's include directories.)
target_include_directories(ExaGOOP PRIVATE "${CMAKE_SOURCE_DIR}/Source")


# ── 2. After your install() targets, add UDF install rules ───────────────────

# Directory where ExaGOOPUDF.cmake and the helper scripts will live
set(EXAGOOP_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/cmake")

# Bake the install-time paths into the helper scripts
configure_file(
    "${CMAKE_SOURCE_DIR}/tools/exagoop-build-udf.in"
    "${CMAKE_BINARY_DIR}/exagoop-build-udf"
    @ONLY)

configure_file(
    "${CMAKE_SOURCE_DIR}/tools/exagoop-build-udf.bat.in"
    "${CMAKE_BINARY_DIR}/exagoop-build-udf.bat"
    @ONLY)

# Install the satellite CMakeLists and helper scripts
install(FILES
    "${CMAKE_SOURCE_DIR}/cmake/ExaGOOPUDF.cmake"
    DESTINATION cmake)

install(PROGRAMS
    "${CMAKE_BINARY_DIR}/exagoop-build-udf"
    DESTINATION bin)

install(FILES
    "${CMAKE_BINARY_DIR}/exagoop-build-udf.bat"
    DESTINATION bin)

install(FILES
    "${CMAKE_SOURCE_DIR}/tools/GNUmakefile.udf"
    DESTINATION share/exagoop)

# Install UDF templates so users have a starting point
install(FILES
    "${CMAKE_SOURCE_DIR}/Source/udf_templates/levelset_simple_template.cpp"
    "${CMAKE_SOURCE_DIR}/Source/udf_templates/levelset_advanced_template.cpp"
    DESTINATION share/exagoop/udf_templates)


# ── 3. Variables used inside ExaGOOPUDF.cmake (auto-filled at configure time) ─
# These are resolved when the user runs exagoop-build-udf, not at ExaGOOP
# build time. They come from the paths CMake already knows.

# AMReX_DIR is set by find_package(AMReX) — used inside the configure_file above.
# If your project uses a different variable name, replace AMReX_DIR below.
#
# Note: @ONLY substitution in the .in files uses these variable names literally,
# so the names here must match the @VAR@ tokens in the .in files.
#
#   @EXAGOOP_CMAKE_DIR@  →  where ExaGOOPUDF.cmake is installed
#   @AMReX_DIR@          →  AMReX cmake config directory
#   @CMAKE_INSTALL_PREFIX@  →  ExaGOOP install prefix (for help text)
#
# These are standard CMake variables — no extra work needed.
