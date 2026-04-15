function(build_exagoop_exe exagoop_exe_name)

  set(SRC_DIR ${CMAKE_SOURCE_DIR}/Source)

  # ---------------------------------------------------------------
  # Windows: enforce conforming preprocessor BEFORE add_executable
  # so the flag propagates to all source files in this TU.
  # Required by CUDA 12+ / CCCL on MSVC.
  # ---------------------------------------------------------------
  if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor"
        PARENT_SCOPE)
  endif()

  add_executable(${EXAGOOP_EXE_NAME} "")
  set_target_properties(${EXAGOOP_EXE_NAME}
      PROPERTIES OUTPUT_NAME ${EXAGOOP_EXE_NAME})

  target_include_directories(${EXAGOOP_EXE_NAME}
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
              ${SRC_DIR}
              ${CMAKE_BINARY_DIR})

  target_sources(${EXAGOOP_EXE_NAME}
     PRIVATE
       ${SRC_DIR}/constants.H
       ${SRC_DIR}/aesthetics.H
       ${SRC_DIR}/aesthetics.cpp
       ${SRC_DIR}/utilities.H
       ${SRC_DIR}/utilities.cpp
       ${SRC_DIR}/mpm_diagnostics.cpp
       ${SRC_DIR}/mpm_init.cpp
       ${SRC_DIR}/mpm_particle_container.H
       ${SRC_DIR}/mpm_particle_timestep.cpp
       ${SRC_DIR}/nodal_data_ops.H
       ${SRC_DIR}/constitutive_models.H
       ${SRC_DIR}/mpm_eb.cpp
       ${SRC_DIR}/mpm_kernels.H
       ${SRC_DIR}/mpm_particle_grid_ops.cpp
       ${SRC_DIR}/mpm_specs.H
       ${SRC_DIR}/interpolants.H
       ${SRC_DIR}/mpm_check_pair.H
       ${SRC_DIR}/mpm_eb.H
       ${SRC_DIR}/mpm_udf_loader.H
       ${SRC_DIR}/mpm_particle_container.cpp
       ${SRC_DIR}/mpm_particle_outputs.cpp
       ${SRC_DIR}/nodal_data_ops.cpp
       ${SRC_DIR}/main.cpp
       # CXX-only TU — contains EB2::Build with UDF lambda; must never be
       # compiled by nvcc (see file header for explanation).
       ${SRC_DIR}/mpm_eb_udf_build.cpp
  )

  # ---------------------------------------------------------------
  # mpm_eb.cpp and mpm_eb_udf_build.cpp must ALWAYS be compiled by g++
  # (LANGUAGE CXX), never by nvcc.
  #
  # mpm_eb_udf_build.cpp (pre-existing requirement):
  #   EB2::Build with a host-lambda causes nvcc's device-function scanner to
  #   pull in stub.c, producing an unresolvable link error.
  #
  # mpm_eb.cpp (new requirement):
  #   nvcc compiling the GShopLevel<GeometryShop<SphereIF>> / GFab / MultiGFab
  #   template machinery generates host code that differs subtly from g++'s
  #   output (different template instantiation), causing a segfault inside
  #   BaseFab<uint32_t>::define() during EB2::Build at runtime.
  #
  # Both files use the LIGHTER shim (amrex_cxx_qualifiers_only.h) rather than
  # the old heavy shim (amrex_gpu_host_only.h).  The lighter shim:
  #   - keeps AMREX_USE_GPU/CUDA defined → class layouts identical to CUDA TUs
  #     (prevents ABI mismatch for heap-allocated EBFArrayBoxFactory / MultiFab)
  #   - strips GPU function qualifiers → g++ compiles AMREX_GPU_HOST_DEVICE as
  #     ordinary inline functions
  #   - includes <cuda_runtime_api.h> → g++ resolves cudaStream_t / cudaEvent_t
  #     that appear in AMReX class definitions under #ifdef AMREX_USE_CUDA
  #
  # We also pass the CUDA toolkit include directories (-I flags) so that g++
  # can find cuda_runtime_api.h when AMREX_USE_CUDA is defined.
  # CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES is set automatically by CMake when
  # enable_language(CUDA) has been called.
  # ---------------------------------------------------------------

  set(_cxx_eb_opts
      "-include" "${CMAKE_CURRENT_LIST_DIR}/amrex_cxx_qualifiers_only.h")
  foreach(_dir ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
      list(APPEND _cxx_eb_opts "-I${_dir}")
  endforeach()

  set_source_files_properties(
    ${SRC_DIR}/mpm_eb.cpp
    ${SRC_DIR}/mpm_eb_udf_build.cpp
    PROPERTIES
        LANGUAGE CXX
        COMPILE_OPTIONS "${_cxx_eb_opts}")

  # ---------------------------------------------------------------
  # Windows-specific compile definitions and flags
  # ---------------------------------------------------------------
  if(WIN32)
    target_compile_definitions(${EXAGOOP_EXE_NAME} PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)

    target_compile_options(${EXAGOOP_EXE_NAME} PRIVATE
        # C4244: conversion from 'long' to 'int' (MSVC treats long as 32-bit)
        # C4005: macro redefinition (AMReX lexer headers)
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244 /wd4005 /Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4244 -Xcompiler=/wd4005>)
  endif()

  # ---------------------------------------------------------------
  # CUDA: mark all remaining .cpp sources as CUDA TUs.
  #
  # mpm_eb.cpp and mpm_eb_udf_build.cpp are already pinned to CXX above
  # and are removed from the list before the LANGUAGE CUDA override.
  #
  # Why we do NOT set CUDA_SEPARABLE_COMPILATION ON here:
  #   AMReX is compiled with RDC (AMReX_GPU_RDC defaults to ON).  If
  #   ExaGOOP's CUDA TUs also used RDC, the GPU-annotated inline functions
  #   from AMReX_EB2_IF_*.H (included via mpm_eb.H) would be emitted as
  #   exported device symbols in both AMReX's objects AND ExaGOOP's objects,
  #   creating an ODR violation that corrupts the host heap at runtime.
  #   Without RDC, ExaGOOP's device code is compiled to standalone cubins;
  #   every AMREX_FORCE_INLINE device function is inlined at its call site
  #   and no device symbols are exported, so there is nothing to clash with
  #   AMReX's RDC objects.  ExaGOOP has no cross-TU device calls, so RDC
  #   is not needed.
  # ---------------------------------------------------------------
  if(EXAGOOP_ENABLE_CUDA)
    get_target_property(ALL_SOURCES ${EXAGOOP_EXE_NAME} SOURCES)
    set(CUDA_SOURCES ${ALL_SOURCES})
    list(FILTER CUDA_SOURCES INCLUDE REGEX "\\.cpp$")
    # Both CXX-pinned files are already excluded; remove them from the list
    # so the LANGUAGE CUDA assignment below cannot override their CXX setting.
    list(REMOVE_ITEM CUDA_SOURCES "${SRC_DIR}/mpm_eb_udf_build.cpp")
    list(REMOVE_ITEM CUDA_SOURCES "${SRC_DIR}/mpm_eb.cpp")

    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

    # No CUDA_SEPARABLE_COMPILATION — see comment above.
    target_compile_options(${EXAGOOP_EXE_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas --disable-optimizer-constants>)
  endif()

  # ---------------------------------------------------------------
  # Link libraries
  # ---------------------------------------------------------------
  target_link_libraries(${EXAGOOP_EXE_NAME} PRIVATE amrex)

  # Dynamic loader library (needed by mpm_udf_loader.H / dlopen)
  if(NOT WIN32)
    target_link_libraries(${EXAGOOP_EXE_NAME} PRIVATE dl)
  endif()
  # On Windows, LoadLibrary lives in kernel32 which is linked by default.

  # ---------------------------------------------------------------
  # Install
  # ---------------------------------------------------------------
  install(TARGETS ${EXAGOOP_EXE_NAME}
          RUNTIME DESTINATION bin
          ARCHIVE DESTINATION lib
          LIBRARY DESTINATION lib)

endfunction()
