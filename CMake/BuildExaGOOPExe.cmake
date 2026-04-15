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
  # Force mpm_eb_udf_build.cpp to compile as CXX even in CUDA builds.
  # This must be set AFTER target_sources and BEFORE the CUDA block.
  # ---------------------------------------------------------------

  # AFTER: pre-include the shim to pre-empt AMReX_GpuQualifiers.H with
  # empty GPU qualifiers before AMReX_Config.H bakes in AMREX_USE_GPU.
  # CMAKE_CURRENT_LIST_DIR is the CMake/ directory where this file lives.
  set_source_files_properties(
    ${SRC_DIR}/mpm_eb_udf_build.cpp
    PROPERTIES
        LANGUAGE CXX
        COMPILE_OPTIONS
            "-include;${CMAKE_CURRENT_LIST_DIR}/amrex_gpu_host_only.h")

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
  # CUDA: compile all .cpp sources as CUDA except the CXX-only file.
  # The set_source_files_properties above pins mpm_eb_udf_build.cpp
  # to CXX before this loop runs, so it is excluded automatically by
  # the INCLUDE REGEX filter (it matches .cpp but its LANGUAGE property
  # is already CXX, which CUDA's language override should not touch —
  # however we explicitly remove it from the list to be safe).
  # ---------------------------------------------------------------
  if(EXAGOOP_ENABLE_CUDA)
    get_target_property(ALL_SOURCES ${EXAGOOP_EXE_NAME} SOURCES)
    set(CUDA_SOURCES ${ALL_SOURCES})
    list(FILTER CUDA_SOURCES INCLUDE REGEX "\\.cpp$")
    # Remove the CXX-only file from the CUDA compilation list
    list(REMOVE_ITEM CUDA_SOURCES "${SRC_DIR}/mpm_eb_udf_build.cpp")
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

    set_target_properties(${EXAGOOP_EXE_NAME}
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
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
