function(build_exagoop_exe exagoop_exe_name)

  set(SRC_DIR ${CMAKE_SOURCE_DIR}/Source)

  #add_executable(${exagoop_exe_name} "")
  
  if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor --keep --keep-dir C:/cuda_temp")
endif()

  add_executable(${EXAGOOP_EXE_NAME} "")
  set_target_properties(${EXAGOOP_EXE_NAME} PROPERTIES OUTPUT_NAME ${EXAGOOP_EXE_NAME})

  if(WIN32)
    target_link_libraries(${EXAGOOP_EXE_NAME} PRIVATE Ws2_32)
endif()
  

  target_include_directories(${EXAGOOP_EXE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(${EXAGOOP_EXE_NAME} PRIVATE ${SRC_DIR})
  target_include_directories(${EXAGOOP_EXE_NAME} PRIVATE ${CMAKE_BINARY_DIR})

  if(WIN32)
    target_compile_options(${EXAGOOP_EXE_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244 /wd4005 /Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=/wd4244
            -Xcompiler=/wd4005
            -Xcompiler=/Zc:preprocessor
            -Xcompiler=/Zc:__cplusplus
        >
    )
endif()


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
       ${SRC_DIR}/mpm_eb_udf_build.cpp
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
  )


  if(EXAGOOP_ENABLE_CUDA)
    set(pctargets "${exagoop_exe_name}")
    foreach(tgt IN LISTS pctargets)
      get_target_property(EXAGOOP_SOURCES ${tgt} SOURCES)
      list(FILTER EXAGOOP_SOURCES INCLUDE REGEX "\\.cpp")
      # All .cpp files (including mpm_eb_udf_build.cpp) are compiled as CUDA.
      # When AMREX_USE_CUDA is defined, AMReX headers are pervasively CUDA-aware
      # (cudaStream_t, blockDim, __clz, etc.) and cannot be parsed by bare g++.
      # nvcc uses g++ as its host compiler internally, so the host-side ABI for
      # all TUs is identical — there is no ODR risk. The previous approach of
      # keeping mpm_eb_udf_build.cpp as LANGUAGE CXX was the source of the
      # __host__/__device__/cudaStream_t errors seen during compilation.
      set_source_files_properties(${EXAGOOP_SOURCES} PROPERTIES LANGUAGE CUDA)
    endforeach()
    set_target_properties(${exagoop_exe_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${exagoop_exe_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas --disable-optimizer-constants>)
  endif()

  target_link_libraries(${EXAGOOP_EXE_NAME} PRIVATE amrex)

  install(TARGETS ${EXAGOOP_EXE_NAME}
          RUNTIME DESTINATION bin
          ARCHIVE DESTINATION lib
          LIBRARY DESTINATION lib)

endfunction()
