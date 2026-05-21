function(build_exagoop_exe exagoop_exe_name)

  set(SRC_DIR ${CMAKE_SOURCE_DIR}/Source)

  add_executable(${exagoop_exe_name} "")
  set_target_properties(${exagoop_exe_name} PROPERTIES OUTPUT_NAME ${exagoop_exe_name})

  if(WIN32)
    target_link_libraries(${exagoop_exe_name} PRIVATE Ws2_32)
  endif()

  target_include_directories(${exagoop_exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(${exagoop_exe_name} PRIVATE ${SRC_DIR})
  target_include_directories(${exagoop_exe_name} PRIVATE ${CMAKE_BINARY_DIR})

  if(WIN32)
    target_compile_options(${exagoop_exe_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244 /wd4005 /Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=/wd4244
            -Xcompiler=/wd4005
            -Xcompiler=/Zc:preprocessor
            -Xcompiler=/Zc:__cplusplus
            -Xcompiler=/Zc:preprocessor
        >
    )
  endif()

  target_sources(${exagoop_exe_name}
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
      set_source_files_properties(${EXAGOOP_SOURCES} PROPERTIES LANGUAGE CUDA)
    endforeach()
    set_target_properties(${exagoop_exe_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${exagoop_exe_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas --disable-optimizer-constants>)
  endif()

  target_link_libraries(${exagoop_exe_name} PRIVATE amrex)

  if(EXAGOOP_ENABLE_SYCL)
    target_link_options(${exagoop_exe_name} PRIVATE -Wl,--allow-multiple-definition)
  endif()

  install(TARGETS ${exagoop_exe_name}
          RUNTIME DESTINATION bin
          ARCHIVE DESTINATION lib
          LIBRARY DESTINATION lib)

endfunction()
