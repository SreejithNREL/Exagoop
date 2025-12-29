# Disable loop-not-vectorized warnings on Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang)$")

  # Map dimension → AMReX target name
  if(EXAGOOP_DIM EQUAL 1)
    set(AMREX_TARGET amrex_1d)
  elseif(EXAGOOP_DIM EQUAL 2)
    set(AMREX_TARGET amrex_2d)
  elseif(EXAGOOP_DIM EQUAL 3)
    set(AMREX_TARGET amrex_3d)
  else()
    message(FATAL_ERROR "Unsupported EXAGOOP_DIM = ${EXAGOOP_DIM}")
  endif()

  # Common flags for all dims
  target_compile_options(
    ${AMREX_TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wno-pass-failed>
  )

  # Extra Clang-only flag
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(
      ${AMREX_TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-ffp-exception-behavior=maytrap>
    )
  endif()

endif()

