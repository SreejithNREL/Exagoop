// amrex_cxx_qualifiers_only.h
//
// Lighter CXX shim for source files that:
//   (a) must be compiled by g++ (not nvcc), AND
//   (b) must have class layouts identical to CUDA TUs.
//
// Contrast with the HEAVY shim (amrex_gpu_host_only.h), which strips
// AMREX_USE_GPU/CUDA entirely.  The heavy shim is safe when no heap-
// allocated AMReX GPU objects cross the ABI boundary, but causes silent
// corruption when a g++-compiled TU creates (via new) objects whose size
// or layout differs from what CUDA TUs expect.
//
// This LIGHTER shim does NOT undefine AMREX_USE_GPU or AMREX_USE_CUDA.
// Class layouts (FabArray, MultiFab, EBFArrayBoxFactory, ...) therefore
// match those seen by CUDA TUs.  It only pre-fills AMReX_GpuQualifiers.H
// with empty macros so that g++ treats AMREX_GPU_HOST_DEVICE (and friends)
// as plain C++ function qualifiers.
//
// Because AMREX_USE_CUDA remains defined, AMReX headers that guard struct
// members with #ifdef AMREX_USE_CUDA will reference cudaStream_t,
// cudaEvent_t, etc.  We include <cuda_runtime_api.h> (after <cstddef> to
// avoid std::size_t / std::nullptr_t ordering conflicts) to make those
// types visible to g++.
//
// IMPORTANT: use only // comments in this file.  The C-style block comment
// delimiter */ appears literally inside glob patterns like Src/**\/*.H and
// would prematurely close any surrounding /* ... */ comment.

// ---------------------------------------------------------------------------
// Step 1: include AMReX config so all AMREX_USE_* macros are defined.
// The include guard prevents re-definition when the TU includes the same
// header transitively later.
// ---------------------------------------------------------------------------
#include <AMReX_Config.H>

#ifdef AMREX_USE_CUDA

// ---------------------------------------------------------------------------
// Step 2a: pre-include C++ stdlib headers BEFORE the CUDA toolkit headers.
//
// cuda_runtime_api.h pulls in C's <stddef.h> (via driver_types.h ->
// builtin_types.h -> stddef.h) which defines size_t and nullptr_t in the
// *global* namespace rather than in std::.  When C++ headers like
// <type_traits> or <ext/type_traits.h> are later included, they fail because
// std::nullptr_t and std::size_t are not yet defined.
//
// Pulling in <cstddef> (and <cstdint> for good measure) first ensures the
// std:: versions are established before the CUDA headers arrive.
// ---------------------------------------------------------------------------
#  include <cstddef>
#  include <cstdint>

// ---------------------------------------------------------------------------
// Step 2b: include the CUDA runtime API so that g++ can resolve cudaStream_t,
// cudaEvent_t, cudaError_t, etc. that appear in AMReX class definitions under
// #ifdef AMREX_USE_CUDA.
// ---------------------------------------------------------------------------
#  include <cuda_runtime_api.h>

// ---------------------------------------------------------------------------
// Step 2c: __clz / __clzll stubs for g++ compilation.
//
// AMReX_Algorithm.H defines clz_wrapper templates under
//   #if defined AMREX_USE_CUDA
// and marks them AMREX_GPU_DEVICE.  Our qualifier shim (Step 3) makes
// AMREX_GPU_DEVICE an empty macro, so g++ parses those templates as ordinary
// host functions whose bodies reference __clz / __clzll.
//
// __clz and __clzll are CUDA device intrinsics only declared when __CUDACC__
// is defined.  Because their arguments are explicitly cast to non-template
// types, g++ two-phase lookup requires a declaration at template-definition
// time (not just at instantiation).
//
// We provide host-callable equivalents using __builtin_clz / __builtin_clzll.
// The #ifndef __CUDACC__ guard makes the stubs invisible in real CUDA TUs
// where the genuine CUDA intrinsics are already in scope.
//
// A grep of AMReX 26.04 Src headers confirms __clz and __clzll are the only
// CUDA intrinsics used under AMREX_USE_CUDA-only guards (without an
// additional __CUDACC__ check) across the whole AMReX source tree.
// ---------------------------------------------------------------------------
#  ifndef __CUDACC__
#    include <climits>
     static inline int __clz(int x) noexcept {
         return x != 0 ? __builtin_clz((unsigned int)x)
                       : (int)(sizeof(int) * CHAR_BIT);
     }
     static inline int __clzll(long long int x) noexcept {
         return x != 0 ? __builtin_clzll((unsigned long long)x)
                       : (int)(sizeof(long long int) * CHAR_BIT);
     }
#  endif // !__CUDACC__

#endif // AMREX_USE_CUDA

// ---------------------------------------------------------------------------
// Step 3: pre-fill the AMReX GPU qualifier guard with no-op macros.
//
// When AMReX_GpuQualifiers.H is included later, its #ifndef guard fires and
// it skips re-defining these macros, so all AMREX_GPU_{HOST,DEVICE,...}
// decorators become empty.  g++ then compiles AMREX_GPU_HOST_DEVICE functions
// as ordinary inline functions with no __device__ / __host__ annotations.
// ---------------------------------------------------------------------------
#ifndef AMREX_GPU_QUALIFIERS_H_
#define AMREX_GPU_QUALIFIERS_H_

#define AMREX_GPU_HOST
#define AMREX_GPU_DEVICE
#define AMREX_GPU_GLOBAL
#define AMREX_GPU_HOST_DEVICE
#define AMREX_GPU_CONSTANT
#define AMREX_GPU_MANAGED
#define AMREX_GPU_DEVICE_MANAGED

#define AMREX_DEVICE_COMPILE 0

#define AMREX_IMPL_STRIP_PARENS(X) AMREX_IMPL_ESC(AMREX_IMPL_ISH X)
#define AMREX_IMPL_ISH(...)        AMREX_IMPL_ISH __VA_ARGS__
#define AMREX_IMPL_ESC(...)        AMREX_IMPL_ESC_(__VA_ARGS__)
#define AMREX_IMPL_ESC_(...)       AMREX_IMPL_VAN_##__VA_ARGS__
#define AMREX_IMPL_VAN_AMREX_IMPL_ISH

#define AMREX_IF_ON_DEVICE(CODE) {}
#define AMREX_IF_ON_HOST(CODE)   { AMREX_IMPL_STRIP_PARENS(CODE) }

#define AMREX_WRONG_NUM_ARGS(...)    static_assert(false, "Wrong number of arguments to macro")
#define AMREX_GET_DGV_MACRO(_1,_2,_3,NAME,...) NAME
#define AMREX_DEVICE_GLOBAL_VARIABLE(...)                          \
    AMREX_GET_DGV_MACRO(__VA_ARGS__, AMREX_DGVARR, AMREX_DGV,     \
                        AMREX_WRONG_NUM_ARGS)(__VA_ARGS__)
#define AMREX_DGV(type,name)          type name
#define AMREX_DGVARR(type,num,name)   type name[num]

#endif // AMREX_GPU_QUALIFIERS_H_
