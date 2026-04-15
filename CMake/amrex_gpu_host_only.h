
#include <AMReX_Config.H>

#ifdef AMREX_USE_CUDA
#  undef AMREX_USE_CUDA
#endif
#ifdef AMREX_USE_GPU
#  undef AMREX_USE_GPU
#endif
#ifdef AMREX_USE_HIP
#  undef AMREX_USE_HIP
#endif
#ifdef AMREX_USE_NVML
#  undef AMREX_USE_NVML
#endif
#ifdef AMREX_USE_GPU_RDC
#  undef AMREX_USE_GPU_RDC
#endif
#ifdef BL_COALESCE_FABS
#  undef BL_COALESCE_FABS
#endif
/* Restore the no-GPU fallback value for AMREX_GPU_MAX_THREADS */
#ifdef AMREX_GPU_MAX_THREADS
#  undef AMREX_GPU_MAX_THREADS
#endif
#define AMREX_GPU_MAX_THREADS 0

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

#endif /* AMREX_GPU_QUALIFIERS_H_ */
