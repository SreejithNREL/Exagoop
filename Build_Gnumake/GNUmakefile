MPM_HOME ?= ../../
AMREX_HOME ?= ${MPM_HOME}/Submodules/amrex

EBASE = ExaGOOP

include $(MPM_HOME)/Build_Gnumake/Make.local 
include $(AMREX_HOME)/Tools/GNUMake/Make.defs

# ===============================================================
# AMReX configuration
# ===============================================================
DIM     = 2
COMP       = gnu
PRECISION  = DOUBLE
USE_EB     = FALSE
USE_HYPRE  = FALSE

# ===============================================================
# Profiling
# ===============================================================
PROFILE        = FALSE
TINY_PROFILE   = FALSE
COMM_PROFILE   = FALSE
TRACE_PROFILE  = FALSE
MEM_PROFILE    = FALSE
USE_GPROF      = FALSE
BL_NO_FORT     = TRUE

# ===============================================================
# Parallelism
# ===============================================================
USE_MPI   = TRUE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_SYCL  = FALSE

# ===============================================================
# Debugging
# ===============================================================
DEBUG             = FALSE
FSANITIZER        = FALSE
THREAD_SANITIZER  = FALSE

# ===============================================================
# ExaGOOP configuration
# ===============================================================
USE_TEMP     = TRUE
USE_PARTICLES = TRUE

# ===============================================================
# Precision 
# ===============================================================
PRECISION ?= DOUBLE

ifeq ($(PRECISION),SINGLE)
  DEFINES += -DAMREX_USE_FLOAT
  DEFINES += -DEXAGOOP_PRECISION_STR=\"single\"
endif

ifeq ($(PRECISION),DOUBLE)
  DEFINES += -DAMREX_USE_DOUBLE
  DEFINES += -DEXAGOOP_PRECISION_STR=\"double\"
endif


# ===============================================================
# GPU backend detection
# ===============================================================
ifeq ($(USE_CUDA),TRUE)
  CXXFLAGS += -DEXAGOOP_GPU_BACKEND=\"CUDA\"
endif

ifeq ($(USE_HIP),TRUE)
  CXXFLAGS += -DEXAGOOP_GPU_BACKEND=\"HIP\"
endif

ifeq ($(USE_SYCL),TRUE)
  CXXFLAGS += -DEXAGOOP_GPU_BACKEND=\"SYCL\"
endif

ifndef EXAGOOP_GPU_BACKEND
  CXXFLAGS += -DEXAGOOP_GPU_BACKEND=\"None\"
endif

# ===============================================================
# Enforce EB rules (AMReX limitation)
# ===============================================================
ifeq ($(DIM),1)
  ifeq ($(USE_EB),TRUE)
    $(warning USE_EB=TRUE requested, but EB is not available in 1D. Forcing USE_EB=FALSE.)
  endif
  USE_EB := FALSE
endif

DEFINES += -DUSE_EB=$(USE_EB)
DEFINES += -DUSE_TEMP=$(USE_TEMP)
DEFINES += -DTRUE=1 -DFALSE=0

include $(AMREX_HOME)/Tools/GNUMake/Make.defs
# ===============================================================
# Include AMReX and ExaGOOP sources
# ===============================================================

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Particle/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package
#include $(AMREX_HOME)/Src/Extern/HDF5/Make.package

# ===============================================================
# Git hashes (ExaGOOP + AMReX)
# ===============================================================
EXAGOOP_GIT_HASH := $(shell git rev-parse --short HEAD)
CXXFLAGS += -DEXAGOOP_GIT_HASH=\"$(EXAGOOP_GIT_HASH)\"

AMREX_GIT_HASH := $(shell git -C $(AMREX_HOME) rev-parse --short HEAD)
CXXFLAGS += -DAMREX_GIT_HASH=\"$(AMREX_GIT_HASH)\"

ifeq ($(USE_EB),TRUE)
  ifneq ($(filter $(DIM),2 3 4 5 6 7 8 9),)
    include $(AMREX_HOME)/Src/EB/Make.package
  endif
endif

include $(MPM_HOME)/Source/Make.package

INCLUDE_LOCATIONS += $(MPM_HOME)/Source
VPATH_LOCATIONS   += $(MPM_HOME)/Source

include $(AMREX_HOME)/Tools/GNUMake/Make.rules


