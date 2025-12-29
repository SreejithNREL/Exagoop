MPM_HOME ?= ../
AMREX_HOME ?= ${MPM_HOME}/Submodules/amrex

EBASE = ExaGOOP

DEBUG	= FALSE

DIM	= 1

COMP    = gnu

USE_TEMP = FALSE


USE_MPI   = TRUE 
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_EB    = TRUE

# ------------------------------------------------------------------
# Enforce: EB is not available in 1D
# ------------------------------------------------------------------
ifeq ($(DIM),1)
  ifeq ($(USE_EB),TRUE)
    $(warning USE_EB=TRUE requested, but EB is not available in 1D. Forcing USE_EB=FALSE.)
  endif
  USE_EB := FALSE
endif

DEFINES += -DUSE_EB=$(USE_EB)
DEFINES += -DTRUE=1 -DFALSE=0


TINY_PROFILE = FALSE
USE_PARTICLES = TRUE
BL_NO_FORT = TRUE

DEFINES += -DUSE_TEMP=$(USE_TEMP)
DEFINES += -DUSE_EB=$(USE_EB)
DEFINES += -DTRUE=1 -DFALSE=0

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Particle/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package
# True if DIM >= 2
GE_DIM_2 := $(filter $(DIM),2 3 4 5 6 7 8 9)

ifeq ($(USE_EB),TRUE)
  ifneq ($(GE_DIM_2),)
    include $(AMREX_HOME)/Src/EB/Make.package
  endif
endif

include $(MPM_HOME)/Source/Make.package

INCLUDE_LOCATIONS += $(MPM_HOME)/Source
VPATH_LOCATIONS   += $(MPM_HOME)/Source

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
