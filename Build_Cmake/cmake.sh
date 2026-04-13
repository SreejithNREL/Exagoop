#!/bin/bash
# ==============================================================
# ExaGOOP CMake Configure Script
# Portable across macOS, Linux, HPC (CUDA/HIP)
#
# Usage:
#   ./configure.sh                        # default: CPU only
#   ENABLE_CUDA=ON ./configure.sh         # CUDA build
#   ENABLE_HIP=ON ./configure.sh          # HIP build
#   ENABLE_HDF5=ON ./configure.sh         # with HDF5
#   ENABLE_MPI=OFF ./configure.sh         # no MPI
#
# Environment overrides:
#   CXX, CC                   compiler overrides
#   HDF5_ROOT                 path to HDF5 installation
#   CUDA_ARCH                 e.g. 80 for A100, 70 for V100, Auto
#   AMD_ARCH                  e.g. gfx90a for MI250
# ==============================================================

# --------------------------------------------------------------
# Detect OS
# --------------------------------------------------------------
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux ;;
    Darwin*)    MACHINE=Mac ;;
    CYGWIN*)    MACHINE=Windows ;;
    MINGW*)     MACHINE=Windows ;;
    *)          MACHINE=Unknown ;;
esac
echo "Detected OS: ${MACHINE}"

# --------------------------------------------------------------
# Parallel build jobs
# --------------------------------------------------------------
if [ "${MACHINE}" = "Mac" ]; then
    NUM_CORES=$(sysctl -n hw.ncpu)
elif [ "${MACHINE}" = "Windows" ]; then
    NUM_CORES=${NUMBER_OF_PROCESSORS:-4}
else
    NUM_CORES=$(nproc)
fi
echo "Using ${NUM_CORES} cores for build"

# --------------------------------------------------------------
# Compiler detection
# Respect user-set CXX/CC, otherwise find best available
# --------------------------------------------------------------
if [ -z "${CXX}" ]; then
    if command -v clang++ &> /dev/null; then
        CXX=$(which clang++)
    elif command -v g++ &> /dev/null; then
        CXX=$(which g++)
    elif command -v icpx &> /dev/null; then
        CXX=$(which icpx)
    fi
fi

if [ -z "${CC}" ]; then
    if command -v clang &> /dev/null; then
        CC=$(which clang)
    elif command -v gcc &> /dev/null; then
        CC=$(which gcc)
    elif command -v icx &> /dev/null; then
        CC=$(which icx)
    fi
fi
echo "CXX: ${CXX}"
echo "CC:  ${CC}"

# --------------------------------------------------------------
# GPU backend (mutually exclusive)
# --------------------------------------------------------------
ENABLE_CUDA=${ENABLE_CUDA:-OFF}
ENABLE_HIP=${ENABLE_HIP:-OFF}
ENABLE_SYCL=${ENABLE_SYCL:-OFF}

# CUDA architecture: default Auto, override with CUDA_ARCH=80 etc.
CUDA_ARCH=${CUDA_ARCH:-Auto}
# AMD architecture: default gfx90a (MI250), override with AMD_ARCH=gfx908 etc.
AMD_ARCH=${AMD_ARCH:-gfx90a}

# --------------------------------------------------------------
# MPI
# --------------------------------------------------------------
ENABLE_MPI=${ENABLE_MPI:-ON}

# MPI oversubscribe flag (OpenMPI only, skip on MPICH/Cray)
MPI_IMPL=$(mpirun --version 2>&1 | head -1)
if echo "${MPI_IMPL}" | grep -qi "open mpi"; then
    MPIEXEC_PREFLAGS="--oversubscribe"
else
    MPIEXEC_PREFLAGS=""
fi

# --------------------------------------------------------------
# HDF5
# --------------------------------------------------------------
ENABLE_HDF5=${ENABLE_HDF5:-OFF}
ENABLE_HDF5_PARALLEL=${ENABLE_HDF5_PARALLEL:-OFF}

# HDF5 path detection if enabled
HDF5_FLAGS=""
if [ "${ENABLE_HDF5}" = "ON" ]; then
    if [ -z "${HDF5_ROOT}" ]; then
        # Try common locations
        if [ "${MACHINE}" = "Mac" ]; then
            # Homebrew Apple Silicon
            if [ -d "/opt/homebrew/opt/hdf5-mpi" ]; then
                HDF5_ROOT="/opt/homebrew/opt/hdf5-mpi"
            # Homebrew Intel Mac
            elif [ -d "/usr/local/opt/hdf5-mpi" ]; then
                HDF5_ROOT="/usr/local/opt/hdf5-mpi"
            elif [ -d "/usr/local/opt/hdf5" ]; then
                HDF5_ROOT="/usr/local/opt/hdf5"
            fi
        elif [ "${MACHINE}" = "Linux" ]; then
            # Spack, module, or system install
            if [ -d "/usr/lib/x86_64-linux-gnu/hdf5/openmpi" ]; then
                HDF5_ROOT="/usr/lib/x86_64-linux-gnu/hdf5/openmpi"
            elif command -v h5cc &> /dev/null; then
                HDF5_ROOT=$(h5cc -showconfig 2>/dev/null | grep "Installation point" | awk '{print $NF}')
            fi
        fi
    fi

    if [ -z "${HDF5_ROOT}" ]; then
        echo "WARNING: ENABLE_HDF5=ON but HDF5_ROOT not found. Set HDF5_ROOT manually."
    else
        echo "HDF5_ROOT: ${HDF5_ROOT}"
        # Detect shared lib extension
        if [ "${MACHINE}" = "Mac" ]; then
            LIB_EXT="dylib"
        elif [ "${MACHINE}" = "Windows" ]; then
            LIB_EXT="dll"
        else
            LIB_EXT="so"
        fi

        HDF5_FLAGS="\
      -DHDF5_ROOT=${HDF5_ROOT} \
      -DHDF5_INCLUDE_DIR=${HDF5_ROOT}/include \
      -DHDF5_LIBRARIES=${HDF5_ROOT}/lib/libhdf5.${LIB_EXT};${HDF5_ROOT}/lib/libhdf5_hl.${LIB_EXT} \
      -DHDF5_PREFER_PARALLEL=${ENABLE_HDF5_PARALLEL}"
    fi
fi

# --------------------------------------------------------------
# Python
# --------------------------------------------------------------
if command -v python3 &> /dev/null; then
    PYTHON_EXE=$(which python3)
else
    PYTHON_EXE=$(which python)
fi

# --------------------------------------------------------------
# CMake configure
# --------------------------------------------------------------
cmake \
    -DCMAKE_INSTALL_PREFIX:PATH=./install \
    -DMPIEXEC_PREFLAGS:STRING="${MPIEXEC_PREFLAGS}" \
    -DCMAKE_CXX_COMPILER:STRING="${CXX}" \
    -DCMAKE_C_COMPILER:STRING="${CC}" \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DEXAGOOP_DIM=2 \
    -DEXAGOOP_PRECISION:STRING=DOUBLE \
    -DEXAGOOP_ENABLE_EB=ON \
    -DEXAGOOP_USE_TEMP=ON \
    -DEXAGOOP_ENABLE_MPI:BOOL=${ENABLE_MPI} \
    -DEXAGOOP_ENABLE_OMP:BOOL=OFF \
    -DEXAGOOP_ENABLE_CUDA:BOOL=${ENABLE_CUDA} \
    -DEXAGOOP_ENABLE_HIP:BOOL=${ENABLE_HIP} \
    -DEXAGOOP_ENABLE_SYCL:BOOL=${ENABLE_SYCL} \
    -DAMReX_CUDA_ARCH=${CUDA_ARCH} \
    -DAMReX_AMD_ARCH="${AMD_ARCH}" \
    -DPYTHON_EXECUTABLE="${PYTHON_EXE}" \
    -DEXAGOOP_USE_HDF5=${ENABLE_HDF5} \
    -DEXAGOOP_USE_HDF5_PARALLEL=${ENABLE_HDF5_PARALLEL} \
    ${HDF5_FLAGS} \
    ..

# --------------------------------------------------------------
# Build
# --------------------------------------------------------------
cmake --build . --parallel ${NUM_CORES}

# --------------------------------------------------------------
# Test (uncomment to enable)
# --------------------------------------------------------------
# ctest -j ${NUM_CORES}
