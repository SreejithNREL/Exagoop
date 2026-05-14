#!/bin/bash
# Set HDF5_ROOT in your environment before running this script.
# Examples:
#   export HDF5_ROOT=/opt/homebrew/opt/hdf5-mpi               (macOS Homebrew ARM64)
#   export HDF5_ROOT=/usr/local/opt/hdf5-mpi                  (macOS Homebrew Intel)
#   export HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/openmpi   (Ubuntu/Debian Linux)
#   export HDF5_ROOT=/path/to/hdf5                            (custom install / Spack)

# To use a non-default compiler, add to the cmake call below:
#   -DCMAKE_CXX_COMPILER:STRING=/path/to/your/c++ \
#   -DCMAKE_C_COMPILER:STRING=/path/to/your/cc   \
cmake -DCMAKE_INSTALL_PREFIX:PATH=./install \
      -DMPIEXEC_PREFLAGS:STRING=--oversubscribe \
      -DCMAKE_BUILD_TYPE:STRING=Release \
      -DEXAGOOP_USE_TEMP=OFF \
      -DEXAGOOP_ENABLE_MPI:BOOL=ON \
      -DEXAGOOP_ENABLE_CUDA:BOOL=OFF \
      -DEXAGOOP_ENABLE_HIP:BOOL=OFF \
      -DAMReX_CUDA_ARCH=Auto \
      -DAMReX_AMD_ARCH="gfx90a" \
      -DPYTHON_EXECUTABLE=$(which python3) \
      -DEXAGOOP_PRECISION:STRING=DOUBLE \
      -DEXAGOOP_DIM=2 \
      -DEXAGOOP_USE_HDF5_PARALLEL=ON \
      -DHDF5_ROOT=$HDF5_ROOT \
      -DHDF5_PREFER_PARALLEL=ON \
      -DEXAGOOP_USE_HDF5=ON \
      ..
#make
cmake --build . --parallel $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) #&> output.txt
#ctest -j $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
