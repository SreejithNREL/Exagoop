#!/bin/bash
cmake -DCMAKE_INSTALL_PREFIX:PATH=./install \
      -DMPIEXEC_PREFLAGS:STRING=--oversubscribe \
      -DCMAKE_CXX_COMPILER:STRING=$(which clang++) \
      -DCMAKE_C_COMPILER:STRING=$(which clang) \
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
      -DHDF5_ROOT=/opt/homebrew/opt/hdf5-mpi \
      -DHDF5_INCLUDE_DIR=/opt/homebrew/opt/hdf5-mpi/include \
      -DHDF5_LIBRARIES="/opt/homebrew/opt/hdf5-mpi/lib/libhdf5.dylib;/opt/homebrew/opt/hdf5-mpi/lib/libhdf5_hl.dylib" \
      -DHDF5_PREFER_PARALLEL=ON \
      -DEXAGOOP_USE_HDF5=ON \
      ..
#make
cmake --build . --parallel $(sysctl -n hw.ncpu) #&> output.txt
#ctest -j $(sysctl -n hw.ncpu)
