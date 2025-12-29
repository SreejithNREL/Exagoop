#!/bin/bash
cmake -DCMAKE_INSTALL_PREFIX:PATH=./install \
      -DMPIEXEC_PREFLAGS:STRING=--oversubscribe \
      -DCMAKE_CXX_COMPILER:STRING=$(which amdclang++) \
      -DCMAKE_C_COMPILER:STRING=$(which amdclang) \
      -DCMAKE_BUILD_TYPE:STRING=Release \
      -DEXAGOOP_USE_TEMP=OFF \
      -DEXAGOOP_ENABLE_MPI:BOOL=ON \
      -DEXAGOOP_ENABLE_CUDA:BOOL=OFF \
      -DEXAGOOP_ENABLE_HIP:BOOL=OFF \
      -DAMReX_CUDA_ARCH=Auto \
      -DAMReX_AMD_ARCH="gfx90a" \
      -DPYTHON_EXECUTABLE=$(which python3) \
      -DEXAGOOP_PRECISION:STRING=DOUBLE \
      -DEXAGOOP_DIM=1 \
      ..
#make
cmake --build . --parallel $(sysctl -n hw.ncpu) #&> output.txt
#ctest -j $(sysctl -n hw.ncpu)
