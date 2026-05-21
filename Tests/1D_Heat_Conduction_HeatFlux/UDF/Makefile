CXX = g++
CXXFLAGS = -O2 -fPIC -std=c++11

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  DIRICHLET_LIB = libudf_temp_dirichlet.dylib
  HEATFLUX_LIB  = libudf_temp_heatflux.dylib
  SHARED_FLAG   = -dynamiclib
else
  DIRICHLET_LIB = libudf_temp_dirichlet.so
  HEATFLUX_LIB  = libudf_temp_heatflux.so
  SHARED_FLAG   = -shared
endif

all: $(DIRICHLET_LIB) $(HEATFLUX_LIB)

$(DIRICHLET_LIB): udf_temp_dirichlet.cpp
	$(CXX) $(CXXFLAGS) $(SHARED_FLAG) -o $@ $< -lm

$(HEATFLUX_LIB): udf_temp_heatflux.cpp
	$(CXX) $(CXXFLAGS) $(SHARED_FLAG) -o $@ $< -lm

clean:
	rm -f libudf_temp_dirichlet.so libudf_temp_dirichlet.dylib \
	      libudf_temp_heatflux.so  libudf_temp_heatflux.dylib
