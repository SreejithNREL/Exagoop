CXX = g++
CXXFLAGS = -O2 -fPIC -std=c++11

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  CONVECTIVE_LIB = libudf_temp_convective.dylib
  SHARED_FLAG    = -dynamiclib
else
  CONVECTIVE_LIB = libudf_temp_convective.so
  SHARED_FLAG    = -shared
endif

all: $(CONVECTIVE_LIB)

$(CONVECTIVE_LIB): udf_temp_convective.cpp
	$(CXX) $(CXXFLAGS) $(SHARED_FLAG) -o $@ $< -lm

clean:
	rm -f libudf_temp_convective.so libudf_temp_convective.dylib
