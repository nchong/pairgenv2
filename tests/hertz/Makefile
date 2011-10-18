# COMPILE OPTIONS
CXX = g++
LOG_LEVEL = LOG_WARN
override CXXFLAGS += -O2 -Wall -Wextra -Werror -pedantic -Wno-variadic-macros -fPIC

# COMMON INCLUDE
override INCLUDEDIR += -I ../../../inc -I ../../common
# COMMON LIBS
override LIB = -L../../../lib -lclwrapper -lscan -lgpunl -lcommon

# OPENCL LIBRARIES BASED ON OS
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
OPENCL_LIB = -framework OpenCL
OPENCL_INC =
SHARED = -dynamiclib
endif

ifeq ($(UNAME), Linux)
.PHONY: .check-env
AMDAPP ?= /opt/AMDAPP/include
.check-env:
	@if [ ! -d "${AMDAPP}" ]; then \
		echo "ERROR: set AMDAPP variable."; exit 1; \
	fi
OPENCL_LIB = -l OpenCL 
OPENCL_INC = -I ${AMDAPP}
SHARED = -shared
endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) -c -o $@ $<

OBJS = hertz_gpuneighlist.o	hertz_wrapper.o

driver: driver.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) -o $@ $^ $(LIB) $(OPENCL_LIB)

hertz_gpuneighlist.cpp hertz_wrapper.cpp:
	../../pairgen.py hertz.yml
	cp reference.hertz_pair_kernel.cl new.hertz_pair_kernel.cl
	mv -f new.hertz_pair_kernel.cl hertz_pair_kernel.cl

clean:
	rm -f hertz_* driver
