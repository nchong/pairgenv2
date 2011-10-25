# GLOBAL BUILD OPTIONS
OUT=libhertz.so

# COMPILE OPTIONS
CXX = g++
LOG_LEVEL = LOG_WARN
override CXXFLAGS += -O2 -Wall -Wextra -Werror -pedantic -Wno-variadic-macros -fPIC

NVCC = nvcc
override NVCCFLAGS += -m64 -Xcompiler "-Wall -fPIC"

# COMMON INCLUDE
override INCLUDEDIR += -I ../../../inc -I ../../common
# COMMON LIBS
override LIB = -L../../../lib -lclwrapper -lscan -lgpunl -lcommon

# OPENCL LIBRARIES BASED ON OS
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
SHARED = -dynamiclib
OPENCL_LIB = -framework OpenCL
OPENCL_INC =
CUDADIR ?= /usr/local/cuda/
endif

ifeq ($(UNAME), Linux)
SHARED = -shared
AMDAPP ?= /opt/AMDAPP/include
OPENCL_LIB = -lOpenCL
OPENCL_INC = -I $(AMDAPP)
CUDADIR ?= /apps/cuda/4.0.11
.PHONY: .check-opencl-env
.check-opencl-env:
	@if [ ! -d "${AMDAPP}" ]; then \
		echo "ERROR: set AMDAPP variable."; exit 1; \
	fi
endif

# CUDA LIBRARIES
.PHONY: .check-cuda-env
.check-cuda-env:
	@if [ ! -d "$(CUDADIR)" ]; then \
		echo "ERROR: set CUDADIR variable."; exit 1; \
	fi
CUDA_LIB = -L$(CUDADIR)/lib -lcuda -lcudart
CUDA_INC = -I $(CUDADIR)/include

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDEDIR) -c -o $@ $<

all: .check-cuda-env $(OUT) driver_serial_timer driver_posix_timer

OBJS = hertz_clneighlist.o	hertz_wrapper.o hertz_cudaneighlist.o

$(OUT): $(OBJS)
	$(CXX) $(CXXFLAGS) \
		$(INCLUDEDIR) $(OPENCL_INC) $(CUDA_INC) \
		$(LIB) $(OPENCL_LIB) $(CUDA_LIB) \
		$(SHARED) -o $@ $^

driver_serial_timer: driver.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) $(CUDA_INC) -I. -o $@ $^ $(LIB) -L. -lhertz $(OPENCL_LIB) $(CUDA_LIB)

driver_posix_timer: driver.cpp
	$(CXX) $(CXXFLAGS) -DPOSIX_TIMER $(INCLUDEDIR) $(OPENCL_INC) $(CUDA_INC) -I. -o $@ $^ $(LIB) -L. -lhertz $(OPENCL_LIB) $(CUDA_LIB) -lrt

hertz_cudaneighlist.cu hertz_clneighlist.cpp hertz_wrapper.cpp:
	../../pairgen.py hertz.yml
	cp reference.hertz_kernel.h new.hertz_kernel.h
	mv -f new.hertz_kernel.h hertz_kernel.h

clean:
	rm -f hertz_* driver_*_timer $(OUT)
