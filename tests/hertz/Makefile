OUT=libhertz.so

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
OPENCL_LIB = -lOpenCL
OPENCL_INC = -I ${AMDAPP}
SHARED = -shared
endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) -c -o $@ $<

all: $(OUT) driver_serial_timer driver_posix_timer

OBJS = hertz_gpuneighlist.o	hertz_wrapper.o

$(OUT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) $(LIB) $(OPENCL) $(SHARED) -o $@ $^

driver_serial_timer: driver.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) -I. -o $@ $^ $(LIB) -L. -lhertz $(OPENCL_LIB)

driver_posix_timer: driver.cpp
	$(CXX) $(CXXFLAGS) -DPOSIX_TIMER $(INCLUDEDIR) $(OPENCL_INC) -I. -o $@ $^ $(LIB) -L. -lhertz $(OPENCL_LIB) -lrt

hertz_gpuneighlist.cpp hertz_wrapper.cpp:
	../../pairgen.py hertz.yml
	cp reference.hertz_kernel.h new.hertz_kernel.h
	mv -f new.hertz_kernel.h hertz_kernel.h

clean:
	rm -f hertz_* driver_*_timer $(OUT)
