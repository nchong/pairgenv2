# GLOBAL BUILD OPTIONS
EMBED_CL=1
OUT = libgpunl.so

# OPENCL TO HEADER CONVERSION
OPENCL2INCLUDE=../cl2include.sh

# COMPILE OPTIONS
CXX = g++
LOG_LEVEL = LOG_WARN
override CXXFLAGS += -O2 -Wall -Wextra -Werror -pedantic -Wno-variadic-macros -DLOG_LEVEL=$(LOG_LEVEL) -fPIC
NVCC = nvcc
override NVCCFLAGS += -O2 -m64 -Xcompiler "-Wall -fPIC"

# COMMON INCLUDE
override INCLUDEDIR += -I ../../inc -I /usr/local/cuda/include/
# COMMON LIBS
override LIB += -L../../lib -lclwrapper -lscan -lcommon
override LIB += -L/usr/local/cuda/lib -lcuda -lcudart

# OPENCL LIBRARIES BASED ON OS
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
OPENCL_LIB = -framework OpenCL
OPENCL_INC =
SHARED = -dynamiclib
all: $(OUT)
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
all: .check-env $(OUT)
endif

# FINALLY THE RULES
OBJS = clneighlist.o cudaneighlist.o

$(OUT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) $(LIB) $(OPENCL) $(SHARED) -o $@ $^

clneighlist.o: clneighlist.cpp decode.cl.h
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) -D EMBED_CL=$(EMBED_CL) -c -o $@ $<

%.cl.h: %.cl
	$(OPENCL2INCLUDE) $< $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDEDIR) -c -o $@ $<

test: test_clneighlist.cpp clneighlist.o
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) -o $@ $^ $(LIB) $(OPENCL_LIB)

.PHONY: clean
clean:
	rm -f *.o test $(OUT)
