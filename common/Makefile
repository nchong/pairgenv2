# GLOBAL BUILD OPTIONS
EMBED_CL=true
OUT = libgpunl.so

# OPENCL TO HEADER CONVERSION
OPENCL2INCLUDE=../cl2include.sh

# COMPILE OPTIONS
CXX = g++
LOG_LEVEL = LOG_WARN
override CXXFLAGS += -O2 -Wall -Wextra -Werror -pedantic -Wno-variadic-macros -DLOG_LEVEL=$(LOG_LEVEL) -fPIC

# COMMON INCLUDE
override INCLUDEDIR += -I ../../inc
# COMMON LIBS
override LIB = -L../../lib -lclwrapper -lscan -lcommon

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
OBJS = gpuneighlist.o

$(OUT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) $(LIB) $(OPENCL) $(SHARED) -o $@ $^

gpuneighlist.o: gpuneighlist.cpp decode.cl.h
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) -D EMBED_CL=$(EMBED_CL) -c -o $@ $<

%.cl.h: %.cl
	$(OPENCL2INCLUDE) $< $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) -c -o $@ $<

test: test_gpuneighlist.cpp gpuneighlist.o
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(OPENCL_INC) -o $@ $^ $(LIB) $(OPENCL_LIB)

.PHONY: clean
clean:
	rm -f *.o test $(OUT)
