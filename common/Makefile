include ../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))

# ---------------------------------------------------------------------------
# BUILD OPTIONS
# ---------------------------------------------------------------------------
#
# If EMBED_CL is
#   0 then the OpenCL kernels are compiled from file.
#   1 then the OpenCL kernels are inserted as a char array into the program source (default).
# It is best to set this to 1 to avoid having to carry around the OpenCL source files.
EMBED_CL ?= 1

# If HOST_DECODE is
#   0 then we use the device to decode neighbor lists (default).
#   1 then we fallback to using a host implementation.
HOST_DECODE ?= 0

# If PARANOID is
#   0 then we assume the decode was successful.
#   1 then we do an end-to-end check of the decode.
PARANOID  ?= 0
# ---------------------------------------------------------------------------

OUT = libgpunl.so
all: $(OUT)
override CXXFLAGS += -DHOST_DECODE=$(HOST_DECODE) -DPARANOID=$(PARANOID)

# OPENCL TO HEADER CONVERSION
OPENCL2INCLUDE=$(TOP)/cl2include.sh

OBJS = clneighlist.o cudaneighlist.o hostneighlist.o

$(OUT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) $(SHARED) -o $@ $^

clneighlist.o: clneighlist.cpp decode.cl.h
	$(CXX) -DEMBED_CL=$(EMBED_CL) $(CXXFLAGS) $(INC) -c -o $@ $< 

%.cl.h: %.cl
	$(OPENCL2INCLUDE) $< $@

cltest: test_clneighlist.cpp clneighlist.o hostneighlist.o
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $@ $^

cudatest: test_cudaneighlist.cpp cudaneighlist.o hostneighlist.o
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o decode.cl.h cltest cudatest $(OUT)
