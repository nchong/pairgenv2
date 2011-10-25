include ../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))

# GLOBAL BUILD OPTIONS
EMBED_CL?=1
OUT = libgpunl.so
all: $(OUT)

# OPENCL TO HEADER CONVERSION
OPENCL2INCLUDE=$(TOP)/cl2include.sh

OBJS = clneighlist.o cudaneighlist.o

$(OUT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) $(SHARED) -o $@ $^

clneighlist.o: clneighlist.cpp decode.cl.h
	$(CXX) -DEMBED_CL=$(EMBED_CL) $(CXXFLAGS) $(INC) -c -o $@ $< 

%.cl.h: %.cl
	$(OPENCL2INCLUDE) $< $@

test: test_clneighlist.cpp clneighlist.o
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) $(SHARED) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o test $(OUT)
