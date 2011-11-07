include ../../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))

# GLOBAL BUILD OPTIONS
BLOCK_SIZE=32
override NVCCFLAGS += -DBLOCK_SIZE=$(BLOCK_SIZE)

# OUTPUTS
OUT = libhertz.so
SERIAL_OUT = serial
CUDA_OUT = cuda_tpa cuda_bpa
CL_OUT = cl_tpa cl_bpa
all: $(SERIAL_OUT) $(OUT) $(CUDA_OUT) $(CL_OUT)
INC += -I $(TOP)/common
LIB += -lgpunl

OBJS = hertz_clneighlist.o hertz_wrapper.o
OBJS += hertz_cudaneighlist.o hertz_cudawrapper.o

$(SERIAL_OUT): serialdriver.cpp hertz_kernel.h
	$(CXX) $(CXXFLAGS) -DPOSIX_TIMER $(INC) $(LIB) -o $@ $<

$(OUT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INC) $(SHARED) -o $@ $^ $(LIB)

cl_tpa: driver.cpp $(OUT)
	$(CXX) -DBLOCK_SIZE=$(BLOCK_SIZE) -DKERNEL=TPA -DPOSIX_TIMER $(CXXFLAGS) $(INC) -I. -o $@ $< $(LIB) -L. -lhertz -lrt

cl_bpa: driver.cpp $(OUT)
	$(CXX) -DBLOCK_SIZE=$(BLOCK_SIZE) -DKERNEL=BPA -DPOSIX_TIMER $(CXXFLAGS) $(INC) -I. -o $@ $< $(LIB) -L. -lhertz -lrt

cuda_tpa: cudadriver.cpp $(OUT)
	$(CXX) -DBLOCK_SIZE=$(BLOCK_SIZE) -DKERNEL=TPA -DPOSIX_TIMER $(CXXFLAGS) $(INC) -I. -o $@ $< $(LIB) -L. -lhertz -lrt

cuda_bpa: cudadriver.cpp $(OUT)
	$(CXX) -DBLOCK_SIZE=$(BLOCK_SIZE) -DKERNEL=BPA -DPOSIX_TIMER $(CXXFLAGS) $(INC) -I. -o $@ $< $(LIB) -L. -lhertz -lrt

hertz_kernel.h hertz_cudaneighlist.cu hertz_cudawrapper.cu hertz_clneighlist.cpp hertz_wrapper.cpp:
	../../pairgen.py hertz.yml
	cp reference.hertz_kernel.h new.hertz_kernel.h
	mv -f new.hertz_kernel.h hertz_kernel.h

.PHONY: clean veryclean
clean:
	rm -f *.o $(SERIAL_OUT) $(OUT) $(CUDA_OUT) $(CL_OUT)
veryclean: clean
	rm -f hertz_*
