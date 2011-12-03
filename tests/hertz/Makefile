include ../../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))

# GLOBAL BUILD OPTIONS
override BLOCK_SIZE ?= 32 #AFFECTS CUDA BUILDS ONLY
override BUILD_DIR ?= .

# COMMON FLAGS
override DEBUG ?= 0
override CXXFLAGS += -DDEBUG=$(DEBUG)
override KERNELFLAGS ?=
override NVCCFLAGS += -Xptxas -v -DDEBUG=$(DEBUG) -DBLOCK_SIZE=$(BLOCK_SIZE) $(KERNELFLAGS)
DRIVERFLAGS = -DBLOCK_SIZE=$(BLOCK_SIZE) -DPOSIX_TIMER
ifneq ($(strip $(TRACE)),)
  override CXXFLAGS  += -DTRACE=$(TRACE)
  override NVCCFLAGS += -DTRACE=$(TRACE)
endif

# OUTPUTS
OUT = libhertz.so
SERIAL_OUT = serial
CUDA_OUT = cuda_tpa cuda_bpa
CL_OUT = cl
BIN_OUT = calc
all: $(SERIAL_OUT) $(OUT) $(CUDA_OUT) $(CL_OUT) $(BIN_OUT)
INC += -I $(TOP)/common
LIB += -lgpunl

OBJS = hertz_clneighlist.o hertz_wrapper.o
OBJS += hertz_cudaneighlist.o

$(OUT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INC) $(SHARED) -o $(BUILD_DIR)/$@ $^ $(LIB)

$(SERIAL_OUT): serialdriver.cpp hertz_kernel.h
	$(CXX) $(CXXFLAGS) $(DRIVERFLAGS) $(INC) $(LIB) -o $(BUILD_DIR)/$@ $<

cl: driver.cpp $(OUT)
	$(CXX) $(DRIVERFLAGS) $(CXXFLAGS) $(INC) -I. -o $(BUILD_DIR)/$@ $< $(LIB) -L$(BUILD_DIR) -lhertz -lrt

cuda_tpa: cudadriver.cpp hertz_cudawrapper.o $(OUT)
	$(CXX) -DKERNEL=TPA -DKERNELFLAGS='$(KERNELFLAGS)' $(DRIVERFLAGS) $(CXXFLAGS) $(INC) -I. -o $(BUILD_DIR)/$@ $< hertz_cudawrapper.o $(LIB) -L$(BUILD_DIR) -lhertz -lrt

cuda_bpa: cudadriver.cpp hertz_cudawrapper.o $(OUT)
	$(CXX) -DKERNEL=BPA -DKERNELFLAGS='$(KERNELFLAGS)' $(DRIVERFLAGS) $(CXXFLAGS) $(INC) -I. -o $(BUILD_DIR)/$@ $< hertz_cudawrapper.o $(LIB) -L$(BUILD_DIR) -lhertz -lrt

calc: hertz_calc.cpp
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/$@ $<

hertz_calc.cpp hertz_kernel.h hertz_cudaneighlist.cu hertz_cudawrapper.cu hertz_clneighlist.cpp hertz_wrapper.cpp:
	../../pairgen.py hertz.yml
	cp reference.hertz_kernel.h new.hertz_kernel.h
	mv -f new.hertz_kernel.h hertz_kernel.h

.PHONY: clean veryclean
clean:
	rm -f *.o $(SERIAL_OUT) $(OUT) $(CUDA_OUT) $(CL_OUT)
veryclean: clean
	rm -f hertz_* $(BIN_OUT)
