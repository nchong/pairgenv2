include ../../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))

# GLOBAL BUILD OPTIONS
override BLOCK_SIZE ?= 32 #AFFECTS CUDA BUILDS ONLY
override BUILD_DIR ?= .

# COMMON FLAGS
override NVCCFLAGS += -DBLOCK_SIZE=$(BLOCK_SIZE)
DRIVERFLAGS = -DBLOCK_SIZE=$(BLOCK_SIZE) -DPOSIX_TIMER
ifneq ($(strip $(TRACE)),)
  override CXXFLAGS  += -DTRACE=$(TRACE)
  override NVCCFLAGS += -DTRACE=$(TRACE)
endif

# OUTPUTS
OUT = libhertz.so
SERIAL_OUT = serial
CUDA_OUT = cuda_tpa cuda_bpa \
           cuda_tpa_modulo cuda_tpa_rangecheck \
           cuda_bpa_i cuda_bpa_nbor cuda_bpa_i_nbor
CL_OUT = cl
all: $(SERIAL_OUT) $(OUT) $(CUDA_OUT) $(CL_OUT)
INC += -I $(TOP)/common
LIB += -lgpunl

OBJS = hertz_clneighlist.o hertz_wrapper.o
OBJS += hertz_cudaneighlist.o hertz_cudawrapper.o

$(OUT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INC) $(SHARED) -o $(BUILD_DIR)/$@ $^ $(LIB)

$(SERIAL_OUT): serialdriver.cpp hertz_kernel.h
	$(CXX) $(CXXFLAGS) $(DRIVERFLAGS) $(INC) $(LIB) -o $(BUILD_DIR)/$@ $<

cl: driver.cpp $(OUT)
	$(CXX) $(DRIVERFLAGS) $(CXXFLAGS) $(INC) -I. -o $(BUILD_DIR)/$@ $< $(LIB) -L$(BUILD_DIR) -lhertz -lrt

cuda_tpa: cudadriver.cpp $(OUT)
	$(CXX) -DKERNEL=TPA $(DRIVERFLAGS) $(CXXFLAGS) $(INC) -I. -o $(BUILD_DIR)/$@ $< $(LIB) -L$(BUILD_DIR) -lhertz -lrt

cuda_bpa: cudadriver.cpp $(OUT)
	$(CXX) -DKERNEL=BPA $(DRIVERFLAGS) $(CXXFLAGS) $(INC) -I. -o $(BUILD_DIR)/$@ $< $(LIB) -L$(BUILD_DIR) -lhertz -lrt

cuda_tpa_modulo: cudadriver.cpp $(OUT)
	$(CXX) -DSTAGE_PARTICLE_I_DATA              -DKERNEL=TPA $(DRIVERFLAGS) $(CXXFLAGS) $(INC) -I. -o $(BUILD_DIR)/$@ $< $(LIB) -L$(BUILD_DIR) -lhertz -lrt

cuda_tpa_rangecheck: cudadriver.cpp $(OUT)
	$(CXX) -DSTAGE_PARTICLE_I_DATA -DRANGECHECK -DKERNEL=TPA $(DRIVERFLAGS) $(CXXFLAGS) $(INC) -I. -o $(BUILD_DIR)/$@ $< $(LIB) -L$(BUILD_DIR) -lhertz -lrt

cuda_bpa_i: cudadriver.cpp $(OUT)
	$(CXX) -DSTAGE_PARTICLE_I_DATA                       -DKERNEL=BPA $(DRIVERFLAGS) $(CXXFLAGS) $(INC) -I. -o $(BUILD_DIR)/$@ $< $(LIB) -L$(BUILD_DIR) -lhertz -lrt

cuda_bpa_nbor: cudadriver.cpp $(OUT)
	$(CXX)                         -DSTAGE_NEIGHBOR_DATA -DKERNEL=BPA $(DRIVERFLAGS) $(CXXFLAGS) $(INC) -I. -o $(BUILD_DIR)/$@ $< $(LIB) -L$(BUILD_DIR) -lhertz -lrt

cuda_bpa_i_nbor: cudadriver.cpp $(OUT)
	$(CXX) -DSTAGE_PARTICLE_I_DATA -DSTAGE_NEIGHBOR_DATA -DKERNEL=BPA $(DRIVERFLAGS) $(CXXFLAGS) $(INC) -I. -o $(BUILD_DIR)/$@ $< $(LIB) -L$(BUILD_DIR) -lhertz -lrt

hertz_kernel.h hertz_cudaneighlist.cu hertz_cudawrapper.cu hertz_clneighlist.cpp hertz_wrapper.cpp:
	../../pairgen.py hertz.yml
	cp reference.hertz_kernel.h new.hertz_kernel.h
	mv -f new.hertz_kernel.h hertz_kernel.h

.PHONY: clean veryclean
clean:
	rm -f *.o $(SERIAL_OUT) $(OUT) $(CUDA_OUT) $(CL_OUT)
veryclean: clean
	rm -f hertz_*
