include ../../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))

# GLOBAL BUILD OPTIONS
OUT=libhertz.so
all: $(OUT) cl_cpu_timer cuda_cpu_timer cl_posix_timer cuda_posix_timer
LIB += -lgpunl

OBJS = hertz_clneighlist.o hertz_wrapper.o
OBJS += hertz_cudaneighlist.o hertz_cudawrapper.o

$(OUT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) $(SHARED) -o $@ $^

cl_cpu_timer: driver.cpp
	$(CXX) $(CXXFLAGS) $(INC) -I. -o $@ $^ $(LIB) -L. -lhertz

cuda_cpu_timer: cudadriver.cpp
	$(CXX) $(CXXFLAGS) $(INC) -I. -o $@ $^ $(LIB) -L. -lhertz

cl_posix_timer: driver.cpp
	$(CXX) -DPOSIX_TIMER $(CXXFLAGS) $(INC) -I. -o $@ $^ $(LIB) -L. -lhertz -lrt

cuda_posix_timer: cudadriver.cpp
	$(CXX) -DPOSIX_TIMER $(CXXFLAGS) $(INC) -I. -o $@ $^ $(LIB) -L. -lhertz -lrt

hertz_cudaneighlist.cu hertz_clneighlist.cpp hertz_wrapper.cpp hertz_cudawrapper.cu:
	../../pairgen.py hertz.yml
	cp reference.hertz_kernel.h new.hertz_kernel.h
	mv -f new.hertz_kernel.h hertz_kernel.h

.PHONY: clean
clean:
	rm -f hertz_* cl_cpu_timer cuda_cpu_timer cl_posix_timer cuda_posix_timer $(OUT)
