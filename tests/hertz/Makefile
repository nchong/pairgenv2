include ../../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))

# GLOBAL BUILD OPTIONS
OUT=libhertz.so
all: $(OUT) driver_serial_timer driver_posix_timer
LIB += -lgpunl

OBJS = hertz_clneighlist.o	hertz_wrapper.o hertz_cudaneighlist.o

$(OUT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) $(SHARED) -o $@ $^

driver_serial_timer: driver.cpp
	$(CXX) $(CXXFLAGS) $(INC) -I. -o $@ $^ $(LIB) -L. -lhertz

driver_posix_timer: driver.cpp
	$(CXX) -DPOSIX_TIMER $(CXXFLAGS) $(INC) -I. -o $@ $^ $(LIB) -L. -lhertz -lrt

hertz_cudaneighlist.cu hertz_clneighlist.cpp hertz_wrapper.cpp:
	../../pairgen.py hertz.yml
	cp reference.hertz_kernel.h new.hertz_kernel.h
	mv -f new.hertz_kernel.h hertz_kernel.h

.PHONY: clean
clean:
	rm -f hertz_* driver_*_timer $(OUT)
