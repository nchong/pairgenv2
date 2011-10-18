# COMPILE OPTIONS
CXX = g++
LOG_LEVEL = LOG_WARN
override CXXFLAGS += -O2 -Wall -Wextra -Werror -pedantic -Wno-variadic-macros -fPIC

# COMMON INCLUDE
override INCLUDEDIR += -I ../../../inc -I ../../common
# COMMON LIBS
override LIB = -L../../../lib -lclwrapper -lscan -lgpunl -lcommon

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) -c -o $@ $<

OBJS = hertz_gpuneighlist.o	hertz_wrapper.o

driver: driver.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) -o $@ $^ $(LIB)

hertz_gpuneighlist.cpp hertz_wrapper.cpp:
	../../pairgen.py hertz.yml
	cp reference.hertz_pair_kernel.cl new.hertz_pair_kernel.cl
	mv -f new.hertz_pair_kernel.cl hertz_pair_kernel.cl

clean:
	rm -f hertz_* driver
