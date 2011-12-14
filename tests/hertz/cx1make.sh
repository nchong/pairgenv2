#!/bin/bash

# required tools
module load python/2.6.4
module load intel-suite/11.1
export CLDIR=/apps/opencl/ati-2.5
export CUDADIR=/apps/cuda/4.0.17

# turn off some annoying icc warnings
unset LD_PRELOAD

# common build options
export NVCCFLAGS='-Xptxas -dlcm=ca'
export CXX=icc
# disable warnings
export CXXFLAGS='-wd869,383,981,1572,444'
# 869: parameter <x> was never referenced
# http://origin-software.intel.com/en-us/articles/cdiag869/
# This is warning about typei and typej not being used in [hertz_kernel.h].
# The standard trick to comment out the parameter name like so:
#    int /* typei is unused */, int /* typej is unused */,
# causes the OpenCL compiler to die, so we'll have to put up with suppressing the warning here instead.

# 383: value copied to temporary, reference to temporary used
# http://origin-software.intel.com/en-us/articles/cdiag383/
# This is warning about pushing onto our vector of timers (one_time and per_iter).
# As the link says, using a temporary here is perfectly safe.

# 981: operands are evaluated in unspecified order
# http://origin-software.intel.com/en-us/articles/cdiag981/
# This warning fires a lot inside [framework.h] because we do things like:
#     min_element(v.begin(), v.end()); //v is a vector
# As the link says, this warning is more useful for C than C++

# 1572: floating-point equality and inequality comparisons are unreliable
# http://origin-software.intel.com/en-us/articles/cdiag1572/
# This warning fires for the fp [percentage_error] function in [framework.h]:
#     if (expected == computed) ...
#     if (expected == 0.0) ...

# 444: destructor for base class is not virtual
# http://origin-software.intel.com/en-us/articles/cdiag444/
# http://www.parashift.com/c++-faq-lite/virtual-functions.html#faq-20.7
# This fires for CLNeighList [common/clneighlist.cpp] which is derived by HertzCLNeighList [hertz_clneighlist.cpp].
# This is only an issue if someone will delete a HertzCLNeighList object using a pointer to CLNeighList.
# This shouldn't be a problem for us.

# cuda variants
make veryclean; rm -f cuda_*
make KERNELFLAGS='-DSTAGE_PARTICLE_I_DATA' cuda_tpa cuda_bpa
mv -f cuda_tpa cuda_tpa_modulo
mv -f cuda_bpa cuda_bpa_i

rm -f hertz_cudawrapper.o
make KERNELFLAGS='-DSTAGE_PARTICLE_I_DATA -DRANGECHECK' cuda_tpa
mv -f cuda_tpa cuda_tpa_rangecheck

rm -f hertz_cudawrapper.o
make KERNELFLAGS='                        -DSTAGE_NEIGHBOR_DATA' cuda_bpa
mv -f cuda_bpa cuda_bpa_nbor

rm -f hertz_cudawrapper.o
make KERNELFLAGS='-DSTAGE_PARTICLE_I_DATA -DSTAGE_NEIGHBOR_DATA' cuda_bpa
mv -f cuda_bpa cuda_bpa_i_nbor

# baseline implementations (serial, cl, cuda_tpa, cuda_bpa)
rm -f hertz_cudawrapper.o
make
