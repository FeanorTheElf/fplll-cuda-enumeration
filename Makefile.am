# these are custom variables, but are named similar to the corresponding automake
# variables, as they similarly contain user-provided resp. configured flags to the compiler
AM_CUDA_CXXFLAGS = --std=c++11
AM_CUDA_LIBS = -lcudart

lib_LTLIBRARIES = libcudaenum.la
libcudaenum_la_SOURCES = src/cuda_wrapper.cu
libcudaenum_la_LIBADD = $(CUDA_LDFLAGS) $(AM_CUDA_LIBS)

bin_PROGRAMS = cudaenum_test
cudaenum_test_SOURCES = src/test.cpp src/testdata.cpp
cudaenum_test_LDADD = libcudaenum.la

# libtools does not support nvcc. I have thought of the following workarounds
# - do not use libtools
# - use libtools and tell it to use nvcc as a C++ compiler
# - use a custom libtool-like script
#
# The first idea seems to be not recommended, it might void some portability advantages
# and it would also be inconsistent with fplll.
# The second approach has an issue with PIC code generation, as nvcc does not accept the
# PIC flags libtools might pass to it, but without those PIC flags, the libtool descriptor
# (the *.lo file) is unusable.
# Therefore, we settle for the last option, following a guide that can be found at
# www.clusterchimps.org/autotools.php. This is mainly what the python file `cudalt.py`
# is used for. This is triggered by the following rules:

LINK = $(LIBTOOL) --mode=link $(CC) -o $@

.cu.o:
	$(NVCC) $(CUDA_CXXFLAGS) $(AM_CUDA_CXXFLAGS) -o $@ -c $<

.cu.lo: 
	python $(top_srcdir)/cudalt.py "$(LIBTOOL)" $@ "$(NVCC)" $(CUDA_CXXFLAGS) $(AM_CUDA_CXXFLAGS) --compiler-options="$(CXXFLAGS) $(AM_CXXFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(CPPFLAGS) $(AM_CPPFLAGS)" -c $<