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

LINK = $(LIBTOOL) --mode=link $(CC) -o $@

.cu.o:
	$(NVCC) $(CUDA_CXXFLAGS) $(AM_CUDA_CXXFLAGS) -o $@ -c $<

.cu.lo: 
	python $(top_srcdir)/cudalt.py "$(LIBTOOL)" $@ "$(NVCC)" $(CUDA_CXXFLAGS) $(AM_CUDA_CXXFLAGS) --compiler-options="$(CXXFLAGS) $(AM_CXXFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(CPPFLAGS) $(AM_CPPFLAGS)" -c $<