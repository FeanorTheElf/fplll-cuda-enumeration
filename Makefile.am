AM_CUDA_CFLAGS = --std=c++11
AM_CUDA_LIBS = -lcudart

lib_LTLIBRARIES = libcudaenum.la
libcudaenum_la_SOURCES = src/cuda_wrapper.cu
libcudaenum_la_LIBADD = $(CUDA_LDFLAGS) $(AM_CUDA_LIBS)

bin_PROGRAMS = cudaenum_test
cudaenum_test_SOURCES = src/test.cpp src/testdata.cpp
cudaenum_test_LDADD = libcudaenum.la

LINK = $(LIBTOOL) --mode=link $(CC) -o $@

.cu.o:
	$(NVCC) $(CUDA_CFLAGS) $(AM_CUDA_CFLAGS) -o $@ -c $<

.cu.lo: 
	python $(top_srcdir)/cudalt.py "$(LIBTOOL)" $@ "$(NVCC)" $(CUDA_CFLAGS) $(AM_CUDA_CFLAGS) --compiler-options="$(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS)" -c $<