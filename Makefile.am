lib_LTLIBRARIES = libcudaenum.la
libcudaenum_la_SOURCES = src/cuda_wrapper.cu

LINK = $(LIBTOOL) --mode=link $(CC) -o $@ $(CUDA_LDFLAGS) $(CUDA_LIBS)
NVCC_FLAGS = -D NDEBUG --std=c++11

.cu.o:
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

.cu.lo: 
	$(top_srcdir)/cudalt.py "$(LIBTOOL)" $@ "$(NVCC)" $(NVCC_FLAGS) --compiler-options="$(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS)" -c $<