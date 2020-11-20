ifeq (,$(NVCC))
ifneq (,$(wildcard /usr/local/cuda/bin/nvcc))
NVCC=/usr/local/cuda/bin/nvcc
endif
endif

NVCC?=nvcc
CXX?=g++

all:
	$(NVCC) -ccbin $(CXX) -D NDEBUG -O3 src/test.cpp src/testdata.cpp src/cuda_wrapper.cu
