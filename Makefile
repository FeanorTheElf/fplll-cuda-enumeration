ifeq (,$(NVCC))
ifneq (,$(wildcard /usr/local/cuda/bin/nvcc))
NVCC=/usr/local/cuda/bin/nvcc
endif
endif

NVCC?=nvcc
CXX?=g++

all:
	$(NVCC) -ccbin $(CXX) -O3 -D NDEBUG src/test.cpp src/testdata.cpp src/cuda_wrapper.cu

dbg:
	$(NVCC) -ccbin $(CXX) -g -G -O0 -D DEBUG src/test.cpp src/testdata.cpp src/cuda_wrapper.cu

lib:
	$(NVCC) -ccbin $(CXX) --compiler-options -fPIC --shared -D NDEBUG -O3 src/cuda_wrapper.cu -o libcudaenum.so