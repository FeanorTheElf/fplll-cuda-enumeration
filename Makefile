
all:
	nvcc -D NDEBUG -O3 src/test.cpp src/testdata.cpp src/cuda_wrapper.cu