ifeq (,$(NVCC))
ifneq (,$(wildcard /usr/local/cuda/bin/nvcc))
NVCC=/usr/local/cuda/bin/nvcc
endif
endif

NVCC?=nvcc
CXX?=g++

NVCC_FLAGS=--std=c++11
DBG_FLAGS=-g -G -O0 -D NDEBUG
RLS_FLAGS=-O3 -D NDEBUG
SRC=src/cuda_wrapper.cu
TEST_SRC=$(SRC) src/test.cpp src/testdata.cpp

rls:
	$(NVCC) -ccbin $(CXX) $(NVCC_FLAGS) $(RLS_FLAGS) $(TEST_SRC) -o cudaenum_test

dbg:
	$(NVCC) -ccbin $(CXX) $(NVCC_FLAGS) $(DBG_FLAGS) $(TEST_SRC) -o cudaenum_test

perf:
	$(NVCC) -ccbin $(CXX) $(NVCC_FLAGS) $(RLS_FLAGS) -D PERF_TEST $(TEST_SRC) -o cudaenum_test

test: dbg
	./cudaenum_test

lib:
	$(NVCC) -ccbin $(CXX) --compiler-options -fPIC --shared $(NVCC_FLAGS) $(RLS_FLAGS) $(SRC) -o libcudaenum.so

dbglib:
	$(NVCC) -ccbin $(CXX) --compiler-options -fPIC --shared $(NVCC_FLAGS) $(DBG_FLAGS) $(SRC) -o libcudaenum.so
