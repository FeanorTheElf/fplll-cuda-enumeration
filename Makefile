ifeq (,$(NVCC))
ifneq (,$(wildcard /usr/local/cuda/bin/nvcc))
NVCC=/usr/local/cuda/bin/nvcc
endif
endif

NVCC?=nvcc
CXX?=g++

DBG_FLAGS=-g -G -O0 -D DEBUG
RLS_FLAGS=-O3 -D NDEBUG
SRC=src/cuda_wrapper.cu
TEST_SRC=$(SRC) src/test.cpp src/testdata.cpp

rls:
	$(NVCC) -ccbin $(CXX) $(RLS_FLAGS) $(TEST_SRC) -o cudaenum_test

dbg:
	$(NVCC) -ccbin $(CXX) $(DBG_FLAGS) $(TEST_SRC) -o cudaenum_test

perf:
	$(NVCC) -ccbin $(CXX) $(RLS_FLAGS) -D PERF_TEST $(TEST_SRC) -o cudaenum_test

test: dbg
	./cudaenum_test

lib:
	$(NVCC) -ccbin $(CXX) --compiler-options -fPIC --shared $(RLS_FLAGS) $(SRC) -o libcudaenum.so

dbg_lib:
	$(NVCC) -ccbin $(CXX) --compiler-options -fPIC --shared $(DBG_FLAGS) $(SRC) -o libcudaenum.so