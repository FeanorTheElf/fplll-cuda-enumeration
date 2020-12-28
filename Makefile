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

# Builds an executable that runs the basic tests
rls:
	$(NVCC) -ccbin $(CXX) $(NVCC_FLAGS) $(RLS_FLAGS) $(TEST_SRC) -o cudaenum_test

# Same as rls, but in debug mode, i.e. unoptimized and with debug information
dbg:
	$(NVCC) -ccbin $(CXX) $(NVCC_FLAGS) $(DBG_FLAGS) $(TEST_SRC) -o cudaenum_test

# Builds an executable that runs the basic tests on a single cpu thread; Therefore, these tests are not
# comprehensive, but can ensure that at least the basic logic works, even if no gpu is present
cpudbg:
	$(NVCC) -ccbin $(CXX) $(NVCC_FLAGS) $(DBG_FLAGS) -D TEST_CPU_ONLY $(TEST_SRC) -o cudaenum_test

# Builds an executable that runs the basic and one greater tests, suitable as benchmark
perf:
	$(NVCC) -ccbin $(CXX) $(NVCC_FLAGS) $(RLS_FLAGS) -D PERF_TEST $(TEST_SRC) -o cudaenum_test

# Builds and executes the tests
test: dbg
	./cudaenum_test

# Builds and executes the tests on a single cpu thread - see cpudbg
cputest: cpudbg
	./cudaenum_test

# Builds the library
lib:
	$(NVCC) -ccbin $(CXX) --compiler-options -fPIC --shared $(NVCC_FLAGS) $(RLS_FLAGS) $(SRC) -o libcudaenum.so

# Builds the library in debug mode, i.e. unoptimized and with debug information
dbglib:
	$(NVCC) -ccbin $(CXX) --compiler-options -fPIC --shared $(NVCC_FLAGS) $(DBG_FLAGS) $(SRC) -o libcudaenum.so
