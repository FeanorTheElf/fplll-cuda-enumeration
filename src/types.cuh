/*
   (C) 2020 Simon Pohmann.
   This file is part of fplll. fplll is free software: you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation,
   either version 2.1 of the License, or (at your option) any later version.
   fplll is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public License
   along with fplll. If not, see <http://www.gnu.org/licenses/>. */

#ifndef FPLLL_TYPES_CUH
#define FPLLL_TYPES_CUH

/**
 * Defines the basic types used during the enumeration
 */

#include "cuda_runtime.h"
#include "atomic.h"

typedef double enumf;
typedef double enumi;

struct Matrix
{
  const enumf *ptr;
  unsigned int ld;

  __device__ __host__ inline Matrix(const enumf *ptr, unsigned int ld) : ptr(ptr), ld(ld) {}

  __device__ __host__ inline Matrix() : ptr(nullptr), ld(0) {}

  __device__ __host__ inline enumf at(unsigned int row, unsigned int col) const
  {
    return ptr[row * ld + col];
  }

  __device__ __host__ inline Matrix block(unsigned int start_row, unsigned int start_col) const
  {
    return Matrix(&ptr[start_col + start_row * ld], ld);
  }
};

template <unsigned int levels, unsigned int dimensions_per_level, unsigned int max_nodes_per_level>
struct SubtreeEnumerationBuffer;

struct PerfCounter
{
  uint64_t *counter;

  __device__ __host__ inline PerfCounter(uint64_t *target) : counter(target) {}

  __device__ __host__ inline void inc(unsigned int level) { aggregated_atomic_inc(&counter[level]); }

  __device__ __host__ inline PerfCounter offset_level(unsigned int start_level) {
    return PerfCounter(&counter[start_level]);
  }
};

__device__ unsigned long long perf[1] = { 0 };

__device__ __host__ inline unsigned long long from() {
#ifdef __CUDA_ARCH__
    return clock64();
#else
    return 0;
#endif
}

__device__ __host__ inline void to(unsigned long long x) {
#ifdef __CUDA_ARCH__
    atomicAdd(&perf[0], clock64() - x);
#endif
}

__host__ inline void reset_profiling_counter() {
    const unsigned long long zero = 0;
    cudaMemcpyToSymbol(perf, &zero, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
}

__host__ inline void print_profiling_counter() {
    unsigned long long hperf;
    cudaMemcpyFromSymbol(&hperf, perf, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
    std::cout << "profiling counter " << (hperf / 1e9) << " Gcycles" << std::endl;
}

#define FROM(x) unsigned long long x = from();
#define TO(x) to(x);

#endif