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

typedef std::function<float(enumf, enumi*)> process_sol_fn;

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

__device__ uint64_t perf[2] = { 0 };

__device__ __host__ inline uint64_t from() {
#ifdef __CUDA_ARCH__
    return clock64();
#else
    return 0;
#endif
}

__device__ __host__ inline void to(uint64_t x) {
#ifdef __CUDA_ARCH__
    atomic_add(&perf[0], clock64() - x);
#endif
}

__host__ inline void reset_profiling_counter() {
    const uint64_t zero[2] = { 0, 0 };
    cudaMemcpyToSymbol(perf, &zero, 2 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

__host__ inline void print_profiling_counter() {
    uint64_t hperf[2];
    cudaMemcpyFromSymbol(&hperf, perf, 2 * sizeof(uint64_t), 0, cudaMemcpyDeviceToHost);
    std::cout << "profiling counter " << (hperf[0] / 1e9) << " Gcycles, " << (hperf[1] / 1e9) << " Gcycles" << std::endl;
}

#define FROM(x) uint64_t x = from();
#define TO(x) to(x);

__device__ __host__ inline void profile_active_thread_percentage() {
#ifdef __CUDA_ARCH__
    aggregated_atomic_inc(&perf[0]);
    if (cooperative_groups::coalesced_threads().thread_rank() == 0) {
        atomic_add(&perf[1], 32);
    }
#endif
}

#endif