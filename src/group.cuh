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

#ifndef FPLLL_PREFIX_CUH
#define FPLLL_PREFIX_CUH

/**
 * This file contains functions for prefix counting on different cooperation
 * levels, which is required for stream compactification
 */

#include "cuda_runtime.h"
#include "cooperative_groups.h"
#include <limits>
#include <assert.h>

constexpr inline unsigned int int_log2_rec(unsigned int x) {
  return x == 1 ? 0 : int_log2_rec(x >> 1) + 1;
}

constexpr inline unsigned int int_log2(unsigned int x) {
  return x == 0 ? std::numeric_limits<unsigned int>::max() : int_log2_rec(x);
}

static_assert(int_log2(0) == std::numeric_limits<unsigned int>::max(), "error in log2");
static_assert(int_log2(1) == 0, "error in log2");
static_assert(int_log2(7) == 2, "error in log2");
static_assert(int_log2(8) == 3, "error in log2");

template<typename CG, unsigned int block_size>
class PrefixCounter;

template<unsigned int block_size>
class PrefixCounter<cooperative_groups::thread_block, block_size> {

    unsigned char* shared_mem;

    /**
    Performs reduction from 2^(level + 1) sums to 2^level sums, and updates the cell_offsets accordingly.
    */
    template<unsigned int level>
    __device__ inline void prefix_sum_reduction_step(cooperative_groups::thread_block& group, unsigned int tid, unsigned int* accumulator, unsigned int* cell_offsets) {
        if (block_size_log >= level + 1) {
            unsigned int buffer;
            if (tid < (1 << level)) {
                buffer = accumulator[2 * tid] + accumulator[2 * tid + 1];
            }
            if ((tid >> (block_size_log - level - 1)) & 1) {
                cell_offsets[tid] += accumulator[(tid >> (block_size_log - level - 1)) - 1];
            }
            group.sync();
            if (tid < (1 << level)) {
                accumulator[tid] = buffer;
            }
            group.sync();
        }
    }

    template<unsigned int level>
    __device__ inline void prefix_count_reduction_step(cooperative_groups::thread_block& group, unsigned int tid, unsigned int* accumulator, unsigned int* cell_offsets) {
        constexpr unsigned int warpCountLog = block_size_log - 5;
        if (warpCountLog >= level + 1) {
            prefix_sum_reduction_step<level>(group, tid, accumulator, cell_offsets);
        }
    }

public:

    static_assert(block_size == (1L << int_log2(block_size)), "Expected BlockSize to be a power of 2");
    constexpr static unsigned int block_size_log = int_log2(block_size);

    __device__ inline PrefixCounter(unsigned char *shared_mem) : shared_mem(shared_mem) {}

    constexpr static unsigned int shared_mem_size_in_bytes = (block_size + block_size / 32) * sizeof(unsigned int);

    __device__ inline unsigned int prefix_count(cooperative_groups::thread_block &group,
                                                bool predicate, unsigned int &total_len)
    {
        assert(blockDim.x == block_size);
        assert(blockDim.y == 1 && blockDim.z == 1);

        unsigned int* cell_offsets = reinterpret_cast<unsigned int*>(shared_mem);
        unsigned int* accumulator = &reinterpret_cast<unsigned int*>(shared_mem)[block_size];

        const unsigned int warpid = threadIdx.x / 32;
        const unsigned int laneid = threadIdx.x % 32;
        const unsigned int in_warp_values = __ballot_sync(0xFFFFFFFF, predicate ? 1 : 0);
        const unsigned int in_warp_accumulation = __popc(in_warp_values);
        const unsigned int in_warp_prefix_count = __popc(in_warp_values << (32 - laneid));

        cell_offsets[threadIdx.x] = in_warp_prefix_count;
        if (laneid == 0) {
            accumulator[warpid] = in_warp_accumulation;
        }
        group.sync();

        const unsigned int tid = threadIdx.x;

        // now perform standard reduction with the remaining 1024/32 == 32 values
        prefix_count_reduction_step<4>(group, tid, accumulator, cell_offsets);
        prefix_count_reduction_step<3>(group, tid, accumulator, cell_offsets);
        prefix_count_reduction_step<2>(group, tid, accumulator, cell_offsets);
        prefix_count_reduction_step<1>(group, tid, accumulator, cell_offsets);
        prefix_count_reduction_step<0>(group, tid, accumulator, cell_offsets);

        group.sync();

        total_len = accumulator[0];
        return cell_offsets[threadIdx.x];
    }
};

template <unsigned int block_size, typename PG> class PrefixCounter<cooperative_groups::thread_block_tile<32, PG>, block_size>
{

public:
  static_assert(block_size == (1L << int_log2(block_size)),
                "Expected BlockSize to be a power of 2");
  constexpr static unsigned int block_size_log = int_log2(block_size);

  __device__ inline PrefixCounter() {}

  constexpr static unsigned int shared_mem_size_in_bytes = 0;

  __device__ inline unsigned int prefix_count(cooperative_groups::thread_block_tile<32, PG> &group,
                                              bool predicate, unsigned int &total_len)
  {
    assert(blockDim.x == block_size);
    assert(blockDim.y == 1 && blockDim.z == 1);

    const unsigned int laneid               = threadIdx.x % 32;
    const unsigned int in_warp_values       = __ballot_sync(0xFFFFFFFF, predicate ? 1 : 0);
    const unsigned int in_warp_accumulation = __popc(in_warp_values);
    const unsigned int in_warp_prefix_count = __popc(in_warp_values << (32 - laneid));

    total_len = in_warp_accumulation;
    return in_warp_prefix_count;
  }
};

class ThreadGroupSingleThread {

public:

    __device__ __host__ inline void sync() {}

    __device__ __host__ inline unsigned int thread_rank() {
        return 0;
    }

    __device__ __host__ inline unsigned int size() {
        return 1;
    }

    __device__ __host__ inline unsigned int group_index() {
        return 0;
    }

    __device__ inline unsigned int group_index_in_block() {
        return 0;
    }

    __device__ __host__ constexpr inline unsigned int group_count(unsigned int block_count) {
        return 1;
    }

    __device__ __host__ inline unsigned int prefix_count(bool predicate, unsigned int& total_len)
    {
        total_len = predicate ? 1 : 0;
        return 0;
    }

    __device__ inline bool all(bool predicate)
    {
        return predicate;
    }

    template<typename T>
    __device__ inline T shuffle(T value, unsigned int src)
    {
        return value;
    }
};

template<unsigned int block_size>
class ThreadGroupWarp {

    cooperative_groups::thread_block_tile<32, cooperative_groups::thread_block> cg;
    PrefixCounter<cooperative_groups::thread_block_tile<32, cooperative_groups::thread_block>, block_size> prefix_counter;

public:

    __device__ ThreadGroupWarp(cooperative_groups::thread_block block_group) : cg(cooperative_groups::tiled_partition<32>(block_group)), prefix_counter() {}

    __device__ inline void sync() {
        cg.sync();
    }

    __device__ inline unsigned int thread_rank() {
        return cg.thread_rank();
    }

    __device__ __host__ constexpr inline unsigned int size() {
#ifdef __CUDA_ARCH__
        assert(cg.size() == 32);
#endif
        return 32;
    }

    __device__ inline unsigned int group_index() {
        return threadIdx.x / 32 + blockIdx.x * (blockDim.x / 32);
    }

    __device__ inline unsigned int group_index_in_block() {
        return threadIdx.x / 32;
    }

    __device__ __host__ constexpr inline unsigned int group_count(unsigned int block_count) {
        return block_count * block_size / size();
    }

    __device__ inline unsigned int prefix_count(bool predicate, unsigned int& total_len)
    {
        return prefix_counter.prefix_count(cg, predicate, total_len);
    }

    __device__ inline bool all(bool predicate)
    {
        return cg.all(predicate);
    }

    template<typename T>
    __device__ inline T shuffle(T value, unsigned int src)
    {
        return cg.shfl(value, src);
    }
};

#endif