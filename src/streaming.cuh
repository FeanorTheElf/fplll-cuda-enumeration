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

#ifndef FPLLL_STREAMING_CUH
#define FPLLL_STREAMING_CUH

/**
 * This file defines the queue functionality, encapsulated by two endpoints, that is necessary to 
 * stream solution points from the device to the host, to be processed by the evaluator
 */

#include "cooperative_groups.h"
#include "cuda_runtime.h"
#include <assert.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>

#include "atomic.h"
#include "memory.h"
#include "cuda_check.h"
#include "cuda_util.cuh"
#include "types.cuh"

/**
 * Evaluator proxy that streams all found points to the host, via a circular buffer
 */
template <unsigned int buffer_size>
class PointStreamEvaluator
{
	// these are accessible by the host
	enumi *result_buffer;
	enumf *result_squared_norms;
	unsigned int *has_written_round;

	// these are internal for collecting the points
	unsigned int *write_to_index;
	unsigned int point_index;

	__device__ __host__ constexpr static unsigned int used_memory_size_in_bytes(unsigned int point_dimension)
	{
		return sizeof(enumi) * buffer_size * point_dimension + (sizeof(enumf) + sizeof(unsigned int)) * buffer_size;
	}

public:
	__device__ __host__ inline PointStreamEvaluator(unsigned char *memory, unsigned int *write_to_index)
		: result_squared_norms(reinterpret_cast<enumf *>(memory)),
		  has_written_round(reinterpret_cast<unsigned int *>(memory + sizeof(enumf) * buffer_size)),
		  result_buffer(reinterpret_cast<enumi *>(memory + (sizeof(enumf) + sizeof(unsigned int)) * buffer_size)),
		  write_to_index(write_to_index)
	{
	}

	__device__ __host__ constexpr static unsigned int memory_size_in_bytes(unsigned int point_dimension)
	{
		// ensure alignment
		return ((used_memory_size_in_bytes(point_dimension) - 1) / sizeof(enumf) + 1) * sizeof(enumf);
	}

	__device__ __host__ inline void operator()(enumi x, unsigned int coordinate, enumf norm_square,
											   unsigned int point_dimension)
	{
		if (coordinate == 0)
		{
			point_index = aggregated_atomic_inc(write_to_index) % buffer_size;
			result_squared_norms[point_index] = norm_square;
		}

		result_buffer[point_index * point_dimension + coordinate] = x;

		if (coordinate == point_dimension - 1)
		{
			threadfence_system();
			atomic_add(&has_written_round[point_index], 1);
		}
	}
};

template <unsigned int buffer_size>
class PointStreamEndpoint
{

	void *device_memory;
	unsigned int evaluator_count;
	unsigned int point_dimension;
	PinnedPtr<unsigned char> host_memory;
	PinnedPtr<enumf> host_enumeration_bounds;
	enumf *device_enumeration_bounds;
	const enumf *relative_enumeration_bounds;
	enumf global_enumeration_radius_squared;
	PinnedPtr<uint64_t> host_searched_nodes;
	const uint64_t *device_searched_nodes;
	std::vector<std::unique_ptr<unsigned int[]>> last_has_written_round;
	CudaStream stream;

	inline void *evaluator_memory(unsigned int evaluator_id)
	{
		return static_cast<void *>(host_memory.get() + evaluator_id * PointStreamEvaluator<buffer_size>::memory_size_in_bytes(point_dimension));
	}

	inline unsigned int *host_has_written_round(unsigned int evaluator_id)
	{
		return reinterpret_cast<unsigned int *>(static_cast<unsigned char *>(evaluator_memory(evaluator_id)) + sizeof(enumf) * buffer_size);
	}

	inline unsigned int *device_has_written_round(unsigned int evaluator_id)
	{
		unsigned char *device_evaluator_memory = static_cast<unsigned char *>(device_memory) + PointStreamEvaluator<buffer_size>::memory_size_in_bytes(point_dimension) * evaluator_id;
		return reinterpret_cast<unsigned int *>(device_evaluator_memory + sizeof(enumf) * buffer_size);
	}

	inline void update_enumeration_bounds()
	{
		for (unsigned int i = 0; i < point_dimension; ++i)
		{
			host_enumeration_bounds.get()[i] = relative_enumeration_bounds[i] * global_enumeration_radius_squared;
		}
		check(cudaMemcpyAsync(device_enumeration_bounds, host_enumeration_bounds.get(), point_dimension * sizeof(enumf), cudaMemcpyHostToDevice, stream.get()));
	}

public:
	/**
   * Initializes the streaming endpoint on the host side to be able to collect points from multiple device side endpoints and
   * update the enumeration bounds when points are found.
   *
   * @param device_memory - contigous segment on the device containing the batches of memory corresponding to each evaluator
   * @param device_enumeration_bounds - memory on the device containing the n absolute enumeration bounds for each level, will be updated when points are found
   * @param relative_enumeration_bounds - memory on the host containing the n enumeration bounds relative to the global enumeration bound
   * @param initial_radius_squared - the initial global enumeration bound, this can be decreased whenever new points are found
   */
	inline PointStreamEndpoint(unsigned char *device_memory, enumf *device_enumeration_bounds, const uint64_t *device_searched_nodes, const enumf *relative_enumeration_bounds, enumf initial_radius_squared, unsigned int evaluator_count, unsigned int point_dimension)
		: evaluator_count(evaluator_count),
		  point_dimension(point_dimension),
		  device_memory(device_memory),
		  host_memory(alloc_pinned_memory<unsigned char>(evaluator_count * PointStreamEvaluator<buffer_size>::memory_size_in_bytes(point_dimension))),
		  host_enumeration_bounds(alloc_pinned_memory<enumf>(point_dimension)),
		  device_enumeration_bounds(device_enumeration_bounds),
		  relative_enumeration_bounds(relative_enumeration_bounds),
		  global_enumeration_radius_squared(initial_radius_squared),
		  device_searched_nodes(device_searched_nodes),
		  host_searched_nodes(alloc_pinned_memory<uint64_t>(point_dimension))
	{
		const unsigned int used_device = 0;
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, used_device);
		if (deviceProp.asyncEngineCount <= 1)
		{
			throw "your cuda device does not support asynchronous kernel execution and data transfer, which is required for solution point streaming";
		}
		cudaStream_t raw_stream;
		check(cudaStreamCreate(&raw_stream));
		stream.reset(raw_stream);
		for (unsigned int i = 0; i < evaluator_count; ++i)
		{
			last_has_written_round.push_back(std::unique_ptr<unsigned int[]>(new unsigned int[buffer_size]));
		}
	}

	/**
     * Initializes the memory on host and device
     */
	__host__ inline void init()
	{
		for (unsigned int i = 0; i < evaluator_count; ++i)
		{
			// use default cuda stream to prevent overlap with kernel execution & point query
			check(cudaMemset(device_has_written_round(i), 0, buffer_size * sizeof(unsigned int)));
			std::memset(host_has_written_round(i), 0, buffer_size * sizeof(unsigned int));
		}
		update_enumeration_bounds();
		check(cudaStreamSynchronize(stream.get()));
	}

	/**
     * Queries all points that have been found since the last time this function was called, and passes them
     * to the given callback. The enumeration bound returned by the callback is written back to the device.
     */
	template <typename Fn, bool print_status = true>
	__host__ inline void query_new_points(Fn callback)
	{
		for (unsigned int i = 0; i < evaluator_count; ++i)
		{
			std::memcpy(last_has_written_round[i].get(), host_has_written_round(i), buffer_size * sizeof(unsigned int));
		}
		const unsigned int total_size = PointStreamEvaluator<buffer_size>::memory_size_in_bytes(point_dimension) * evaluator_count;
		check(cudaMemcpyAsync(host_memory.get(), device_memory, total_size, cudaMemcpyDeviceToHost, stream.get()));
		check(cudaStreamSynchronize(stream.get()));

		unsigned int point_count = 0;
		for (unsigned int i = 0; i < evaluator_count; ++i)
		{
			for (unsigned int j = 0; j < buffer_size; ++j)
			{
				const unsigned int last_written_count = last_has_written_round[i][j];
				const unsigned int new_written_count = host_has_written_round(i)[j];
				if (last_written_count == new_written_count)
				{
					// no new data here in this buffer entry
				}
				else if (last_written_count + 1 == new_written_count)
				{
					++point_count;
					enumf norm_square = static_cast<enumf *>(evaluator_memory(i))[j];
					enumi *points = reinterpret_cast<enumi *>(static_cast<unsigned char *>(evaluator_memory(i)) + (sizeof(enumf) + sizeof(unsigned int)) * buffer_size);
					enumi *x = &points[j * point_dimension];
					enumf evaluator_enum_bound = callback(norm_square, x);
					global_enumeration_radius_squared = std::min<enumf>(evaluator_enum_bound, global_enumeration_radius_squared);
				}
				else
				{
					if (print_status)
					{
						std::cout << "Buffer not big enough, block " << i << " stored " << (new_written_count - last_written_count) << " points at index " << j << std::endl;
					}
					throw "buffer not big enough to hold all solution points found between two queries";
				}
			}
		}
		if (point_count > 0)
		{
			check(cudaStreamSynchronize(stream.get()));
			update_enumeration_bounds();
			if (print_status)
			{
				std::cout << "Got " << point_count << " new solution points and can decrease enum bound to " << global_enumeration_radius_squared << std::endl;
			}
			check(cudaStreamSynchronize(stream.get()));
		}
	}

	__host__ void print_currently_searched_nodes()
	{
		check(cudaMemcpyAsync(host_searched_nodes.get(), device_searched_nodes, point_dimension * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream.get()));
		check(cudaStreamSynchronize(stream.get()));
		std::cout << "Currently searched ";
		for (auto it = host_searched_nodes.get(); it != host_searched_nodes.get() + point_dimension; ++it) {
			std::cout << *it << ", ";
		}
		std::cout << std::endl;
	}

	__host__ inline void wait_for_event(cudaEvent_t event)
	{
		check(cudaStreamWaitEvent(stream.get(), event, 0));
	}
};

#endif