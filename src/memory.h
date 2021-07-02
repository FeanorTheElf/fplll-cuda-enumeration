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

#ifndef FPLLL_CUDA_MEMORY_H
#define FPLLL_CUDA_MEMORY_H

/**
 * This file contains typedefs for unique pointers to the most
 * often needed cuda resources
 */

#include "cuda_runtime.h"
#include <memory>
#include "cuda_check.h"

template <typename T>
struct CudaDeleter
{
	inline void operator()(T *ptr) const { check(cudaFree(ptr)); }
};

template <typename T>
struct PinnedDeleter
{
	inline void operator()(T *ptr) const { check(cudaFreeHost(ptr)); }
};

struct EventDeleter
{
	inline void operator()(cudaEvent_t ptr) const { check(cudaEventDestroy(ptr)); }
};

struct StreamDeleter
{
	inline void operator()(cudaStream_t ptr) const { check(cudaStreamDestroy(ptr)); }
};

template <typename T>
using CudaPtr = std::unique_ptr<T, CudaDeleter<T>>;

template <typename T>
using PinnedPtr = std::unique_ptr<T, PinnedDeleter<T>>;

using CudaEvent = std::unique_ptr<std::remove_pointer<cudaEvent_t>::type, EventDeleter>;

using CudaStream = std::unique_ptr<std::remove_pointer<cudaStream_t>::type, StreamDeleter>;

template <typename T>
inline CudaPtr<T> alloc_device_memory(size_t len, const char *file, const unsigned int line)
{
	T *result = nullptr;
	checkCudaError(cudaMalloc(&result, len * sizeof(T)), file, line);
	return CudaPtr<T>(result);
}

template <typename T>
inline PinnedPtr<T> alloc_pinned_memory(size_t len)
{
	T *result = nullptr;
	checkCudaError(cudaMallocHost(&result, len * sizeof(T)), __FILE__, __LINE__);
	return PinnedPtr<T>(result);
}

#define cuda_alloc(type, len) ::alloc_device_memory<type>(len, __FILE__, __LINE__)

#endif