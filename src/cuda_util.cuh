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

#ifndef FPLLL_CUDA_UTIL_CUH
#define FPLLL_CUDA_UTIL_CUH

#include "cuda_runtime.h"
#include <iostream>
#include <memory>
#include <ctime>
#include "memory.h"

__device__ __host__ inline unsigned int thread_id()
{
#ifdef __CUDA_ARCH__
  return threadIdx.x + blockIdx.x * blockDim.x;
#else
  return 0;
#endif
}

__device__ __host__ inline unsigned int thread_id_in_block()
{
#ifdef __CUDA_ARCH__
  return threadIdx.x;
#else
  return 0;
#endif
}

__device__ __host__ inline unsigned long long time()
{
#ifdef __CUDA_ARCH__
  return clock64();
#else
  return clock();
#endif
}

__device__ __host__ inline void runtime_error() {
#ifdef __CUDA_ARCH__
  __trap();
#else
  throw;
#endif
}

template<typename... Args>
struct FnWrapper {
  std::function<void(Args...)> fn;

  FnWrapper(std::function<void(Args...)> fn) : fn(fn) {}

  __device__ __host__ void operator()(Args... args) {
#ifdef __CUDA_ARCH__
    printf("Fatal bug");
    runtime_error();
#else
    fn(args...);
#endif
  }
};

#endif