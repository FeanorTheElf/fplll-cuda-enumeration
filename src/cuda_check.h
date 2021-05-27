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

#ifndef FPLLL_CUDA_CHECK_H
#define FPLLL_CUDA_CHECK_H

#include "cuda_runtime.h"
#include <iostream>

struct CudaError
{
	cudaError_t status;

	constexpr inline CudaError(cudaError_t status) : status(status) {}
};

inline void checkCudaError(cudaError_t status, const char *file, const unsigned int line)
{
	if (status != cudaSuccess)
	{
		std::cerr << "Error: " << cudaGetErrorString(status) << " at " << file << ":" << line
				  << std::endl;
		throw CudaError(status);
	}
}

#define check(x) checkCudaError(x, __FILE__, __LINE__)

#endif
