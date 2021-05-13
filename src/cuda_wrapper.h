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

#ifndef FPLLL_CUDA_WRAPPER_H
#define FPLLL_CUDA_WRAPPER_H

/**
 * Adapter that wraps the heavily templated cuda enumeration functionality
 * and provides an easily usable interface, as well as one compatible with fplll.
 */
#include "api.h"
#include <memory>

namespace cuenum
{

typedef std::function<float(double, double*)> process_sol_fn;

struct CudaEnumOpts
{
  // maximal amount of paths that will be searched using recursive enumeration during each algorithm
  // step by each thread. When this is exceeded, the recursive enumeration state is stored and
  // resumed in the next step. Use for load balancing to prevent threads with small subtrees to wait
  // for threads with very big subtrees.
  unsigned int max_subtree_paths;
  // stop children generation when the percentage of parent points that have still unprocessed
  // children drops beneath this percentage
  float min_active_parents_percentage;
  // height of the subtrees that are searched using recursive enumeration.
  unsigned int dimensions_per_level;
  // how many start points should be assigned to each cooperative group
  unsigned int initial_nodes_per_group;
  // how many cuda threads to use for the search. If this is not a multiple of the block
  // size, it will be rounded up to one
  unsigned int thread_count;
};

constexpr CudaEnumOpts default_opts = {50, .5, 3, 8, 32 * 256};
extern CudaEnumOpts used_opts;

std::vector<uint64_t> search_enumeration(const double* mu, const double* rdiag,
    const unsigned int enum_dimensions,
    const double* start_point_coefficients, unsigned int start_point_count,
    unsigned int start_point_dim, const double* pruning, double initial_radius,
    process_sol_fn evaluator, CudaEnumOpts opts = default_opts);

/**
 * Allocates memory and fills it with the given start points, so that it is directly copyable to the device
 * memory space (and therefore a correct parameter for search_enumeration()). The start points are given
 * as an iterator over indexable objects, each containing the start_point_dim coefficients of one start point.
 * The memory is allocated in page-locked memory to improve copy efficiency, but the provided unique pointer 
 * will correctly free it.
 */
template <typename InputIt>
inline std::unique_ptr<double[]>
create_start_point_array(size_t start_point_count, size_t start_point_dim,
                         InputIt begin, InputIt end)
{
  std::unique_ptr<double[]> result(new double[start_point_count * start_point_dim]);
  size_t i = 0;
  for (InputIt it = begin; it != end; ++it)
  {
    const auto& point = it->second;
    for (size_t j = 0; j < start_point_dim; ++j)
    {
      result[i * start_point_dim + j] = static_cast<double>(point[j]);
    }
    ++i;
  }
  if (i != start_point_count)
  {
    throw "Given and actual start point counts do not match";
  }
  return result;
}

}  // namespace cuenum

std::array<uint64_t, FPLLL_EXTENUM_MAX_EXTENUM_DIM> fplll_cuda_enum(const int dim, double maxdist, std::function<extenum_cb_set_config> cbfunc,
  std::function<extenum_cb_process_sol> cbsol, std::function<extenum_cb_process_subsol> cbsubsol,
  bool dual = false, bool findsubsols = false);
  
#endif