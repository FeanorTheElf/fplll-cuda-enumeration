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

#include "atomic.h"
#include "enum.cuh"
#include "cuda_wrapper.h"
#include <map>

namespace cuenum {

template <int min> struct int_marker
{
};

template <int dimensions_per_level, int levels>
inline std::vector<uint64_t> search_enumeration_choose_levels(
    const enumf *mu, const enumf *rdiag, const unsigned int enum_levels,
    const enumi *start_point_coefficients, unsigned int start_point_count,
    unsigned int start_point_dim, enumf initial_radius, process_sol_fn evaluator,
    CudaEnumOpts enum_opts, int_marker<dimensions_per_level>, int_marker<levels>, int_marker<0>)
{
  if (enum_levels != levels) {
    throw "enumeration dimension must be within the allowed interval";
  }
  assert(enum_opts.max_subtree_paths * enumerate_cooperative_group_size <
         cudaenum_max_nodes_per_level);
  unsigned int max_children_node_count =
      cudaenum_max_nodes_per_level - enum_opts.max_subtree_paths * enumerate_cooperative_group_size;
  Opts<levels, dimensions_per_level, cudaenum_max_nodes_per_level> opts = {
      enum_opts.max_subtree_paths, enum_opts.min_active_parents_percentage, max_children_node_count,
      enum_opts.initial_nodes_per_group, enum_opts.thread_count};
  return enumerate(mu, rdiag, start_point_coefficients, start_point_dim, start_point_count, initial_radius,
            evaluator, opts);
}

template <int dimensions_per_level, int min_levels, int delta_levels>
inline std::vector<uint64_t> search_enumeration_choose_levels(const enumf *mu, const enumf *rdiag,
                                             const unsigned int enum_levels,
                                             const enumi *start_point_coefficients,
                                             unsigned int start_point_count,
                                             unsigned int start_point_dim, enumf initial_radius,
                                             process_sol_fn evaluator, CudaEnumOpts enum_opts,
                                             int_marker<dimensions_per_level>,
                                             int_marker<min_levels>, int_marker<delta_levels>)
{
  static_assert(delta_levels >= 0, "delta_levels >= 0 must hold");
  assert(enum_levels >= min_levels);
  assert(enum_levels <= min_levels + delta_levels);

  constexpr unsigned int delta_mid = delta_levels / 2;
  if (enum_levels <= min_levels + delta_mid)
  {
    return search_enumeration_choose_levels(mu, rdiag, enum_levels, start_point_coefficients,
                                     start_point_count, start_point_dim, initial_radius, evaluator,
                                     enum_opts, int_marker<dimensions_per_level>(),
                                     int_marker<min_levels>(), int_marker<delta_mid>());
  }
  else
  {
    return search_enumeration_choose_levels(
        mu, rdiag, enum_levels, start_point_coefficients, start_point_count, start_point_dim,
        initial_radius, evaluator, enum_opts, int_marker<dimensions_per_level>(),
        int_marker<min_levels + delta_mid + 1>(), int_marker<delta_levels - delta_mid - 1>());
  }
}

template <int dimensions_per_level>
inline std::vector<uint64_t> search_enumeration_choose_dims_per_level(
    const enumf *mu, const enumf *rdiag, const unsigned int enum_dimensions,
    const enumi *start_point_coefficients, unsigned int start_point_count,
    unsigned int start_point_dim, enumf initial_radius, process_sol_fn evaluator,
    CudaEnumOpts enum_opts, int_marker<dimensions_per_level>, int_marker<0>)
{
  if (enum_opts.dimensions_per_level != dimensions_per_level) {
    throw "enum_opts.dimensions_per_level not within allowed interval";
  }
  if (enum_dimensions % dimensions_per_level != 0) {
    throw "enumeration dimension count (i.e. dimensions minus start point dimensions) must be divisible by enum_opts.dimensions_per_level";
  }
  unsigned int enum_levels = enum_dimensions / dimensions_per_level;

  return search_enumeration_choose_levels(mu, rdiag, enum_levels, start_point_coefficients,
                                   start_point_count, start_point_dim, initial_radius, evaluator,
                                   enum_opts, int_marker<dimensions_per_level>(), int_marker<1>(),
                                   int_marker<cudaenum_max_levels(dimensions_per_level) - 1>());
}

template <int min_dimensions_per_level, int delta_dimensions_per_level>
inline std::vector<uint64_t> search_enumeration_choose_dims_per_level(
    const enumf *mu, const enumf *rdiag, const unsigned int enum_dimensions,
    const enumi *start_point_coefficients, unsigned int start_point_count,
    unsigned int start_point_dim, enumf initial_radius, process_sol_fn evaluator,
    CudaEnumOpts enum_opts, int_marker<min_dimensions_per_level>,
    int_marker<delta_dimensions_per_level>)
{
  static_assert(delta_dimensions_per_level >= 0, "delta_dimensions_per_level >= 0 must hold");
  assert(enum_opts.dimensions_per_level >= min_dimensions_per_level);
  assert(enum_opts.dimensions_per_level <= min_dimensions_per_level + delta_dimensions_per_level);

  constexpr unsigned int delta_mid = delta_dimensions_per_level / 2;
  if (enum_opts.dimensions_per_level <= min_dimensions_per_level + delta_mid)
  {
    return search_enumeration_choose_dims_per_level(
        mu, rdiag, enum_dimensions, start_point_coefficients, start_point_count, start_point_dim,
        initial_radius, evaluator, enum_opts, int_marker<min_dimensions_per_level>(),
        int_marker<delta_mid>());
  }
  else
  {
    return search_enumeration_choose_dims_per_level(
        mu, rdiag, enum_dimensions, start_point_coefficients, start_point_count, start_point_dim,
        initial_radius, evaluator, enum_opts, int_marker<min_dimensions_per_level + delta_mid + 1>(),
        int_marker<delta_dimensions_per_level - delta_mid - 1>());
  }
}

std::vector<uint64_t> search_enumeration(const double *mu, const double *rdiag,
                             const unsigned int enum_dimensions,
                             const enumi *start_point_coefficients, unsigned int start_point_count,
                             unsigned int start_point_dim, process_sol_fn evaluator,
                             double initial_radius, CudaEnumOpts opts)
{
  return search_enumeration_choose_dims_per_level(mu, rdiag, enum_dimensions, start_point_coefficients,
                                           start_point_count, start_point_dim, initial_radius,
                                           evaluator, opts, int_marker<1>(),
                                           int_marker<cudaenum_max_dims_per_level - 1>());
}

typedef std::function<void(const enumi*, enumf)> simple_callback_fn;

template<unsigned int max_dim, int dims>
inline void recenum_choose_template_instance(
  CudaEnumeration<max_dim> enumeration, 
  unsigned int dim, 
  simple_callback_fn callback, uint64_t *nodes,
  int_marker<dims>, int_marker<0>
) {
  static_assert(dims > 0, "Start point dimension count must be >= 0");
  assert(dims == dim);
  unsigned int max_paths = std::numeric_limits<unsigned int>::max();
  PerfCounter node_counter(nodes);
  FnWrapper<const enumi*, enumf> wrapped_callback(callback);
  enumeration.template enumerate_recursive<FnWrapper<const enumi*, enumf>, CoefficientIterator, dims - 1>(wrapped_callback, max_paths, node_counter, kk_marker<dims - 1>(), CoefficientIterator());
}

template<unsigned int max_dim, int min_dim, int delta_dim>
inline void recenum_choose_template_instance(
  CudaEnumeration<max_dim> enumeration, 
  unsigned int dim, 
  simple_callback_fn callback, uint64_t *nodes, 
  int_marker<min_dim>, int_marker<delta_dim>
) {
  constexpr int delta_mid = delta_dim / 2;
  if (dim <= min_dim + delta_mid)
  {
    return recenum_choose_template_instance(
      enumeration, dim, callback, nodes, int_marker<min_dim>(), int_marker<delta_mid>());
  }
  else
  {
    return recenum_choose_template_instance(
      enumeration, dim, callback, nodes, int_marker<min_dim + delta_mid + 1>(), int_marker<delta_dim - delta_mid - 1>());
  }
}

template<unsigned int max_startdim>
PinnedPtr<enumi> enumerate_start_points(const int dim, const int start_dims, double radius_squared, const enumf* mu, const enumf* rdiag, unsigned int& start_point_count, uint64_t* nodes) {

  std::multimap<enumf, std::vector<enumi>> start_points;

  uint32_t radius_squared_location = float_to_int_order_preserving_bijection(radius_squared);
  Matrix mu_matrix = Matrix(mu, dim).block(dim - start_dims, dim - start_dims);
  CudaEnumeration<max_startdim> enumobj(mu_matrix, &rdiag[dim - start_dims], &radius_squared_location, start_dims);

  simple_callback_fn callback = [&start_points, start_dims](const enumi *x, enumf squared_norm) {
    start_points.insert(std::make_pair(squared_norm, std::vector<enumi>(x, &x[start_dims]))); 
  };

  constexpr int min_dim = cudaenum_min_startdim;
  constexpr int delta_dim = max_startdim - cudaenum_min_startdim;
  
  recenum_choose_template_instance(enumobj, start_dims, callback, nodes, int_marker<min_dim>(), int_marker<delta_dim>());

  start_point_count = start_points.size();
  return create_start_point_array(start_points.size(), start_dims, start_points.begin(), start_points.end());
}

}

std::array<uint64_t, FPLLL_EXTENUM_MAX_EXTENUM_DIM> fplll_cuda_enum(const int dim, enumf maxdist, std::function<extenum_cb_set_config> cbfunc,
  std::function<extenum_cb_process_sol> cbsol, std::function<extenum_cb_process_subsol> cbsubsol,
  bool dual, bool findsubsols) 
{
  if (dual) {
    throw "Unsupported operation: dual == true";
  } else if (findsubsols) {
    throw "Unsupported operation: findsubsols == true";
  }

  std::array<uint64_t, FPLLL_EXTENUM_MAX_EXTENUM_DIM> result = {};
  PinnedPtr<enumf> mu = alloc_pinned_memory<enumf>(dim * dim);
  PinnedPtr<enumf> rdiag = alloc_pinned_memory<enumf>(dim);
  std::unique_ptr<enumf[]> pruning(new enumf[dim]);
  enumf radius = std::sqrt(maxdist);

  cbfunc(mu.get(), dim, true, rdiag.get(), pruning.get());

  // functions in this library require 1 on the diagonal, not zero
  for (unsigned int i = 0; i < dim; ++i) {
    mu.get()[i * dim + i] = 1;
  }

  cuenum::CudaEnumOpts opts = cuenum::default_opts;
  
  int start_dims = cuenum::cudaenum_min_startdim;
  while ((dim - start_dims) % opts.dimensions_per_level != 0) {
    ++start_dims;
  }
  if (start_dims >= dim) {
      // use fallback, as cuda enumeration in such small dimensions is too much overhead
      return result;
  }

  unsigned int start_point_count = 0;
  uint64_t* start_enum_node_counts = &result[dim - start_dims];
  PinnedPtr<enumi> start_point_array = cuenum::enumerate_start_points<cuenum::cudaenum_min_startdim + cuenum::cudaenum_max_dims_per_level>(dim, start_dims, maxdist, mu.get(), rdiag.get(), start_point_count, start_enum_node_counts);

  std::vector<uint64_t> node_counts = cuenum::search_enumeration(mu.get(), rdiag.get(), dim - start_dims, start_point_array.get(), 
    start_point_count, start_dims, cbsol, radius, opts);

  std::copy(node_counts.begin(), node_counts.end(), result.begin());
  return result;
}

constexpr extenum_fc_enumerate* __check_function_interface_matches = fplll_cuda_enum;