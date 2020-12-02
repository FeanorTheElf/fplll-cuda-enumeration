#ifndef FPLLL_RECENUM_CUH
#define FPLLL_RECENUM_CUH

/**
 * This file contains a recursive enumeration algorithm (similar to the base enumeration in fplll),
 * that may be called from the device and is extremely fast, but may lead to much thread divergence,
 * if used for a many dimensions
 */

#include "cuda_runtime.h"
#include "types.cuh"

namespace cuenum
{

template <int kk> struct kk_marker
{
};

struct CoefficientIterator {
  
  __device__ __host__ inline enumi operator()(enumi last_coeff, const enumf center)
  {
    const enumi rounded_center = static_cast<enumi>(round(center));
    last_coeff                 = 2 * rounded_center - last_coeff;
    if (center >= rounded_center)
    {
      return last_coeff + static_cast<int>(last_coeff >= rounded_center);
    }
    else
    {
      return last_coeff - static_cast<int>(last_coeff <= rounded_center);
    }
  }

};

struct PositiveCoefficientIterator {

  // For the last dimension, it is sufficient to consider only positive coefficients, as a lattice
  // is closed w.r.t negation. In this case, use this coefficient iterator.
  __device__ __host__ constexpr inline enumi operator()(enumi last_coeff, const enumf center)
  {
    return last_coeff - 1;
  }

};

template <unsigned int maxdim> class CudaEnumeration
{
private:

  enumi x[maxdim];
  enumf partdist[maxdim];
  // different to base enumeration of fplll, the second index is shifted
  // _[i][j] contains inner product of i-th orthogonalized basis vector with 
  // B * (0, ..., 0, x[j + 1], ... x[n])
  enumf center_partsums[maxdim][maxdim];
  enumf center[maxdim];
  const uint32_t *radius_squared_location;

  Matrix mu;
  const enumf *rdiag;

public:

  __device__ __host__ CudaEnumeration() {}

  /**
   * Initializes this enumeration object to enumerate points in the lattice spanned by the enum_dim x enum_dim lower-right
   * submatrix of mu, around the origin.
   */  
  __device__ __host__ CudaEnumeration(Matrix mu, const enumf* rdiag, const uint32_t* radius_squared_location, unsigned int enum_dim)
    : mu(mu), rdiag(rdiag), radius_squared_location(radius_squared_location)
  {
    for (unsigned int i = 0; i < maxdim; ++i) {
      x[i] = NAN;
      center_partsums[i][enum_dim - 1] = 0;
    }
    x[enum_dim - 1] = 0;
    center[enum_dim - 1] = 0;
    partdist[enum_dim - 1] = 0;
  }

  template <typename Callback, typename CoeffIt>
  __device__ __host__ bool enumerate_recursive(Callback &callback, unsigned int &max_paths,
                                               PerfCounter &counter,
                                               kk_marker<0>, CoeffIt);

  template <typename Callback, typename CoeffIt, int kk>
  __device__ __host__ bool enumerate_recursive(Callback &callback, unsigned int &max_paths,
                                               PerfCounter &counter,
                                               kk_marker<kk>, CoeffIt);

  template <int kk> __device__ __host__ bool is_enumeration_done() const;

  __device__ __host__ enumf get_radius_squared();

  template <unsigned int levels, unsigned int dimensions_per_level,
            unsigned int max_nodes_per_level>
  friend struct SubtreeEnumerationBuffer;
};

template <unsigned int maxdim>
__device__ __host__ inline enumf CudaEnumeration<maxdim>::get_radius_squared()
{
  return int_to_float_order_preserving_bijection(*radius_squared_location);
}

template <unsigned int maxdim>
template <int kk>
__device__ __host__ inline bool CudaEnumeration<maxdim>::is_enumeration_done() const
{
  static_assert(kk < static_cast<int>(maxdim) && kk >= 0,
                "Tree level count must between 0 and maximal enumeration dimension count");
  return isnan(x[kk]);
}

/**
 * Searches the subtree of height kk + 1 using as root the values stored in this object. The
 * reference max_paths contains an integer that gives the maximal count of tree paths to search
 * (including tree paths that lead to nodes with too great partdist and are therefore cut). After
 * this is exceeded, the tree search is aborted but can be resumed later by calling
 * enumerate_recursive on this object. The function returns whether the subtree was completely
 * searched.
 * 
 * Adjustment of enumerate_recursive() in enumerate_base.cpp of fplll
 */
template <unsigned int maxdim>
template <typename Callback, typename CoeffIt, int kk>
__device__ __host__ inline bool
CudaEnumeration<maxdim>::enumerate_recursive(Callback &callback, unsigned int &max_paths,
                                             PerfCounter &node_counter, kk_marker<kk>, CoeffIt next_coeff)
{
  static_assert(kk < static_cast<int>(maxdim) && kk > 0,
                "Tree level count must between 0 and maximal enumeration dimension count");

  enumf alphak  = x[kk] - center[kk];
  enumf newdist = partdist[kk] + alphak * alphak * rdiag[kk];

  assert(max_paths >= 1);
  assert(!isnan(alphak));
  assert(partdist[kk] >= 0);
  assert(rdiag[kk] >= 0);
  assert(newdist >= 0);

  if (!(newdist <= get_radius_squared()))
  {
    x[kk] = NAN;
    return true;
  }

  partdist[kk - 1] = newdist;

  for (int j = 0; j < kk; ++j)
  {
    center_partsums[j][kk - 1] = center_partsums[j][kk] - x[kk] * mu.at(j, kk);
  }
  assert(!isnan(center_partsums[kk - 1][kk - 1]));

  center[kk - 1] = center_partsums[kk - 1][kk - 1];
  if (isnan(x[kk - 1]))
  {
    x[kk - 1] = round(center[kk - 1]);
  }

  while (true)
  {
    node_counter.inc(kk);
    bool is_done = enumerate_recursive(callback, max_paths, node_counter, kk_marker<kk - 1>(), CoefficientIterator());
    if (!is_done)
    {
      return false;
    }

    x[kk] = next_coeff(x[kk], center[kk]);

    enumf alphak2  = x[kk] - center[kk];
    enumf newdist2 = partdist[kk] + alphak2 * alphak2 * rdiag[kk];
    assert(newdist2 >= 0);

    if (max_paths == 1)
    {
      return false;
    }
    --max_paths;

    if (!(newdist2 <= get_radius_squared()))
    {
      x[kk] = NAN;
      return true;
    }

    partdist[kk - 1] = newdist2;

    for (int j = 0; j < kk; ++j)
    {
      center_partsums[j][kk - 1] = center_partsums[j][kk] - x[kk] * mu.at(j, kk);
    }
    assert(!isnan(center_partsums[kk - 1][kk - 1]));

    center[kk - 1] = center_partsums[kk - 1][kk - 1];
    x[kk - 1]      = round(center[kk - 1]);
  }
}

template <unsigned int maxdim>
template <typename Callback, typename CoeffIt>
__device__ __host__ inline bool
CudaEnumeration<maxdim>::enumerate_recursive(Callback &callback, unsigned int &max_paths,
                                             PerfCounter &node_counter, kk_marker<0>, CoeffIt next_coeff)
{
  constexpr unsigned int kk = 0;
  static_assert(kk < static_cast<int>(maxdim) && kk >= 0,
                "Tree level count must between 0 and maximal enumeration dimension count");

  enumf alphak  = x[kk] - center[kk];
  enumf newdist = partdist[kk] + alphak * alphak * rdiag[kk];

  assert(max_paths >= 1);
  assert(!isnan(x[kk]));
  assert(partdist[kk] >= 0);
  assert(rdiag[kk] >= 0);
  assert(newdist >= 0);

  if (!(newdist <= get_radius_squared()))
  {
    x[kk] = NAN;
    return true;
  }

  callback(x, newdist);

  while (true)
  {
    node_counter.inc(kk);
    x[kk] = next_coeff(x[kk], center[kk]);

    enumf alphak2  = x[kk] - center[kk];
    enumf newdist2 = partdist[kk] + alphak2 * alphak2 * rdiag[kk];
    assert(newdist2 >= 0);

    if (max_paths == 1)
    {
      return false;
    }
    --max_paths;

    if (!(newdist2 <= get_radius_squared()))
    {
      x[kk] = NAN;
      return true;
    }
  
    callback(x, newdist2);
  }
}

}  // namespace cudaenum

#endif