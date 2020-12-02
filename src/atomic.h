#ifndef FPLLL_CUDA_ATOMIC_CUH
#define FPLLL_CUDA_ATOMIC_CUH

/**
 * Defines utility functions performing atomic operations.
 * The functionality differs from the standard atomic operations in two ways:
 * 
 * - Some functions are warp-aggregated (notably aggregated_atomic_inc), which
 *   improves performance. However, watch out that coalesced threads in a warp have to
 *   call the function for the same address 
 * 
 * - The functions are also callable from the host, in which case they are NOT atomic,
 *   but still perform the corresponding operation. Helps if you want to execute the code
 *   single-threaded on the host for debugging.
 */

#include <assert.h>
#include <atomic>

#ifdef __CUDACC__
#define DEVICE_HOST_FUNCTION __device__ __host__
#include "cuda_runtime.h"
#include "cooperative_groups.h"
#else
#define DEVICE_HOST_FUNCTION
#endif

template<typename F, typename T> DEVICE_HOST_FUNCTION T bitcast(F bits)
{
    static_assert(sizeof(F) == sizeof(T), "Can only bit-cast types of same size");
    union Convert {
        T out;
        F in;
    };
    Convert result;
    result.in = bits;
    return result.out;
}

DEVICE_HOST_FUNCTION inline unsigned int float_to_int_order_preserving_bijection(float value)
{
    unsigned int val = bitcast<float, unsigned int>(value);
    unsigned int flip_all_if_negative_mask = static_cast<unsigned int>(-static_cast<int>(val >> 31));
    return val ^ (flip_all_if_negative_mask | 0x80000000);
}

DEVICE_HOST_FUNCTION inline float int_to_float_order_preserving_bijection(unsigned int value)
{
    unsigned int flip_all_if_negative_mask = static_cast<unsigned int>(-static_cast<int>((value >> 31) ^ 1));
    return bitcast<unsigned int, float>(value ^ (flip_all_if_negative_mask | 0x80000000));
}

/**
 * Atomically loads the value at target, adds the given amount and stores the result at target. The
 * loaded (old) value of target will be returned. This function is atomic w.r.t other kernel threads
 * on the device, and not atomic if called from the host. 
 */
DEVICE_HOST_FUNCTION inline uint64_t atomic_add(uint64_t *target, uint64_t amount)
{
  static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "atomicAdd is only defined for unsigned long long, so we can only use it for uint64_t if unsigned long long == uint64_t");
#ifdef __CUDA_ARCH__
    return atomicAdd(reinterpret_cast<unsigned long long*>(target), static_cast<unsigned long long>(amount));
#else
    return (*target += amount) - amount;
#endif
}

/**
 * Atomically loads the value at target, adds the given amount and stores the result at target. The
 * loaded (old) value of target will be returned. This function is atomic w.r.t other kernel threads
 * on the device, and not atomic if called from the host. 
 */
DEVICE_HOST_FUNCTION inline uint32_t atomic_add(uint32_t *target, uint32_t amount)
{
#ifdef __CUDA_ARCH__
  return atomicAdd(target, amount);
#else
  return (*target += amount) - amount;
#endif
}

/**
 * Atomically loads the value at target, increments it and stores the result at target. The
 * loaded (old) value of target will be returned. This function is atomic w.r.t other kernel threads
 * on the device, and not atomic if called from the host. If a multiple, coalesced threads call
 * this function simultaneously with different values of target, the behavior is undefined. In particular,
 * it is safe if all threads in a warp call use the same value for target, independent of how they
 * are coalesced.
 */
DEVICE_HOST_FUNCTION inline uint32_t aggregated_atomic_inc(uint32_t *target)
{
#ifdef __CUDA_ARCH__
  // use warp-aggregated atomic for improved performance
  auto g = cooperative_groups::coalesced_threads();
  uint32_t warp_res;
  if (g.thread_rank() == 0)
  {
    warp_res = atomicAdd(target, g.size());
  }
  return g.shfl(warp_res, 0) + g.thread_rank();
#else
  return (*target)++;
#endif
}

/**
 * Atomically loads the value at target, increments it and stores the result at target. The
 * loaded (old) value of target will be returned. This function is atomic w.r.t other kernel threads
 * on the device, and not atomic if called from the host. If a multiple, coalesced threads call
 * this function simultaneously with different values of target, the behavior is undefined. In particular,
 * it is safe if all threads in a warp call use the same value for target, independent of how they
 * are coalesced.
 */
DEVICE_HOST_FUNCTION inline uint64_t aggregated_atomic_inc(uint64_t *ctr)
{
  static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "atomicAdd is only defined for unsigned long long, so we can only use it for uint64_t if unsigned long long == uint64_t");
#ifdef __CUDA_ARCH__
  // use warp-aggregated atomic for improved performance
  auto g = cooperative_groups::coalesced_threads();
  uint64_t warp_res;
  if (g.thread_rank() == 0)
  {
    warp_res = atomicAdd(reinterpret_cast<unsigned long long*>(ctr), static_cast<unsigned long long>(g.size()));
  }
  return g.shfl(warp_res, 0) + g.thread_rank();
#else
  return (*ctr)++;
#endif
}

/**
 * Atomically loads the value at target, calculates min(loaded_value, value) and stores the result 
 * at target. The loaded (old) value of target will be returned. This function is atomic w.r.t other 
 * kernel threads on the device, and not atomic if called from the host. 
 */
DEVICE_HOST_FUNCTION inline uint32_t atomic_min(uint32_t *target, uint32_t value)
{
#ifdef __CUDA_ARCH__
  return atomicMin(target, value);
#else
  uint32_t result = *target;
  if (value < *target)
  {
    *target = value;
  }
  return result;
#endif
}

/**
 * Atomically loads a representation of the floating point value at target, 
 * calculates min(loaded_value, value) and stores a corresponding representation 
 * of the result at target. Storing only a non-standard representation of the floating
 * point numbers at target works around the problem that no atomics are provided
 * for floats. If you have to directly interact with the representation, use
 * int_to_float_order_preserving_bijection() and its inverse to convert to/from
 * standard floats.
 * The loaded (old) value of target will be returned. 
 * 
 * This function is atomic w.r.t other kernel threads on the device, and not atomic 
 * if called from the host. 
 */
DEVICE_HOST_FUNCTION inline float atomic_min(uint32_t *target, float value)
{
  return int_to_float_order_preserving_bijection(
      atomic_min(target, float_to_int_order_preserving_bijection(value)));
}

DEVICE_HOST_FUNCTION inline uint32_t atomic_load(volatile uint32_t *target)
{
  // in cuda, 64-resp. 32-bit loads cannot be "torn" (if properly aligned)
  return *target;
}

/**
 * Performs a threadfence when called from the device, i.e. ensures that other device threads
 * and reads/copies from the host that observe data written after this call also will observe
 * all writes that happened before this call by the current thread.
 * 
 * Does nothing when called from the host.
 */
DEVICE_HOST_FUNCTION inline void threadfence_system() {
#ifdef __CUDA_ARCH__
  __threadfence_system();
#endif
}

#endif