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

#ifndef FPLLL_ENUM_CUH
#define FPLLL_ENUM_CUH

/**
       * This file contains the enumeration routine itself
       */

#include "cooperative_groups.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdint.h>
#include <functional>
#include <vector>
#include <numeric>

#include "constants.cuh"
#include "atomic.h"
#include "cuda_util.cuh"
#include "group.cuh"
#include "streaming.cuh"
#include "recenum.cuh"

namespace cuenum
{

    constexpr bool CUENUM_TRACE = false;

    /**
        * Wrapper around the store of the state of the enumeration tree search, i.e. all nodes of the tree
        * whose subtrees have to be searched, or are currently searched, ordered by tree level.
        *
        * A node is a dimensions_per_level-level subtree of the enumeration tree that is traversed via
        * enumerate_recursive() and is given by the point corresponding to its root, i.e. a point in the
        * sublattice spanned by the last level * dimensions_per_level basis vectors.
        * The nodes at level 0 correspond to the roots of the subtrees passed to the GPU. The nodes at level
        * levels are the leaves of the enumeration tree, but will not be stored explicitly. Therefore, the
        * last level (levels - 1) will contain the nodes that have still dimensions_per_level unset
        * coefficients.
        *
        * This class is not itself the store, as it does not own the values, it just wraps the pointer to the
        * memory to provide abstract access functionality.
        */
    template <unsigned int levels, unsigned int dimensions_per_level, unsigned int max_nodes_per_level>
    class SubtreeEnumerationBuffer
    {
    private:
        unsigned char *memory;

        constexpr static unsigned int dimensions = levels * dimensions_per_level;

        constexpr static unsigned int enumeration_x_size_in_bytes =
            sizeof(enumi) * levels * dimensions_per_level * max_nodes_per_level;

        constexpr static unsigned int coefficient_size_in_bytes =
            sizeof(enumi) * levels * dimensions_per_level * max_nodes_per_level;

        constexpr static unsigned int center_partsum_size_in_bytes =
            sizeof(enumf) * levels * dimensions * max_nodes_per_level;

        constexpr static unsigned int partdist_size_in_bytes =
            sizeof(enumf) * levels * max_nodes_per_level;

        constexpr static unsigned int parent_indices_size_in_bytes =
            sizeof(unsigned int) * levels * max_nodes_per_level;

        constexpr static unsigned int open_node_count_size_in_bytes = sizeof(unsigned int) * levels;

        constexpr static size_t content_memory_size_in_bytes =
            enumeration_x_size_in_bytes + coefficient_size_in_bytes + center_partsum_size_in_bytes +
            partdist_size_in_bytes + parent_indices_size_in_bytes + open_node_count_size_in_bytes;

        // coefficients of the children enumeration for this point, used to pause and resume
        // enumerate_recursive() shape; [levels, dimensions_per_level, max_nodes_per_level]
        __device__ __host__ inline enumi *enumeration_x()
        {
            return reinterpret_cast<enumi *>(memory + center_partsum_size_in_bytes +
                                             partdist_size_in_bytes);
        }

        // last dimensions_per_level coefficients of the point, the other coefficients must be retrieved
        // by querying the parent node; shape [levels, dimensions_per_level, max_nodes_per_level]
        __device__ __host__ inline enumi *coefficients()
        {
            return reinterpret_cast<enumi *>(memory + center_partsum_size_in_bytes +
                                             partdist_size_in_bytes +
                                             enumeration_x_size_in_bytes);
        }

        // inner products with the scaled lattice vectors and the point, only the first (levels - level) *
        // dimensions_per_level are of interest; shape [levels, dimensions, max_nodes_per_level]
        __device__ __host__ inline enumf *center_partsum()
        {
            return reinterpret_cast<enumf *>(memory);
        }

        // squared norm of the point, projected into the perpendicular subspace to the first (levels -
        // level) * dimensions_per_level basis vectors; shape [levels, max_nodes_per_level]
        __device__ __host__ inline enumf *partdist()
        {
            return reinterpret_cast<enumf *>(memory + center_partsum_size_in_bytes);
        }

        // shape [levels, max_nodes_per_level]
        __device__ __host__ inline unsigned int *parent_indices()
        {
            return reinterpret_cast<unsigned int *>(
                memory + center_partsum_size_in_bytes + partdist_size_in_bytes +
                enumeration_x_size_in_bytes + coefficient_size_in_bytes);
        }

        // shape [levels]
        __device__ __host__ inline unsigned int *open_node_count()
        {
            return reinterpret_cast<unsigned int *>(
                memory + center_partsum_size_in_bytes + partdist_size_in_bytes +
                enumeration_x_size_in_bytes + coefficient_size_in_bytes + parent_indices_size_in_bytes);
        }

    public:
        __device__ __host__ inline SubtreeEnumerationBuffer(unsigned char *memory)
            : memory(memory)
        {
            assert(((intptr_t)memory) % sizeof(enumf) == 0);
        }

        template <typename CG>
        __device__ __host__ inline void init(CG &cooperative_group)
        {
            if (cooperative_group.thread_rank() == 0)
            {
                for (unsigned int i = 0; i < levels; ++i)
                {
                    open_node_count()[i] = 0;
                }
            }
        }

        // ensure that the memory size is a multiple of sizeof(enumf), to have correct alignment
        // if multiple buffers use subsequent memory batches
        constexpr static size_t memory_size_in_bytes =
            ((content_memory_size_in_bytes - 1) / sizeof(enumf) + 1) * sizeof(enumf);

        static_assert(memory_size_in_bytes >= content_memory_size_in_bytes,
                      "Bug in memory_size_in_bytes calculation");

        static_assert(memory_size_in_bytes < std::numeric_limits<unsigned int>::max(),
                      "Requires more memory than indexable with unsigned int");

        __device__ __host__ inline CudaEnumeration<dimensions_per_level>
        get_enumeration(unsigned int tree_level, unsigned int index, Matrix mu_block, const enumf *rdiag,
                        const volatile enumf *pruning_bounds)
        {
            assert(tree_level < levels);
            assert(index < max_nodes_per_level);
            CudaEnumeration<dimensions_per_level> result;

            const unsigned int offset_kk = (levels - tree_level - 1) * dimensions_per_level;

            result.mu = mu_block;
            result.rdiag = &rdiag[offset_kk];
            result.pruning_bounds = &pruning_bounds[offset_kk];

            for (unsigned int i = 0; i < dimensions_per_level; ++i)
            {
                result.x[i] = enumeration_x()[tree_level * dimensions_per_level * max_nodes_per_level +
                                              i * max_nodes_per_level + index];

                const enumf center_partsum_i = center_partsum()[tree_level * dimensions * max_nodes_per_level +
                                                                (offset_kk + i) * max_nodes_per_level + index];
                assert(!isnan(center_partsum_i));
                result.center_partsums[i][dimensions_per_level - 1] = center_partsum_i;
            }

            result.center[dimensions_per_level - 1] =
                center_partsum()[tree_level * dimensions * max_nodes_per_level +
                                 (offset_kk + dimensions_per_level - 1) * max_nodes_per_level + index];
            result.partdist[dimensions_per_level - 1] = partdist()[tree_level * max_nodes_per_level + index];
            return result;
        }

        __device__ __host__ inline void
        set_enumeration(unsigned int tree_level, unsigned int index,
                        const CudaEnumeration<dimensions_per_level> &value)
        {
            assert(tree_level < levels);
            assert(index < max_nodes_per_level);
#ifndef NDEBUG
            const unsigned int offset_kk = (levels - tree_level - 1) * dimensions_per_level;
#endif

            for (unsigned int i = 0; i < dimensions_per_level; ++i)
            {
                enumeration_x()[tree_level * dimensions_per_level * max_nodes_per_level +
                                i * max_nodes_per_level + index] = value.x[i];

                assert(center_partsum()[tree_level * dimensions * max_nodes_per_level +
                                        (offset_kk + i) * max_nodes_per_level + index] ==
                       value.center_partsums[i][dimensions_per_level - 1]);
            }
            assert(center_partsum()[tree_level * dimensions * max_nodes_per_level +
                                    (offset_kk + dimensions_per_level - 1) * max_nodes_per_level + index] ==
                   value.center[dimensions_per_level - 1]);
            assert(partdist()[tree_level * max_nodes_per_level + index] ==
                   value.partdist[dimensions_per_level - 1]);
        }

        /**
           * Given an existing node (with correct parent, coefficients) initializes
           * enumeration_x to allow children enumeration;
           * Expects the single center partsum that is used as center to be initialized, the others
           * may be still uninitialized. Note that partdist and some of the other center partsums
           * are also required for children enumeration, but are not touched by this function.
           */
        __device__ __host__ inline void init_enumeration(unsigned int tree_level, unsigned int index)
        {
            for (unsigned int i = 0; i < dimensions_per_level; ++i)
            {
                enumeration_x()[tree_level * dimensions_per_level * max_nodes_per_level +
                                i * max_nodes_per_level + index] = NAN;
            }
            const unsigned int kk_offset = (levels - tree_level - 1) * dimensions_per_level;
            const enumf center = get_center_partsum(tree_level, index, kk_offset + dimensions_per_level - 1);
            assert(!isnan(center));
            assert(!isnan(get_partdist(tree_level, index)));
            enumeration_x()[tree_level * dimensions_per_level * max_nodes_per_level +
                            (dimensions_per_level - 1) * max_nodes_per_level + index] =
                static_cast<enumi>(round(center));
        }

        /**
           * Checks whether the partsums for a specific target level are initialized by checking a representative entry.
           * Note that one should always initialize all or no partsums for a given level in a certain node, otherwise
           * the information from this function will be too coarse-grained and hence wrong.
           */
        __device__ __host__ inline bool are_partsums_initialized(unsigned int tree_level, unsigned int index, unsigned int target_level)
        {
            const unsigned int kk_offset = (levels - target_level - 1) * dimensions_per_level;
            return !isnan(get_center_partsum(tree_level, index, kk_offset + dimensions_per_level - 1));
        }

        /**
           * Given an existing node initializes the enumeration_x and center partsums with an
           * empty value such that it can be detected and computed later.
           * This will leave the parent and coefficients unchanged, as it will usually be used
           * to allow lazy computation of the center data for this node.
           * 
           * Usually when adding a node, call clear_enumeration(), then set partsums later, and
           * then call init_enumeration().
           */
        __device__ __host__ inline void clear_enumeration(unsigned int tree_level, unsigned int index)
        {
            for (unsigned int i = 0; i < dimensions_per_level; ++i)
            {
                enumeration_x()[tree_level * dimensions_per_level * max_nodes_per_level +
                                i * max_nodes_per_level + index] = NAN;
            }
            for (unsigned int i = 0; i < (levels - tree_level) * dimensions_per_level; ++i)
            {
                center_partsum()[tree_level * dimensions * max_nodes_per_level +
                                 i * max_nodes_per_level + index] = NAN;
            }
        }

        __device__ __host__ inline void set_center_partsum(unsigned int tree_level, unsigned int index,
                                                           unsigned int orth_basis_index, enumf value)
        {
            assert(tree_level < levels);
            assert(index < max_nodes_per_level);
            assert(orth_basis_index < dimensions);
            center_partsum()[tree_level * dimensions * max_nodes_per_level +
                             orth_basis_index * max_nodes_per_level + index] = value;
        }

        __device__ __host__ inline enumf get_center_partsum(unsigned int tree_level, unsigned int index,
                                                            unsigned int orth_basis_index)
        {
            assert(tree_level < levels);
            assert(index < max_nodes_per_level);
            assert(orth_basis_index < dimensions);
            return center_partsum()[tree_level * dimensions * max_nodes_per_level +
                                    orth_basis_index * max_nodes_per_level + index];
        }

        __device__ __host__ inline enumi get_coefficient(unsigned int tree_level, unsigned int index,
                                                         unsigned int coordinate)
        {
            assert(tree_level < levels);
            assert(index < max_nodes_per_level);
            assert(coordinate < dimensions_per_level);
            return coefficients()[tree_level * dimensions_per_level * max_nodes_per_level +
                                  coordinate * max_nodes_per_level + index];
        }

        __device__ __host__ inline void set_coefficient(unsigned int tree_level, unsigned int index,
                                                        unsigned int coordinate, enumi value)
        {
            assert(tree_level < levels);
            assert(index < max_nodes_per_level);
            assert(coordinate < dimensions_per_level);
            coefficients()[tree_level * dimensions_per_level * max_nodes_per_level +
                           coordinate * max_nodes_per_level + index] = value;
        }

        __device__ __host__ inline unsigned int get_parent_index(unsigned int tree_level,
                                                                 unsigned int index)
        {
            assert(tree_level < levels);
            assert(index < max_nodes_per_level);
            return parent_indices()[tree_level * max_nodes_per_level + index];
        }

        __device__ __host__ inline enumf get_partdist(unsigned int tree_level, unsigned int index)
        {
            assert(tree_level < levels);
            assert(index < max_nodes_per_level);
            return partdist()[tree_level * max_nodes_per_level + index];
        }

        __device__ __host__ inline void set_partdist(unsigned int tree_level, unsigned int index,
                                                     enumf value)
        {
            assert(tree_level < levels);
            assert(index < max_nodes_per_level);
            partdist()[tree_level * max_nodes_per_level + index] = value;
        }

        __device__ __host__ inline unsigned int get_node_count(unsigned int tree_level)
        {
            assert(tree_level < levels);
            return open_node_count()[tree_level];
        }

        /**
           * Adds a node on the given tree level with the given parent to the buffer.
           * The coefficients of this node should be initialized directly afterwards via
           * set_coefficient(). After this is done, the node correctly exists but is not yet 
           * initialized for children enumeration, as center_partsum, partdist and enumeration_x
           * are missing.
           * 
           * To initialize them, call init_enumeration. After this is done, the node is called fully 
           * initialized. The center partsums must be additionally set before the actual children 
           * enumeration takes place.
           */
        __device__ __host__ inline unsigned int add_node(unsigned int tree_level,
                                                         unsigned int parent_node_index)
        {
            assert(tree_level < levels);
            const unsigned int new_task_index = aggregated_atomic_inc(&open_node_count()[tree_level]);
            assert(new_task_index < max_nodes_per_level);
            // in this case, we want an error also in Release builds
            if (new_task_index >= max_nodes_per_level)
            {
                runtime_error();
            }
            parent_indices()[tree_level * max_nodes_per_level + new_task_index] = parent_node_index;
            return new_task_index;
        }

        /**
            * Removes all nodes this functions was called for with keep_this_thread_task=false from the tree.
            *
            * To allow an efficient implementation, requires that old_index == node_count -
            * active_thread_count + cooperative_group.thread_rank(), i.e. all threads in the group have to
            * call this function for the last active_thread_count nodes in the tree. Calls with
            * cooperative_group.thread_rank() >= active_thread_count will be ignored.
            */
        template <typename SyncGroup>
        __device__ __host__ inline void
        filter_nodes(SyncGroup &group,
                     unsigned int tree_level, unsigned int old_index, bool keep_this_thread_task,
                     unsigned int active_thread_count)
        {
            assert(tree_level < levels);
            assert(active_thread_count <= open_node_count()[tree_level]);
            assert(old_index ==
                   open_node_count()[tree_level] - active_thread_count + group.thread_rank());
            assert(tree_level + 1 == levels || open_node_count()[tree_level + 1] == 0);

            unsigned int kept_tasks = 0;
            const bool is_active =
                keep_this_thread_task && group.thread_rank() < active_thread_count;
            const unsigned int new_offset = group.prefix_count(is_active, kept_tasks);
            const unsigned int new_index = new_offset + open_node_count()[tree_level] - active_thread_count;

            enumi coefficients_tmp[dimensions_per_level];
            enumi enumeration_x_tmp[dimensions_per_level];
            enumf center_partsum_tmp[dimensions];
            enumf partdist_tmp;
            unsigned int parent_index_tmp;
            if (is_active)
            {
                partdist_tmp = partdist()[tree_level * max_nodes_per_level + old_index];
                parent_index_tmp = parent_indices()[tree_level * max_nodes_per_level + old_index];
                for (unsigned int i = 0; i < dimensions_per_level; ++i)
                {
                    enumeration_x_tmp[i] =
                        enumeration_x()[tree_level * dimensions_per_level * max_nodes_per_level +
                                        i * max_nodes_per_level + old_index];
                    coefficients_tmp[i] =
                        coefficients()[tree_level * dimensions_per_level * max_nodes_per_level +
                                       i * max_nodes_per_level + old_index];
                }
                for (unsigned int i = 0; i < dimensions; ++i)
                {
                    center_partsum_tmp[i] = center_partsum()[tree_level * dimensions * max_nodes_per_level +
                                                             i * max_nodes_per_level + old_index];
                }
            }

            group.sync();

            if (is_active)
            {
                partdist()[tree_level * max_nodes_per_level + new_index] = partdist_tmp;
                parent_indices()[tree_level * max_nodes_per_level + new_index] = parent_index_tmp;
                for (unsigned int i = 0; i < dimensions_per_level; ++i)
                {
                    enumeration_x()[tree_level * dimensions_per_level * max_nodes_per_level +
                                    i * max_nodes_per_level + new_index] = enumeration_x_tmp[i];
                    coefficients()[tree_level * dimensions_per_level * max_nodes_per_level +
                                   i * max_nodes_per_level + new_index] = coefficients_tmp[i];
                }
                for (unsigned int i = 0; i < dimensions; ++i)
                {
                    center_partsum()[tree_level * dimensions * max_nodes_per_level + i * max_nodes_per_level +
                                     new_index] = center_partsum_tmp[i];
                }
            }

            if (group.thread_rank() == 0)
            {
                open_node_count()[tree_level] -= active_thread_count - kept_tasks;
            }
        }
    };

    template <typename eval_sol_fn, unsigned int levels, unsigned int dimensions_per_level,
              unsigned int max_nodes_per_level>
    struct ProcessLeafCallback
    {
        unsigned int level;
        unsigned int parent_index;
        unsigned int start_point_dim;
        eval_sol_fn &process_sol;
        Matrix mu;
        const enumi *start_points;
        SubtreeEnumerationBuffer<levels, dimensions_per_level, max_nodes_per_level> buffer;

        __device__ __host__ void operator()(const enumi *x, enumf squared_norm);
    };

    template <typename eval_sol_fn, unsigned int levels, unsigned int dimensions_per_level,
              unsigned int max_nodes_per_level>
    __device__ __host__ inline void
    ProcessLeafCallback<eval_sol_fn, levels, dimensions_per_level, max_nodes_per_level>::operator()(
        const enumi *x, enumf squared_norm)
    {
        if (squared_norm == 0)
        {
            return;
        }
        const unsigned int total_point_dimension = dimensions_per_level * levels + start_point_dim;
        for (unsigned int i = 0; i < dimensions_per_level; ++i)
        {
            process_sol(x[i], i, squared_norm, total_point_dimension);
        }
        unsigned int index = parent_index;
        for (unsigned int j = levels - 1; j > 0; --j)
        {
            for (unsigned int i = 0; i < dimensions_per_level; ++i)
            {
                process_sol(buffer.get_coefficient(j, index, i), i + (levels - j) * dimensions_per_level,
                            squared_norm, total_point_dimension);
            }
            index = buffer.get_parent_index(j, index);
        }
        index = buffer.get_parent_index(0, index);
        for (unsigned int i = 0; i < start_point_dim; ++i)
        {
            process_sol(start_points[index * start_point_dim + i], i + levels * dimensions_per_level,
                        squared_norm, total_point_dimension);
        }
    }

    template <unsigned int levels, unsigned int dimensions_per_level, unsigned int max_nodes_per_level>
    struct AddToTreeCallback
    {
        unsigned int level;
        unsigned int parent_index;
        SubtreeEnumerationBuffer<levels, dimensions_per_level, max_nodes_per_level> buffer;

        __device__ __host__ void operator()(const enumi *x, enumf squared_norm);
    };

    template <unsigned int levels, unsigned int dimensions_per_level, unsigned int max_nodes_per_level>
    __device__ __host__ inline void
    AddToTreeCallback<levels, dimensions_per_level, max_nodes_per_level>::operator()(const enumi *x,
                                                                                     enumf squared_norm)
    {
        assert(level > 0);
        const unsigned int new_index = buffer.add_node(level, parent_index);
        for (unsigned int j = 0; j < dimensions_per_level; ++j)
        {
            buffer.set_coefficient(level, new_index, j, x[j]);
        }
        buffer.clear_enumeration(level, new_index);
        buffer.set_partdist(level, new_index, squared_norm);
    }

    /**
        * Calculates the difference of this center partsum to the center partsum of the parent point
        */
    template <unsigned int levels, unsigned int dimensions_per_level>
    __device__ __host__ inline enumf calc_center_partsum_delta(unsigned int level, unsigned int index,
                                                               unsigned int center_partsum_index,
                                                               enumi x[dimensions_per_level], Matrix mu)
    {
        unsigned int kk_offset = (levels - level - 1) * dimensions_per_level;
        enumf center_partsum = 0;
        for (unsigned int j = 0; j < dimensions_per_level; ++j)
        {
            center_partsum -= x[j] * mu.at(center_partsum_index, j + dimensions_per_level + kk_offset);
        }
        assert(!isnan(center_partsum));
        return center_partsum;
    }

    struct StrategyOpts
    {
        unsigned int max_subtree_paths;
        // stop children generation when the percentage of parent points that have still unprocessed
        // children drops beneath this percentage
        float min_active_parents_percentage;
        // stop children generation when the count of children exceeds this limit
        unsigned int max_children_buffer_size;
    };

    template <unsigned int levels, unsigned int dimensions_per_level, unsigned int max_nodes_per_level>
    struct TreeLevelEnumerator
    {

    private:
        SubtreeEnumerationBuffer<levels, dimensions_per_level, max_nodes_per_level> &buffer;
        const Matrix &mu;
        const enumf *rdiag;
        PerfCounter &counter;
        const volatile enumf *pruning_bounds;
        const enumi *start_points;
        const unsigned int start_point_dim;
        const StrategyOpts &opts;

    public:
        int level;

        __device__ __host__ TreeLevelEnumerator(
            SubtreeEnumerationBuffer<levels, dimensions_per_level, max_nodes_per_level> &buffer,
            const Matrix &mu,
            const enumf *rdiag,
            PerfCounter &counter,
            const volatile enumf *pruning_bounds,
            const enumi *start_points,
            unsigned int start_point_dim,
            const StrategyOpts &opts)
            : buffer(buffer), mu(mu), rdiag(rdiag), counter(counter), pruning_bounds(pruning_bounds), start_points(start_points), start_point_dim(start_point_dim), opts(opts) {}

        __device__ __host__ inline unsigned int offset_kk() const
        {
            return (levels - level - 1) * dimensions_per_level;
        }

        template <typename SyncGroup>
        __device__ __host__ void ensure_enumeration_initialized(
            SyncGroup &group, int target_level);

        /**
            * Initializes newly generated nodes with all information necessary to perform subtree enumeration,
            * namely the center_partsums and the partdist. Requires the coefficients of these nodes to be
            * already stored in the tree buffer.
            *
            * Newly generated nodes are all nodes on the given level, except the already_calculated_node_count
            * first nodes.
            */
        template <typename SyncGroup>
        __device__ __host__ inline void
        init_new_nodes(SyncGroup &group, unsigned int already_calculated_node_count);

        /**
           * Generates more children for the last group.size() nodes on the given level and adds them to the
           * buffer. The subtree enumerations of the processed nodes are accordingly updated so that they will
           * only yield new children.
           *
           * Synchronization: count(level) write(level) add(level + 1)
           */
        template <typename SyncGroup>
        __device__ __host__ void generate_children(
            SyncGroup &group)
        {
            assert(level < levels - 1);
            assert(level >= 0);

            const unsigned int node_count = buffer.get_node_count(level);
            const unsigned int index = node_count - min(node_count, group.size()) + group.thread_rank();
            const bool active = index < node_count;

            unsigned int max_paths = opts.max_subtree_paths;
            unsigned int existing_nodes = buffer.get_node_count(level + 1);

            ensure_enumeration_initialized(group, level);

            group.sync();

            if (active)
            {
                CudaEnumeration<dimensions_per_level> enumeration = buffer.get_enumeration(
                    level, index, mu.block(offset_kk(), offset_kk()), rdiag, pruning_bounds);

                bool is_done = enumeration.template is_enumeration_done<dimensions_per_level - 1>();

                if (!is_done)
                {
                    typedef AddToTreeCallback<levels, dimensions_per_level, max_nodes_per_level> CallbackType;
                    CallbackType callback = {static_cast<unsigned int>(level + 1), index, buffer};
                    PerfCounter offset_counter = counter.offset_level(offset_kk());
                    CoefficientIterator iter;
                    enumeration.enumerate_recursive(
                        callback, max_paths, offset_counter, kk_marker<dimensions_per_level - 1>(), iter);

                    buffer.set_enumeration(level, index, enumeration);
                }
            }
        }

        /**
           * Searches the subtrees of the last group.size() nodes on the last tree level, possibly finding a
           * new nonzero shortest vector. The subtree enumerations of the processed nodes are accordingly
           * updated so that they will only yield new vectors.
           */
        template <typename SyncGroup, typename eval_sol_fn>
        __device__ __host__ void inline consume_leaves(
            SyncGroup &group, eval_sol_fn &process_sol)
        {
            const unsigned int level = levels - 1;
            const unsigned int node_count = buffer.get_node_count(level);
            const unsigned int index = node_count - min(node_count, group.size()) + group.thread_rank();
            const bool active = index < node_count;
            unsigned int max_paths = opts.max_subtree_paths;

            ensure_enumeration_initialized(group, level);

            group.sync();

            if (active)
            {
                CudaEnumeration<dimensions_per_level> enumeration =
                    buffer.get_enumeration(level, index, mu, rdiag, pruning_bounds);

                typedef ProcessLeafCallback<eval_sol_fn, levels, dimensions_per_level, max_nodes_per_level>
                    CallbackT;
                CallbackT callback = {level + 1, index, start_point_dim, process_sol,
                                      mu, start_points, buffer};
                CoefficientIterator iter;
                enumeration.enumerate_recursive(
                    callback, max_paths, counter, kk_marker<dimensions_per_level - 1>(), iter);

                buffer.set_enumeration(level, index, enumeration);
            }
        }

        /**
            * Calculates the count of nodes among the last roup.size() nodes on the given level whose subtrees
            * have nodes not exceeding the radius limit.
            */
        template <typename SyncGroup>
        __device__ __host__ inline unsigned int get_done_node_count(
            SyncGroup &group)
        {
            const unsigned int node_count = buffer.get_node_count(level);
            const unsigned int index = node_count - min(node_count, group.size()) + group.thread_rank();
            const bool active = index < node_count;

            bool is_done = false;
            if (active)
            {
                is_done = buffer
                              .get_enumeration(level, index, mu.block(offset_kk(), offset_kk()), rdiag, pruning_bounds)
                              .template is_enumeration_done<dimensions_per_level - 1>();
            }
            return group.count(is_done);
        }

        /**
            * Removes all nodes among the last group.size() nodes on the given level whose subtrees have nodes
            * with partdist exceeding the radius limit. Be careful as the buffer can still have nodes
            * referencing such a done node as a parent node, since the enumeration data is updated when
            * children are generated, not when children are fully processed.
            */
        template <typename SyncGroup>
        __device__ __host__ inline void remove_done_nodes(
            SyncGroup &group)
        {
            const unsigned int node_count = buffer.get_node_count(level);
            const unsigned int active_thread_count = min(node_count, group.size());
            const unsigned int index = node_count - active_thread_count + group.thread_rank();

            bool is_done = buffer
                               .get_enumeration(level, index, mu.block(offset_kk(), offset_kk()), rdiag, pruning_bounds)
                               .template is_enumeration_done<dimensions_per_level - 1>();

            group.sync();

            buffer.filter_nodes(group, level, index, !is_done, active_thread_count);
        }

        /**
            * Generates children from the last group.size() nodes on level and adds them to the buffer, until
            * either the children buffer is full or most of these nodes are done.
            */
        template <typename SyncGroup>
        __device__ __host__ inline void process_inner_level(
            SyncGroup &group)
        {
            const unsigned begin_node_count = buffer.get_node_count(level + 1);
            const unsigned int active_thread_count = min(buffer.get_node_count(level), group.size());

            while (true)
            {
                generate_children(group);
                group.sync();
                const unsigned int done_node_count = get_done_node_count(group);
                group.sync();

                if (CUENUM_TRACE && thread_id() == 0)
                {
                    printf("Thread 0: Worked on level %d, next level points are %d, %d nodes of current working pool (%d) are done\n",
                           level, buffer.get_node_count(level + 1), done_node_count, active_thread_count);
                }

                if (buffer.get_node_count(level + 1) >= opts.max_children_buffer_size)
                {
                    break;
                }
                else if (done_node_count >= active_thread_count * (1 - opts.min_active_parents_percentage))
                {
                    break;
                }
                group.sync();
            }
        }

        template <typename SyncGroup, typename eval_sol_fn>
        __device__ __host__ inline void process_leaf_level(
            SyncGroup &group, eval_sol_fn &process_sol)
        {
            const unsigned int level = levels - 1;
            while (buffer.get_node_count(level) > 0)
            {
                consume_leaves(group, process_sol);
                remove_done_nodes(group);
                group.sync();
            }
        }

        /**
            * Removes finished nodes from the parent level of the given level. Does nothing when called on the
            * root.
            */
        template <typename SyncGroup>
        __device__ __host__ inline void cleanup_parent_level(
            SyncGroup &group)
        {
            if (level > 0)
            {
                group.sync();

                level -= 1;
                remove_done_nodes(group);
                level += 1;

                group.sync();

                if (CUENUM_TRACE && thread_id() == 0)
                {
                    printf("Thread 0: Cleaned up level %d, has now %d nodes\n", level - 1, buffer.get_node_count(level - 1));
                }

                group.sync();
            }
        }

        __device__ __host__ inline bool has_nodes()
        {
            return buffer.get_node_count(level) > 0;
        }

        __device__ __host__ inline bool is_inner_level() const
        {
            return level + 1 < levels;
        }
    };

    template <unsigned int levels, unsigned int dimensions_per_level, unsigned int max_nodes_per_level>
    template <typename SyncGroup>
    __device__ __host__ inline void TreeLevelEnumerator<levels, dimensions_per_level, max_nodes_per_level>::init_new_nodes(
        SyncGroup &group, unsigned int already_calculated_node_count)
    {
        for (unsigned int new_index = already_calculated_node_count + group.thread_rank();
             new_index < buffer.get_node_count(level); new_index += group.size())
        {
            unsigned int kk_offset = this->kk_offset();

            const unsigned int parent_index = buffer.get_parent_index(level, new_index);
            enumi x[dimensions_per_level];
            for (unsigned int j = 0; j < dimensions_per_level; ++j)
            {
                x[j] = buffer.get_coefficient(level, new_index, j);
            }

            // sets center_partsum[i] = parent_center_partsum[i] + calc_center_partsum_delta(..., i)
            // to reduce latency, the loop is transformed as to load data now that is needed after some loop
            // cycles (-> software pipelining)
            constexpr unsigned int loop_preload_count = 3;
            constexpr unsigned int loop_preload_offset = loop_preload_count - 1;
            unsigned int i = 0;
            enumf center_partsum;
            enumf preloaded_parent_center_partsums[loop_preload_count];

#pragma unroll
            for (unsigned int j = 0; j < loop_preload_offset; ++j)
            {
                preloaded_parent_center_partsums[j] = buffer.get_center_partsum(level - 1, parent_index, j);
            }

            for (; i + 2 * loop_preload_offset < kk_offset + dimensions_per_level; i += loop_preload_count)
            {
#pragma unroll
                for (unsigned int j = 0; j < loop_preload_count; ++j)
                {
                    preloaded_parent_center_partsums[(j + loop_preload_offset) % loop_preload_count] =
                        buffer.get_center_partsum(level - 1, parent_index, i + j + loop_preload_offset);

                    assert(preloaded_parent_center_partsums[j] ==
                           buffer.get_center_partsum(level - 1, parent_index, i + j));
                    center_partsum =
                        preloaded_parent_center_partsums[j] +
                        calc_center_partsum_delta<levels, dimensions_per_level>(level, new_index, i + j, x, mu);
                    buffer.set_center_partsum(level, new_index, i + j, center_partsum);
                }
            }

            assert(i + 2 * loop_preload_offset - loop_preload_count + 1 <=
                   kk_offset + dimensions_per_level);
            assert(i + 2 * loop_preload_offset >= kk_offset + dimensions_per_level);

#pragma unroll
            for (unsigned int ii = 2 * loop_preload_offset - loop_preload_count + 1;
                 ii <= 2 * loop_preload_offset; ++ii)
            {
                if (i + ii == kk_offset + dimensions_per_level)
                {
#pragma unroll
                    for (unsigned int j = 0; j < ii; ++j)
                    {
                        if (j + loop_preload_offset < ii)
                        {
                            preloaded_parent_center_partsums[(j + loop_preload_offset) % loop_preload_count] =
                                buffer.get_center_partsum(level - 1, parent_index, i + j + loop_preload_offset);
                        }
                        assert(preloaded_parent_center_partsums[j % loop_preload_count] ==
                               buffer.get_center_partsum(level - 1, parent_index, i + j));
                        center_partsum = preloaded_parent_center_partsums[j % loop_preload_count] +
                                         calc_center_partsum_delta<levels, dimensions_per_level>(level, new_index,
                                                                                                 i + j, x, mu);
                        buffer.set_center_partsum(level, new_index, i + j, center_partsum);
                    }
                }
            }

            buffer.init_enumeration(level, new_index);
        }
    }

    template <unsigned int levels, unsigned int dimensions_per_level, unsigned int max_nodes_per_level>
    template <typename SyncGroup>
    __device__ __host__ void TreeLevelEnumerator<levels, dimensions_per_level, max_nodes_per_level>::ensure_enumeration_initialized(
        SyncGroup &group, int target_level)
    {
        const unsigned int first_required_partsum_index = (levels - target_level - 1) * dimensions_per_level;
        unsigned int current_level = level;

        bool is_initialized;
        bool all_initialized;
        // search for the lowest level which has the required partsums set

        do
        {
            const unsigned int node_count = buffer.get_node_count(current_level);
            const unsigned int active_thread_count = min(node_count, group.size());
            const unsigned int index = node_count - active_thread_count + group.thread_rank();

            is_initialized = group.thread_rank() >= active_thread_count || buffer.are_partsums_initialized(current_level, index, target_level);
            all_initialized = group.all(is_initialized);

            current_level -= 1;
        } while (current_level != std::numeric_limits<unsigned int>::max() && !all_initialized);

        current_level += 1;
        // now this is the lowest level that has partsums set

        enumi x[dimensions_per_level];
        enumf partsums[dimensions_per_level];
        unsigned int parent_node_count;

        // load the parent data
        {
            parent_node_count = buffer.get_node_count(current_level);
            const unsigned int index = parent_node_count - min(parent_node_count, group.size()) + group.thread_rank();
            const unsigned int active = index < parent_node_count;

            if (active)
            {
                for (unsigned int i = 0; i < dimensions_per_level; ++i)
                {
                    partsums[i] = buffer.get_center_partsum(current_level, index, first_required_partsum_index + i);
                }
            }
            current_level += 1;
        }

        // now fill all currently uninitialized levels
        for (; current_level <= level; ++current_level)
        {
            const unsigned int node_count = buffer.get_node_count(current_level);
            const unsigned int index = node_count - min(node_count, group.size()) + group.thread_rank();
            const unsigned int active = index < node_count;

            unsigned int parent_index;
            if (active)
            {
                parent_index = buffer.get_parent_index(current_level, index);

                // load coefficients
                for (unsigned int j = 0; j < dimensions_per_level; ++j)
                {
                    x[j] = buffer.get_coefficient(current_level, index, j);
                }
            }
            else
            {
                parent_index = parent_node_count - 1;
            }
            const unsigned int lane_id_of_parent = min(parent_node_count, group.size()) + parent_index - parent_node_count;

            for (unsigned int i = 0; i < dimensions_per_level; ++i)
            {
                const enumf parent_center_partsum = group.shuffle(partsums[i], lane_id_of_parent);
                assert(!isnan(parent_center_partsum));

                if (active)
                {
                    partsums[i] = parent_center_partsum + calc_center_partsum_delta<levels, dimensions_per_level>(current_level, index, first_required_partsum_index + i, x, mu);
                    buffer.set_center_partsum(current_level, index, first_required_partsum_index + i, partsums[i]);
                }
                else
                {
                    partsums[i] = NAN;
                }
            }

            if (current_level == level && active)
            {
                buffer.init_enumeration(level, index);
            }

            parent_node_count = node_count;
        }
    }

    template <typename SyncGroup, typename eval_sol_fn, unsigned int levels,
              unsigned int dimensions_per_level, unsigned int max_nodes_per_level>
    __device__ __host__ inline void
    clear_level(SyncGroup &group, eval_sol_fn &evaluator,
                TreeLevelEnumerator<levels, dimensions_per_level, max_nodes_per_level> &enumerator)
    {
        enumerator.level = 0;
        while (enumerator.level >= 0)
        {
            if (enumerator.is_inner_level())
            {
                if (enumerator.has_nodes() > 0)
                {
                    enumerator.process_inner_level(group);
                    ++enumerator.level;
                }
                else
                {
                    enumerator.cleanup_parent_level(group);
                    --enumerator.level;
                }
            }
            else
            {
                enumerator.process_leaf_level(group, evaluator);
                enumerator.cleanup_parent_level(group);
                --enumerator.level;
            }
            group.sync();
        }
    }

    template <unsigned int levels, unsigned int dimensions_per_level, unsigned int max_nodes_per_level>
    struct Opts
    {
        StrategyOpts tree_clear_opts;
        unsigned int initial_nodes_per_group;
        unsigned int thread_count;
    };

    /**
        * Works together with all threads in the same warp to search the subtrees induced by start_points for
        * a shortest, nonzero lattice point.
        *
        * All parameters that are pointers should point to device memory
        *
        * @param pruning_bounds - memory of at least levels * dimensions_per_level entries containing ABSOLUTE
        * enumeration bounds per tree level. Note that from here on (i.e. in device code), absolute bounds are used,
        * while in host code, relative bounds are used
        */
    template <unsigned int levels, unsigned int dimensions_per_level,
              unsigned int max_nodes_per_level>
    __global__ void __launch_bounds__(enumerate_block_size, 4) enumerate_kernel(unsigned char *buffer_memory, const enumi *start_points,
                                                                                unsigned int *processed_start_point_counter, unsigned int start_point_count,
                                                                                unsigned int start_point_dim, const enumf *mu_ptr, const enumf *rdiag,
                                                                                const volatile enumf *pruning_bounds, uint64_t *perf_counter_memory,
                                                                                unsigned char *point_stream_memory,
                                                                                Opts<levels, dimensions_per_level, max_nodes_per_level> opts)
    {
        typedef ThreadGroupWarp<enumerate_block_size> SyncGroup;
        typedef SubtreeEnumerationBuffer<levels, dimensions_per_level, max_nodes_per_level> SubtreeBuffer;
        typedef PointStreamEvaluator<enumerate_point_stream_buffer_size> Evaluator;

        constexpr unsigned int dimensions = dimensions_per_level * levels;
        constexpr unsigned int group_count_per_block = enumerate_block_size / enumerate_cooperative_group_size;

        constexpr unsigned int mu_shared_memory_size = dimensions * dimensions * sizeof(enumf);
        constexpr unsigned int rdiag_shared_memory_size = dimensions * sizeof(enumf);
        constexpr unsigned int group_shared_counter_shared_memory_size = group_count_per_block * sizeof(unsigned int);
        // we use one evaluator per block, not per group
        constexpr unsigned int point_stream_shared_memory_size = sizeof(unsigned int);
        constexpr unsigned int shared_mem_size =
            mu_shared_memory_size + rdiag_shared_memory_size + group_shared_counter_shared_memory_size + point_stream_shared_memory_size;

        __shared__ unsigned char shared_mem[shared_mem_size];

        SyncGroup group = SyncGroup(cooperative_groups::this_thread_block());
        assert(group.size() == enumerate_cooperative_group_size);

        enumf *mu_shared = reinterpret_cast<enumf *>(shared_mem);
        enumf *rdiag_shared = reinterpret_cast<enumf *>(shared_mem + mu_shared_memory_size);
        unsigned int *group_shared_counter = reinterpret_cast<unsigned int *>(
            shared_mem + group.group_index_in_block() * sizeof(unsigned int) +
            mu_shared_memory_size + rdiag_shared_memory_size);
        unsigned int *point_stream_counter = reinterpret_cast<unsigned int *>(
            shared_mem + mu_shared_memory_size + rdiag_shared_memory_size + group_shared_counter_shared_memory_size);

        const unsigned int ldmu = dimensions + start_point_dim;
        for (unsigned int i = threadIdx.x; i < dimensions * dimensions; i += blockDim.x)
        {
            mu_shared[i] = mu_ptr[i / dimensions * ldmu + i % dimensions];
        }
        for (unsigned int i = threadIdx.x; i < dimensions; i += blockDim.x)
        {
            rdiag_shared[i] = rdiag[i];
        }
        __syncthreads();

        assert(opts.initial_nodes_per_group <= group.size());
        Matrix mu(mu_shared, dimensions);
        PerfCounter node_counter(perf_counter_memory);
        SubtreeBuffer buffer(buffer_memory + group.group_index() * SubtreeBuffer::memory_size_in_bytes);

        Evaluator process_sol(point_stream_memory + blockIdx.x * Evaluator::memory_size_in_bytes(dimensions + start_point_dim), point_stream_counter);
        TreeLevelEnumerator<levels, dimensions_per_level, max_nodes_per_level> enumerator(
            buffer,
            mu,
            rdiag_shared,
            node_counter,
            pruning_bounds,
            start_points,
            start_point_dim,
            opts.tree_clear_opts);

        while (true)
        {
            group.sync();
            if (group.thread_rank() == 0)
            {
                *group_shared_counter =
                    atomic_add(processed_start_point_counter, opts.initial_nodes_per_group);
            }
            buffer.init(group);
            group.sync();

            if (*group_shared_counter >= start_point_count)
            {
                break;
            }
            const unsigned int start_point_index = *group_shared_counter + group.thread_rank();
            const bool active =
                group.thread_rank() < opts.initial_nodes_per_group && start_point_index < start_point_count;
            if (active)
            {
                const enumi *start_point = &start_points[start_point_index * start_point_dim];
                const unsigned int index = buffer.add_node(0, start_point_index);
                for (unsigned int i = 0; i < dimensions; ++i)
                {
                    enumf center_partsum = 0;
                    for (unsigned int j = 0; j < start_point_dim; ++j)
                    {
                        center_partsum -= start_point[j] * mu_ptr[i * ldmu + dimensions + j];
                    }
                    assert(!isnan(center_partsum));
                    buffer.set_center_partsum(0, index, i, center_partsum);
                }
                enumf partdist = 0;
                for (int j = 0; j < start_point_dim; ++j)
                {
                    enumf alpha = start_point[j];
                    for (unsigned int i = j + 1; i < start_point_dim; ++i)
                    {
                        alpha += start_point[i] * mu_ptr[(j + dimensions) * ldmu + i + dimensions];
                    }
                    assert(rdiag[dimensions + j] >= 0);
                    partdist += alpha * alpha * rdiag[dimensions + j];
                    assert(partdist >= 0);
                }
                buffer.set_partdist(0, index, partdist);
                assert(start_point_index > 0 || partdist == 0);
                buffer.init_enumeration(0, index);
            }
            if (CUENUM_TRACE && thread_id() == 0)
            {
                printf("Thread 0: Get %d new nodes\n", opts.initial_nodes_per_group);
            }

            group.sync();

            clear_level<SyncGroup, Evaluator, levels, dimensions_per_level, max_nodes_per_level>(
                group, process_sol, enumerator);
        }
    }

    constexpr unsigned int get_grid_size(unsigned int thread_count)
    {
        return (thread_count - 1) / enumerate_block_size + 1;
    }

    constexpr unsigned int get_started_thread_count(unsigned int thread_count)
    {
        return get_grid_size(thread_count) * enumerate_block_size;
    }

    /**
        * Enumerates all points within the enumeration bound (initialized to parameter initial_radius) and calls
        * the given function on each coordinate of each of these points.
        *
        * For the description of the parameters, have the total lattice dimension n = levels * dimensions_per_level + start_point_dim.
        *
        * All parameters that are pointers should point to host memory (pinned memory is allowed).
        *
        * @param mu - pointer to memory containing the normalized gram-schmidt coefficients, in row-major format.
        * In other words, the memory must contain n consecutive batches of memory, each consisting of n entries storing
        * the values of the corresponding row of the matrix.
        * @param rdiag - n entries containing the squared norms of the gram-schmidt vectors, in one contigous segment of memory
        * @param start_points - Function yielding a pointer to memory containing the start_point_dim coefficients of the i-th start point. This
        * pointer must stay valid until the next call of the function.
        * @param process_sol - callback function called on solution points that were found. This is a host function.
        * @param pruning - array of at least n entries containing enumeration bounds RELATIVE to the global squared enumeration bound
        */
    template <unsigned int levels, unsigned int dimensions_per_level, unsigned int max_nodes_per_level, bool print_status = true>
    std::vector<uint64_t> enumerate(const enumf *mu, const enumf *rdiag, const enumi *start_points,
                                    unsigned int start_point_dim, unsigned int start_point_count, const enumf *pruning, enumf initial_radius,
                                    process_sol_fn process_sol,
                                    Opts<levels, dimensions_per_level, max_nodes_per_level> opts)
    {
        typedef SubtreeEnumerationBuffer<levels, dimensions_per_level, max_nodes_per_level> SubtreeBuffer;
        typedef PointStreamEvaluator<enumerate_point_stream_buffer_size> Evaluator;
        typedef PointStreamEndpoint<enumerate_point_stream_buffer_size> PointStream;

        constexpr unsigned int tree_dimensions = levels * dimensions_per_level;
        const unsigned int mu_n = tree_dimensions + start_point_dim;
        const unsigned int grid_size = get_grid_size(opts.thread_count);
        const unsigned int group_count = grid_size * enumerate_block_size / enumerate_cooperative_group_size;

        CudaPtr<unsigned char> buffer_mem =
            cuda_alloc(unsigned char, SubtreeBuffer::memory_size_in_bytes *group_count);
        CudaPtr<unsigned char> point_stream_memory =
            cuda_alloc(unsigned char, Evaluator::memory_size_in_bytes(mu_n) * grid_size);
        CudaPtr<enumf> pruning_bounds = cuda_alloc(enumf, mu_n);
        CudaPtr<enumf> device_mu = cuda_alloc(enumf, mu_n * mu_n);
        CudaPtr<enumf> device_rdiag = cuda_alloc(enumf, mu_n);
        CudaPtr<uint64_t> node_counter = cuda_alloc(uint64_t, mu_n);
        CudaPtr<enumi> device_start_points = cuda_alloc(enumi, start_point_count * start_point_dim);
        CudaPtr<unsigned int> processed_start_point_count = cuda_alloc(unsigned int, 1);

        check(cudaMemcpy(device_mu.get(), mu, mu_n * mu_n * sizeof(enumf), cudaMemcpyHostToDevice));
        check(cudaMemcpy(device_rdiag.get(), rdiag, mu_n * sizeof(enumf), cudaMemcpyHostToDevice));
        check(cudaMemcpy(device_start_points.get(), start_points,
                         start_point_dim * start_point_count * sizeof(enumi), cudaMemcpyHostToDevice));

        PointStream stream(point_stream_memory.get(), pruning_bounds.get(), node_counter.get(), pruning, initial_radius * initial_radius, grid_size, mu_n);
        stream.init();

        cudaEvent_t raw_event;
        check(cudaEventCreateWithFlags(&raw_event, cudaEventDisableTiming));
        CudaEvent event(raw_event);

        cudaStream_t raw_exec_stream;
        check(cudaStreamCreate(&raw_exec_stream));
        CudaStream exec_stream(raw_exec_stream);

        reset_profiling_counter();

        if (print_status)
        {
            std::cout << "Enumerating " << (levels * dimensions_per_level)
                      << " dimensional lattice using cuda, started " << grid_size << " block with "
                      << enumerate_block_size << " threads each; Beginning with " << start_point_count << " start points" << std::endl;
        }
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        enumerate_kernel<<<dim3(grid_size), dim3(enumerate_block_size), 0, exec_stream.get()>>>(
            buffer_mem.get(), device_start_points.get(), processed_start_point_count.get(),
            start_point_count, start_point_dim, device_mu.get(), device_rdiag.get(), pruning_bounds.get(),
            node_counter.get(), point_stream_memory.get(), opts);

        check(cudaEventRecord(event.get(), exec_stream.get()));

        std::chrono::steady_clock::time_point last_printed = std::chrono::steady_clock::now();
        while (cudaEventQuery(event.get()) != cudaSuccess)
        {
            stream.query_new_points<process_sol_fn, print_status>(process_sol);
            if ((std::chrono::steady_clock::now() - last_printed) > std::chrono::seconds(60))
            {
                stream.print_currently_searched_nodes();
                last_printed = std::chrono::steady_clock::now();
            }
        }
        stream.wait_for_event(event.get());
        stream.query_new_points<process_sol_fn, print_status>(process_sol);
		stream.print_currently_searched_nodes();

        check(cudaDeviceSynchronize());
        check(cudaGetLastError());

        std::array<uint64_t, tree_dimensions> searched_nodes;
        check(cudaMemcpy(&searched_nodes[0], node_counter.get(), tree_dimensions * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));

        if (print_status)
        {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            std::cout << "Enumeration done in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms"
                      << std::endl;

            enumf result_radius;
            check(cudaMemcpy(&result_radius, pruning_bounds.get(), sizeof(enumf), cudaMemcpyDeviceToHost));
            std::cout << "Searched " << std::accumulate(searched_nodes.begin(), searched_nodes.end(), static_cast<uint64_t>(0))
                      << " tree nodes, and decreased enumeration bound down to "
                      << sqrt(result_radius) << std::endl;
        }
        print_profiling_counter();

        return std::vector<uint64_t>(searched_nodes.begin(), searched_nodes.end());
    }

}

#endif