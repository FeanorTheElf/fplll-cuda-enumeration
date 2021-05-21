#include "enum.cuh"

namespace cuenum {

    template <unsigned int levels, unsigned int dimensions_per_level, unsigned int max_nodes_per_level, bool print_status = true>
    std::vector<uint64_t> enumerate_cpu(const enumf* mu_ptr, const enumf* rdiag, const enumi* start_points,
        unsigned int start_point_dim, unsigned int start_point_count, const enumf* pruning, enumf initial_radius,
        process_sol_fn process_sol,
        Opts<levels, dimensions_per_level, max_nodes_per_level> opts)
    {
        typedef single_thread CG;
        typedef SubtreeEnumerationBuffer<levels, dimensions_per_level, max_nodes_per_level> SubtreeBuffer;

        constexpr unsigned int block_size = 1;
        constexpr unsigned int dimensions = dimensions_per_level * levels;
        const unsigned int point_dimension = dimensions + start_point_dim;

        CG group;
        PrefixCounter<CG, block_size> prefix_counter;
        unsigned int counter = 0;
        unsigned int processed_start_point_counter = 0;
        uint64_t perf_counter[dimensions] = {};
        std::unique_ptr<unsigned char[]> buffer_memory(new unsigned char[SubtreeBuffer::memory_size_in_bytes]);
        std::unique_ptr<enumf[]> pruning_bounds(new enumf[point_dimension]);

        std::unique_ptr<enumi[]> solution_point(new enumi[point_dimension]);
        FnWrapper<enumi, unsigned int, enumf, unsigned int> evaluator([&solution_point, &process_sol, &pruning_bounds, &pruning]
        (enumi x, unsigned int coordinate, enumf norm_square, unsigned int point_dimension) {
            solution_point[coordinate] = x;
            if (coordinate == point_dimension - 1) {
                double bound = process_sol(norm_square, solution_point.get());
                for (unsigned int i = 0; i < point_dimension; ++i) {
                    pruning_bounds[i] = bound * pruning[i];
                }
            }
        });
        for (unsigned int i = 0; i < point_dimension; ++i) {
            pruning_bounds[i] = initial_radius * initial_radius * pruning[i];
        }

        const unsigned int ldmu = dimensions + start_point_dim;

        Matrix mu(mu_ptr, point_dimension);
        PerfCounter node_counter(&perf_counter[0]);
        SubtreeBuffer buffer(buffer_memory.get());

        std::vector<uint64_t> result;

        assert(opts.initial_nodes_per_group <= group.size());

        while (true)
        {
            group.sync();
            if (group.thread_rank() == 0)
            {
                counter = atomic_add(&processed_start_point_counter, opts.initial_nodes_per_group);
            }
            buffer.init(group);
            group.sync();

            if (counter >= start_point_count)
            {
                break;
            }
            const unsigned int start_point_index = counter + group.thread_rank();
            const bool active =
                group.thread_rank() < opts.initial_nodes_per_group && start_point_index < start_point_count;
            if (active)
            {
                const enumi* start_point = &start_points[start_point_index * start_point_dim];
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
                buffer.init_enumeration(0, index, partdist, buffer.get_center_partsum(0, index, dimensions - 1));
            }
            if (CUENUM_TRACE && thread_id() == 0)
            {
                printf("Thread 0: Get %d new nodes\n", opts.initial_nodes_per_group);
            }

            group.sync();

            clear_level(group, prefix_counter, &counter, buffer, 0, mu, rdiag,
                pruning_bounds.get(), evaluator, start_points, start_point_dim,
                opts.tree_clear_opts, node_counter);
        }

        return result;
    }

}
