#ifndef FPLLL_CONSTANTS_CUH
#define FPLLL_CONSTANTS_CUH

#include <array>

namespace cuenum {

constexpr unsigned int enumerate_block_size               = 128;
constexpr unsigned int enumerate_cooperative_group_size   = 32;
constexpr unsigned int enumerate_point_stream_buffer_size = 128;

constexpr int cudaenum_max_dims_per_level  = 4;
constexpr int cudaenum_min_startdim        = 6;
constexpr unsigned int cudaenum_max_nodes_per_level = 3100;

constexpr int cudaenum_max_levels(int dimensions_per_level) {
	return 77 / dimensions_per_level;
};

}

#endif