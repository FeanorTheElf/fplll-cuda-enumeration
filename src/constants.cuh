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

#ifndef FPLLL_CONSTANTS_CUH
#define FPLLL_CONSTANTS_CUH

#include <array>

namespace cuenum
{

	constexpr unsigned int enumerate_block_size = 128;
	constexpr unsigned int enumerate_cooperative_group_size = 32;
	constexpr unsigned int enumerate_point_stream_buffer_size = 1280;

	constexpr int cudaenum_max_dims_per_level = 4;
	constexpr int cudaenum_min_startdim = 6;
	constexpr unsigned int cudaenum_max_nodes_per_level = 3100;

	constexpr int cudaenum_max_levels(int dimensions_per_level)
	{
		return 77 / dimensions_per_level;
	};

}

#endif