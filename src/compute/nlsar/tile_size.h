/* Copyright 2015, 2016 Gerald Baier
 *
 * This file is part of despeckCL.
 *
 * despeckCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * despeckCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with despeckCL. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef TILE_SIZE_H
#define TILE_SIZE_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <vector>

namespace nlsar {
    int round_down(const int num, const int multiple);

    int tile_size(cl::Context context,
                  const int search_window_size,
                  const std::vector<int>& patch_sizes,
                  const std::vector<int>& scale_sizes);
}

#endif
