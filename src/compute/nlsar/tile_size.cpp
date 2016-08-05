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

#include "tile_size.h"

#include <algorithm>
#include <omp.h>
#include <cmath>

#include "logging.h"
#include "easylogging++.h"

int nlsar::round_down(const int num, const int multiple)
{
     int remainder = num % multiple;
     return num - remainder;
}

int nlsar::tile_size(cl::Context context,
                     const int search_window_size,
                     const std::vector<int>& patch_sizes,
                     const std::vector<int>& scale_sizes)
{
    const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
    const int scale_size_max = *std::max_element(scale_sizes.begin(), scale_sizes.end());

    const int overlap = search_window_size + patch_size_max + scale_size_max - 3;

    const int n_params = patch_sizes.size() * scale_sizes.size();
    VLOG(0) << "number of parameters = " << n_params;

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::Device dev = devices[0];

    int long global_mem_size;
    dev.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem_size);
    VLOG(0) << "global memory size = " << global_mem_size;

    int long max_mem_alloc_size;
    dev.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_mem_alloc_size);
    VLOG(0) << "maximum memory allocation size = " << max_mem_alloc_size;

    const int n_threads = omp_get_max_threads();
    VLOG(0) << "number of threads = " << n_threads;

    // Most the most memory is required for storing weights, so only this is taken into account.
    // required bytes per pixel = reg_bpp
    const int req_bpp = 4 * search_window_size * search_window_size * n_params;
    const int n_pixels_global = global_mem_size    / (req_bpp * n_threads);
    const int n_pixels_alloc  = max_mem_alloc_size /  req_bpp;
    const int n_pixels = std::min(n_pixels_global, n_pixels_alloc);

    const int tile_size_fit        = std::sqrt(n_pixels);
    const int tile_size_fit_rounded = round_down(tile_size_fit, 64);

    VLOG(0) << "tile_size_fit = "         << tile_size_fit;
    VLOG(0) << "tile_size_fit_rounded = " << tile_size_fit_rounded;

    const float safety_factor = 0.90;
    int safe_tile_size = 0;
    if (tile_size_fit_rounded < safety_factor*tile_size_fit) {
        safe_tile_size = tile_size_fit_rounded;
    } else {
        safe_tile_size = tile_size_fit_rounded - 64;
    }
    VLOG(0) << "safe_sub_image_size = " << safe_tile_size;
    safe_tile_size += overlap;
    VLOG(0) << "safe_sub_image_size with overlap = " << safe_tile_size;


    return safe_tile_size;
}
