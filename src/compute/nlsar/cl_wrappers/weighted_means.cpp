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

#include "weighted_means.h"

#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>

nlsar::weighted_means::weighted_means(const size_t block_size,
                                      cl::Context context,
                                      const int search_window_size,
                                      const int dimension) : kernel_env_single<weighted_means>(block_size, context),
                                                             search_window_size(search_window_size),
                                                             dimension(dimension)
{
    program = build_program(build_opts(), kernel_source);
    kernel  = build_kernel(program, routine_name);
}

nlsar::weighted_means::weighted_means(const weighted_means& other) : kernel_env_single<weighted_means>(other),
                                                                     search_window_size(other.search_window_size),
                                                                     dimension(other.dimension)
{
    program = other.program;
    kernel  = build_kernel(program, routine_name);
}

std::string nlsar::weighted_means::build_opts()
{
    std::ostringstream out;
    out << " -D SEARCH_WINDOW_SIZE=" << search_window_size << " -D BLOCK_SIZE=" << block_size << " -D DIMENSION=" << dimension;
    return default_build_opts() + out.str();
}

void nlsar::weighted_means::run(cl::CommandQueue cmd_queue,
                                cl::Buffer covmat_in,
                                cl::Buffer covmat_out,
                                cl::Buffer weights,
                                cl::Buffer alphas,
                                const int height_ori,
                                const int width_ori,
                                const int search_window_size,
                                const int patch_size,
                                const int window_width)
{
    kernel.setArg(0, covmat_in);
    kernel.setArg(1, covmat_out);
    kernel.setArg(2, weights);
    kernel.setArg(3, alphas);
    kernel.setArg(4, height_ori);
    kernel.setArg(5, width_ori);
    kernel.setArg(6, search_window_size);
    kernel.setArg(7, patch_size);
    kernel.setArg(8, window_width);

    cl::NDRange global_size {(size_t) block_size*( (height_ori - 1)/block_size + 1), \
                             (size_t) block_size*( (width_ori  - 1)/block_size + 1)};

    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
