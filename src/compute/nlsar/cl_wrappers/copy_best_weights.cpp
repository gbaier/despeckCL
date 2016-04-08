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

#include "copy_best_weights.h"

void nlsar::copy_best_weights::run(cl::CommandQueue cmd_queue,
                                   cl::Buffer all_weights,
                                   cl::Buffer best_params,
                                   cl::Buffer best_weights,
                                   const int height_ori,
                                   const int width_ori,
                                   const int search_window_size)
{
    kernel.setArg( 0, all_weights);
    kernel.setArg( 1, best_params);
    kernel.setArg( 2, best_weights);
    kernel.setArg( 3, height_ori);
    kernel.setArg( 4, width_ori);
    kernel.setArg( 5, search_window_size);

    cl::NDRange global_size {(size_t) block_size*( (height_ori*width_ori-1)/block_size + 1)};
    cl::NDRange local_size  {block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
