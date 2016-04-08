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

#include "precompute_filter_values.h"

void nlinsar::precompute_filter_values::run(cl::CommandQueue cmd_queue,
                                            cl::Buffer device_a1,
                                            cl::Buffer device_a2,
                                            cl::Buffer device_dp,
                                            cl::Buffer device_filter_values_a,
                                            cl::Buffer device_filter_values_x_real,
                                            cl::Buffer device_filter_values_x_imag,
                                            const int height_overlap,
                                            const int width_overlap,
                                            const int patch_size)
{
    const int height_sws = height_overlap - patch_size + 1;
    const int width_sws  = width_overlap  - patch_size + 1;

    kernel.setArg(0, device_a1);
    kernel.setArg(1, device_a2);
    kernel.setArg(2, device_dp);

    kernel.setArg(3, device_filter_values_a);
    kernel.setArg(4, device_filter_values_x_real);
    kernel.setArg(5, device_filter_values_x_imag);

    kernel.setArg(6, height_overlap);
    kernel.setArg(7, width_overlap);
    kernel.setArg(8, patch_size);

    cl::NDRange global_size {(size_t) block_size*((height_sws-1)/block_size+1), (size_t) block_size*((width_sws-1)/block_size+1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);

    cmd_queue.finish();
}
