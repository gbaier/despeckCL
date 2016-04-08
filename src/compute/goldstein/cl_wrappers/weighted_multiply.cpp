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

#include "weighted_multiply.h"

void goldstein::weighted_multiply::run(cl::CommandQueue cmd_queue,
                                       cl::Buffer interf_real,
                                       cl::Buffer interf_imag,
                                       const int height,
                                       const int width,
                                       const float alpha)
{
    kernel.setArg( 0, interf_real);
    kernel.setArg( 1, interf_imag);
    kernel.setArg( 2, height);
    kernel.setArg( 3, width);
    kernel.setArg( 4, alpha);

    cl::NDRange global_size {(size_t) block_size*( (width  - 1)/block_size + 1),
                             (size_t) block_size*( (height - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
