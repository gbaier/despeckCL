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

#include "compute_enls_nobias.h"

void nlsar::compute_enls_nobias::run(cl::CommandQueue cmd_queue,
                                     cl::Buffer enls,
                                     cl::Buffer alphas,
                                     cl::Buffer wsums,
                                     cl::Buffer enls_nobias,
                                     const int height_ori,
                                     const int width_ori)
{
    kernel.setArg(0, enls);
    kernel.setArg(1, alphas);
    kernel.setArg(2, wsums);
    kernel.setArg(3, enls_nobias);
    kernel.setArg(4, height_ori);
    kernel.setArg(5, width_ori);

    cl::NDRange global_size {(size_t) block_size*( (height_ori - 1)/block_size + 1), \
                             (size_t) block_size*( (width_ori  - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
