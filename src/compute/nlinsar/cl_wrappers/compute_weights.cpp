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

#include "compute_weights.h"

#include <stdio.h>

#include "compute_weights.h"

void nlinsar::compute_weights::run(cl::CommandQueue cmd_queue,
                                   cl::Buffer patch_similarities,
                                   cl::Buffer patch_kullback_leiblers,
                                   cl::Buffer weights,
                                   const int n_elems,
                                   const float h,
                                   const float T)
{
    kernel.setArg(0, patch_similarities);
    kernel.setArg(1, patch_kullback_leiblers);
    kernel.setArg(2, weights);
    kernel.setArg(3, n_elems);
    kernel.setArg(4, h);
    kernel.setArg(5, T);

    cl::NDRange global_size {block_size*((n_elems - 1) / block_size + 1)};
    cl::NDRange local_size  {block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);

}
