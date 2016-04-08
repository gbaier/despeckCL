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

#include "patches_unpack.h"

void goldstein::patches_unpack::run(cl::CommandQueue cmd_queue,
                                    cl::Buffer interf_real, 
                                    cl::Buffer interf_imag,
                                    cl::Buffer interf_real_unpacked, 
                                    cl::Buffer interf_imag_unpacked,
                                    const int height_unpacked,
                                    const int width_unpacked,
                                    const int patch_size,
                                    const int overlap)
{
    kernel.setArg( 0, interf_real);
    kernel.setArg( 1, interf_imag);
    kernel.setArg( 2, interf_real_unpacked);
    kernel.setArg( 3, interf_imag_unpacked);
    kernel.setArg( 4, height_unpacked);
    kernel.setArg( 5, width_unpacked);
    kernel.setArg( 6, patch_size);
    kernel.setArg( 7, overlap);

    cl::NDRange global_size {(size_t) block_size*( (width_unpacked  - 1)/block_size + 1),
                             (size_t) block_size*( (height_unpacked - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
