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

#include "patches_pack.h"

constexpr const char* goldstein::patches_pack::routine_name;
constexpr const char* goldstein::patches_pack::kernel_source;

void goldstein::patches_pack::run(cl::CommandQueue cmd_queue,
                                  cl::Buffer interf_real_unpacked, 
                                  cl::Buffer interf_imag_unpacked,
                                  cl::Buffer interf_real_packed, 
                                  cl::Buffer interf_imag_packed,
                                  const int height_unpacked,
                                  const int width_unpacked,
                                  const int patch_size,
                                  const int overlap)
{
    /* the unpacked patches are packed in four stages, so that adding up the overlaps
     * of all patches is workgroup save. The unpacked patches are patch with the
     * followich scheme. The scheme depends on the offset of the first patch
     *
     * 1 | 2 | 1 | 2 | 1 | 2 |
     * -----------------------
     * 3 | 4 | 3 | 4 | 3 | 4 | 
     * -----------------------
     * 1 | 2 | 1 | 2 | 1 | 2 |
     * -----------------------
     * 3 | 4 | 3 | 4 | 3 | 4 | 
     *
     ******************************************************/

    std::vector<std::pair<const int, const int>> offsets { {0, 0}, {0, patch_size}, {patch_size, 0}, {patch_size, patch_size} };
    
    kernel.setArg( 0, interf_real_unpacked);
    kernel.setArg( 1, interf_imag_unpacked);
    kernel.setArg( 2, interf_real_packed);
    kernel.setArg( 3, interf_imag_packed);
    kernel.setArg( 4, height_unpacked);
    kernel.setArg( 5, width_unpacked);
    kernel.setArg( 6, patch_size);
    kernel.setArg( 7, overlap);

//    std::cout << width_unpacked << " : " << height_unpacked << std::endl;

    for(auto offset : offsets) {
        const int x_offset = offset.first;
        const int y_offset = offset.second;

        // will only work if we have at least 2 patches in every direction, otherwise the nominator is negative
        cl::NDRange global_size {(size_t) patch_size*((width_unpacked  - x_offset - 1) / (2*patch_size)+1),
                                 (size_t) patch_size*((height_unpacked - y_offset - 1) / (2*patch_size)+1)};
        cl::NDRange local_size  {block_size, block_size};

        kernel.setArg( 8, x_offset);
        kernel.setArg( 9, y_offset);

//        std::cout << "offset: " << x_offset << ", " << y_offset << std::endl;
//        std::cout << "global_size: " << global_size[0] << ", " << global_size[1] << std::endl;

        cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL);
    }
}
