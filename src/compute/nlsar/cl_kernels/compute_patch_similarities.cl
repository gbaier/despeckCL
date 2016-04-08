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

__kernel void compute_patch_similarities (__global float * pixel_similarities,
                                          __global float * patch_similarities,
                                          const int height_sim,
                                          const int width_sim,
                                          const int patch_size,
                                          const int patch_size_max,
                                          __local float * pixel_similarities_local,
                                          __local float * cache)
{
    const int offset = (patch_size_max - patch_size) / 2;

    const int height_ori = height_sim - patch_size_max + 1;
    const int width_ori  = width_sim  - patch_size_max + 1;

    const int output_block_size = get_local_size(0) - patch_size + 1;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tz = get_global_id(2);

    const int out_x = get_group_id(0) * output_block_size + tx;
    const int out_y = get_group_id(1) * output_block_size + ty;

    if ( (out_x < height_sim) && (out_y < width_sim) ) {
        pixel_similarities_local[tx*get_local_size(1) + ty] = pixel_similarities[tz*height_sim*width_sim + (offset+out_x)*width_sim + offset + out_y];
    } else {
        pixel_similarities_local[tx*get_local_size(1) + ty] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;

    // average over 2nd dimension
    if (ty < output_block_size) {
        for(int ky = 0; ky < patch_size; ky++) {
            sum += pixel_similarities_local[tx*get_local_size(1) + ty + ky];
        }
        cache[tx*output_block_size+ty] = sum;
    }
    // use the intermediate results for the 2nd dimension to average over the 1st
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tx < output_block_size) {
        for(int kx = 1; kx < patch_size; kx++) {
            sum += cache[(tx+kx)*output_block_size + ty];
        }
    }

    if ((tx < output_block_size) && (ty < output_block_size)) {
        if (out_x < height_ori && out_y < width_ori) {
            patch_similarities[tz*height_ori*width_ori + out_x*width_ori + out_y] = sum;
        }
    }
}
