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

__kernel void precompute_patch_similarities (__global float * in, __global float * out,
                                             const int height_ori, const int width_ori,
                                             const int patch_size)
{
    const int height_sim = height_ori + patch_size - 1;
    const int width_sim  = width_ori  + patch_size - 1;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tz = get_global_id(2);

    const int out_x = get_group_id(0) * OUTPUT_BLOCK_SIZE + tx;
    const int out_y = get_group_id(1) * OUTPUT_BLOCK_SIZE + ty;

    __local float in_local[BLOCK_SIZE][BLOCK_SIZE];
    __local float intermed_local[BLOCK_SIZE][OUTPUT_BLOCK_SIZE];

    if ( (out_x < height_sim) && (out_y < width_sim) ) {
        in_local[tx][ty] = in[tz*height_sim*width_sim + out_x*width_sim + out_y];
    } else {
        in_local[tx][ty] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;

    // average over 2nd dimension
    if (ty < OUTPUT_BLOCK_SIZE) {
        for(int ky = 0; ky < patch_size; ky++) {
            sum += in_local[tx][ty + ky];
        }
        intermed_local[tx][ty] = sum;
    }
    // use the intermediate results for the 2nd dimension to average over the 1st
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tx < OUTPUT_BLOCK_SIZE) {
        for(int kx = 1; kx < patch_size; kx++) {
            sum += intermed_local[tx+kx][ty];
        }
    }

    if ((tx < OUTPUT_BLOCK_SIZE) && (ty < OUTPUT_BLOCK_SIZE)) {
        if (out_x < height_ori && out_y < width_ori) {
            out[tz*height_ori*width_ori + out_x*width_ori + out_y] = sum;
        }
    }
}
