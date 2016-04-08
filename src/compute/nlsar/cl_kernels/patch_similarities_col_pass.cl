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

__kernel void patch_similarities_col_pass (__global float * intermed_row_avg,
                                           __global float * patch_similarities,
                                           const int height_sim,
                                           const int width_sim,
                                           const int patch_size,
                                           const int patch_size_max,
                                           __local float * intermed_row_avg_local)
{
    const int ori_offset = (patch_size - 1) / 2;
    const int offset = (patch_size_max - patch_size) / 2;

    const int height_ori = height_sim - patch_size_max + 1;
    const int width_ori  = width_sim  - patch_size_max + 1;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tz = get_global_id(2);

    const int z_idx_c = tz*(height_sim-2*offset)*width_ori;
    const int y_idx_c = get_group_id(1)*BLOCK_SIZE_Y*STEPS_COL + ty;
    const int x_idx_c = get_global_id(0);

    // load data into local memory
    // -2 and +2 are due to halo
    for(int b = -2; b < STEPS_COL + 2; b++) {
        const int put_idx = (ty+(b+2)*BLOCK_SIZE_Y)*BLOCK_SIZE_X + tx;
        const int y_idx = ori_offset + y_idx_c + BLOCK_SIZE_Y*b;
        if(y_idx < 0 || y_idx >= height_sim - 2*offset || x_idx_c >= width_ori) { //halo or indices outside valid data
            intermed_row_avg_local[put_idx] = 0;
        } else {
            intermed_row_avg_local[put_idx] = intermed_row_avg[z_idx_c + y_idx*width_ori + x_idx_c];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int psh = (patch_size-1)/2;
    for(int b = 0; b < STEPS_COL; b++) {
        float sum = 0;
        for(int k = -psh; k <= psh; k++) {
            const int idx = (ty+k+(b+2)*BLOCK_SIZE_Y)*BLOCK_SIZE_X + tx;
            sum += intermed_row_avg_local[idx];
        }
        const int y_idx = y_idx_c + BLOCK_SIZE_Y*b;
        if (x_idx_c < width_ori && y_idx < height_ori) {
            const int idx = tz*height_ori*width_ori + \
                                    y_idx*width_ori + \
                                            x_idx_c;
            patch_similarities[idx] = sum;
        }
    }
}
