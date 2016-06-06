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

__kernel void weighted_means (__global float * covmat_in,
                              __global float * covmat_out,
                              __global float * weights,
                              __global float * alphas,
                              const int height_ori,
                              const int width_ori,
                              const int search_window_size,
                              const int patch_size,
                              const int window_width)
{
    const int height_overlap = height_ori + search_window_size + patch_size - 2;
    const int width_overlap  = width_ori  + search_window_size + patch_size - 2;

    const int height_overlap_avg = height_overlap + window_width - 1;
    const int width_overlap_avg  = width_overlap  + window_width - 1;

    const int wsh = (search_window_size - 1)/2;
    const int psh = (patch_size - 1)/2;
    const int wwh = (window_width -1)/2;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    const int out_x = get_group_id(0) * get_local_size(0) + tx;
    const int out_y = get_group_id(1) * get_local_size(1) + ty;

    __local float covmat_local [BLOCK_SIZE+SEARCH_WINDOW_SIZE] [BLOCK_SIZE+SEARCH_WINDOW_SIZE];

    for(int d = 0; d<2*DIMENSION*DIMENSION; d++) {
    for(int x = 0; x<BLOCK_SIZE+SEARCH_WINDOW_SIZE; x += get_local_size(0)) {
        for(int y = 0; y<BLOCK_SIZE+SEARCH_WINDOW_SIZE; y += get_local_size(1)) {
            if ( (tx+x) < (BLOCK_SIZE+SEARCH_WINDOW_SIZE)
              && (ty+y) < (BLOCK_SIZE+SEARCH_WINDOW_SIZE)
              && (out_x+x) < height_overlap
              && (out_y+y) < width_overlap ) {
                    covmat_local [tx+x][ty+y] = covmat_in[d*height_overlap_avg*width_overlap_avg \
                                                           + (out_x+x+wwh+psh)*width_overlap_avg \
                                                                             + (out_y+y+wwh+psh) ];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (out_x < height_ori && out_y < width_ori) {
        float covmat_new = 0.0f;
        float weight_sum = 0.0f;
        const float alpha = alphas[out_x * width_ori + out_y];
        for(int x = 0; x<SEARCH_WINDOW_SIZE; x++ ) {
            for(int y = 0; y<SEARCH_WINDOW_SIZE; y++ ) {
                const float weight = weights[x * SEARCH_WINDOW_SIZE * height_ori * width_ori \
                                                                + y * height_ori * width_ori \
                                                                         + out_x * width_ori \
                                                                                     + out_y];
                weight_sum += weight;
                covmat_new += weight * covmat_local[tx+x][ty+y];
            }
        }
            const int out_idx = d*height_overlap_avg*width_overlap_avg \
                               + (wsh+psh+wwh+out_x)*width_overlap_avg \
                                                  + (wsh+psh+wwh+out_y);
            covmat_out[out_idx] = (1-alpha)*covmat_new/weight_sum + alpha*covmat_local[tx+wsh][ty+wsh];
        }
    barrier(CLK_LOCAL_MEM_FENCE);
    }
}
