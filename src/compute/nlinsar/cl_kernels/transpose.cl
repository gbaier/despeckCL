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

__kernel void transpose(__global float * matrix, __global float * matrix_trans,
                        const int height, const int width)
{
    const int h = get_group_id(0) * THREAD_SIZE_COL + get_local_id(0);
    const int w = get_group_id(1) * THREAD_SIZE_COL + get_local_id(1);

    __local float block[THREAD_SIZE_COL][THREAD_SIZE_COL];

    for (int i = 0; i < THREAD_SIZE_COL; i += THREAD_SIZE_ROW) {
        if( (h+i) < height && w < width ) {
            block[get_local_id(0)+i][get_local_id(1)] = matrix[(h+i)*width + w];
        } else { // this should be unnecessary, since this block will not be copied back to matrix_trans
            block[get_local_id(0)+i][get_local_id(1)] = 0.0f;
        }
    }   

    barrier(CLK_LOCAL_MEM_FENCE);

    const int ht = get_group_id(1) * THREAD_SIZE_COL + get_local_id(0);
    const int wt = get_group_id(0) * THREAD_SIZE_COL + get_local_id(1);

    const int height_t = width;
    const int width_t = height;

    for (int i = 0; i < THREAD_SIZE_COL; i += THREAD_SIZE_ROW) {
        if( (ht+i) < height_t && wt < width_t ) {
            matrix_trans[(ht+i)*width_t + wt] = block[get_local_id(1)][get_local_id(0) + i];
        }
    }
}
