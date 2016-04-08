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

__kernel void compute_number_of_looks (__global float * weights,
                                       __global float * nols,
                                       const int height_ori,
                                       const int width_ori,
                                       const int search_window_size)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    const int swsh = (search_window_size-1)/2;

    float wsum  = 0.0f;
    float wsum2 = 0.0f;

    if (tx < height_ori && ty < width_ori) {
        for(int k = 0; k < search_window_size*search_window_size; k++) {
            const int idx = k * height_ori * width_ori + tx*width_ori + ty;
            const float weight = weights[idx];
            wsum += weight;
            wsum2 += weight*weight;
        }
        nols[tx*width_ori + ty] = wsum*wsum/wsum2;
    }
}
