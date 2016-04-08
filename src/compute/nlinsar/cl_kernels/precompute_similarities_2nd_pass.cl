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

__kernel void precompute_similarities_2nd_pass (__global float * similarities, __global float * kullback_leiblers,
                                                const int height_overlap, const int width_overlap, const int border_max, const int search_window_size)
{
    const int h = get_global_id(0);
    const int w = get_global_id(1);

    const int wsh = (search_window_size - 1)/2;

    const int height_sim = height_overlap - search_window_size + 1;
    const int width_sim  = width_overlap  - search_window_size + 1;

    if( h < height_sim && \
        w < width_sim - border_max ) {
        for(int hh = 0; hh < wsh; hh++) {
            for(int ww = 0; ww < search_window_size; ww++) {
                const int idx = hh * search_window_size * height_sim * width_sim \
                              + ww * height_sim * width_sim \
                              + h * width_sim \
                              + w;

                // symmetric target pixel
                const int idx_sym = (search_window_size - 1 - hh) * search_window_size * height_sim * width_sim \
                                  + (search_window_size - 1 - ww) * height_sim * width_sim \
                                  + (h - (wsh - hh) ) * width_sim \
                                  + (w - (wsh - ww) );

                similarities     [idx] = similarities      [idx_sym];
                kullback_leiblers[idx] = kullback_leiblers [idx_sym];
            }
        }
    }
}
