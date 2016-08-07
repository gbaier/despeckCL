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

__kernel void precompute_filter_values (__global float * ampl_master, __global float * ampl_slave,  __global float * phase,
                                        __global float * filter_data_a, __global float * filter_data_x_real, __global float * filter_data_x_imag,
                                        const int height_overlap, const int width_overlap, const int patch_size)
{
    const int width_sws  = width_overlap  - patch_size + 1;
    const int height_sws = height_overlap - patch_size + 1;
    const int psh = (patch_size-1)/2;

    const int tx = get_group_id(0) * get_local_size(0) + get_local_id(0);
    const int ty = get_group_id(1) * get_local_size(1) + get_local_id(1);

    if (tx < height_sws && ty < width_sws) {
        const float a1 = ampl_master[ (tx+psh)*width_overlap + (ty+psh) ];
        const float a2 = ampl_slave [ (tx+psh)*width_overlap + (ty+psh) ];
        const float dp = phase     [ (tx+psh)*width_overlap + (ty+psh) ];

        filter_data_a      [tx*width_sws + ty] = 0.5f * ( pow(a1, 2.0f) + pow(a2, 2.0f) );
        filter_data_x_real [tx*width_sws + ty] = a1 * a2 * cos(-dp);
        filter_data_x_imag [tx*width_sws + ty] = a1 * a2 * sin(-dp);
    }
}
