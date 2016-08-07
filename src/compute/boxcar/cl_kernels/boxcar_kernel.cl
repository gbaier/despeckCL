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

__kernel void boxcar_kernel (__global float * ampl_master, __global float * ampl_slave,  __global float * phase,
                             __global float * ref_filt,   __global float * phase_filt, __global float * coh_filt,
                             const int height, const int width)
{
    const int window_radius = (WINDOW_WIDTH - 1) / 2;
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    const int in_x = get_group_id(0) * OUTPUT_BLOCK_SIZE + tx;
    const int in_y = get_group_id(1) * OUTPUT_BLOCK_SIZE + ty;

    const int out_x = in_x + window_radius;
    const int out_y = in_y + window_radius;

    __local float ampl_master_data [BLOCK_SIZE][BLOCK_SIZE];
    __local float ampl_slave_data  [BLOCK_SIZE][BLOCK_SIZE];
    __local float phase_data      [BLOCK_SIZE][BLOCK_SIZE];

    if ( (in_x < height) && (in_y < width) ) {
        ampl_master_data [tx][ty] = ampl_master [in_x*width + in_y];
        ampl_slave_data  [tx][ty] = ampl_slave  [in_x*width + in_y];
        phase_data      [tx][ty] = phase      [in_x*width + in_y];
    } else {
        ampl_master_data [tx][ty] = 0;
        ampl_slave_data  [tx][ty] = 0;
        phase_data      [tx][ty] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float a = 0;
    int N = WINDOW_WIDTH*WINDOW_WIDTH;
    float x_real = 0;
    float x_imag = 0;

    if ((tx < OUTPUT_BLOCK_SIZE) && (ty < OUTPUT_BLOCK_SIZE)) {
        for(int kx = 0; kx < WINDOW_WIDTH; kx++) {
            for(int ky = 0; ky < WINDOW_WIDTH; ky++) {
                a              += (pow(ampl_master_data[tx + kx][ty + ky], 2) + pow(ampl_slave_data[tx + kx][ty + ky], 2))/2;
                const float aa  =      ampl_master_data[tx + kx][ty + ky]     *     ampl_slave_data[tx + kx][ty + ky]; 
                x_real         += aa * cos( phase_data[tx + kx][ty + ky] );
                x_imag         += aa * sin( phase_data[tx + kx][ty + ky] );
            }
        }
        if (out_x < height && out_y < width) {
            ref_filt   [out_x*width + out_y] = a/N;
            phase_filt [out_x*width + out_y] = atan2(x_imag, x_real);
            coh_filt    [out_x*width + out_y] = sqrt(pow(x_real, 2) + pow(x_imag, 2))/a;
        }
    }
}
