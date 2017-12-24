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

float correct_phase_range(float phi)
{
    if (phi > M_PI_F)
        phi -= 2.0f*M_PI_F;
    if (phi < -M_PI_F)
        phi += 2.0f*M_PI_F;
    return phi;
}

float correct_coherence_range(float coh)
{
    if (coh > 1.0f)
        coh = 1.0f;
    if (coh < 0.0f)
        coh = 0.0f;
    return coh;
}

bool is_valid(float number)
{
    bool valid;
    if ( isnan(number) || isinf(number) || (number == 0) ){
        valid = false;
    } else {
        valid = true;
    }
    return valid;
}

__kernel void compute_insar  (__global float * filter_data_a, __global float * filter_data_x_real, __global float * filter_data_x_imag,
                              __global float * ref_filt, __global float * phi_filt, __global float * coh_filt,
                              __global float * weights, const int height_ori, const int width_ori,
                              const int search_window_size, const int patch_size)
{
    const int height_overlap = height_ori + search_window_size + patch_size - 2;
    const int width_overlap  = width_ori  + search_window_size + patch_size - 2;
    const int height_sws = height_ori + search_window_size - 1;
    const int width_sws  = width_ori  + search_window_size - 1;

    const int psh = (patch_size-1)/2;
    const int wsh = (search_window_size-1)/2;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    const int out_x = get_group_id(0) * get_local_size(0) + tx;
    const int out_y = get_group_id(1) * get_local_size(1) + ty;

    __local float a_local     [BLOCK_SIZE+SEARCH_WINDOW_SIZE][BLOCK_SIZE+SEARCH_WINDOW_SIZE];
    __local float x_real_local[BLOCK_SIZE+SEARCH_WINDOW_SIZE][BLOCK_SIZE+SEARCH_WINDOW_SIZE];
    __local float x_imag_local[BLOCK_SIZE+SEARCH_WINDOW_SIZE][BLOCK_SIZE+SEARCH_WINDOW_SIZE];

    for(int x = 0; x<BLOCK_SIZE+SEARCH_WINDOW_SIZE; x += get_local_size(0)) {
        for(int y = 0; y<BLOCK_SIZE+SEARCH_WINDOW_SIZE; y += get_local_size(1)) {
            if ( (tx+x) < (BLOCK_SIZE+SEARCH_WINDOW_SIZE) && (ty+y) < (BLOCK_SIZE+SEARCH_WINDOW_SIZE) && (out_x+x) < height_sws && (out_y+y) < width_sws ) {
                a_local      [tx+x][ty+y] = filter_data_a      [ (out_x+x)*width_sws + (out_y+y) ];
                x_real_local [tx+x][ty+y] = filter_data_x_real [ (out_x+x)*width_sws + (out_y+y) ];
                x_imag_local [tx+x][ty+y] = filter_data_x_imag [ (out_x+x)*width_sws + (out_y+y) ];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float N_temp = 0.0f;
    float a_temp = 0.0f;
    float x_real_temp = 0.0f;
    float x_imag_temp = 0.0f;

    if (out_x < height_ori && out_y < width_ori) {
        for(int x = 0; x<SEARCH_WINDOW_SIZE; x++ ) {
            for(int y = 0; y<SEARCH_WINDOW_SIZE; y++ ) {
                // FIXME maybe there is a way to make use of coalesce memory access
                const float weight = weights[ out_x * width_ori * SEARCH_WINDOW_SIZE * SEARCH_WINDOW_SIZE \
                                                        + out_y * SEARCH_WINDOW_SIZE * SEARCH_WINDOW_SIZE \
                                                                                 + x * SEARCH_WINDOW_SIZE \
                                                                                                      + y ];
                N_temp      += weight;
                a_temp      += weight * a_local     [tx+x][ty+y];
                x_real_temp += weight * x_real_local[tx+x][ty+y];
                x_imag_temp += weight * x_imag_local[tx+x][ty+y];
            }
        }
        const float reflectivity          = a_temp/N_temp;
        const float interferometric_phase = atan2(x_imag_temp, x_real_temp);
        const float coherence             = sqrt(pow(x_real_temp, 2.0f) + pow(x_imag_temp, 2.0f))/a_temp;
        const int idx = (out_x+wsh+psh)*width_overlap + (out_y+wsh+psh);
        // I noticed that after some iterations, some computed values are invalid, probably
        // since the weights are zero, due to the Kullback-Leibler-divergence
        // in this case we just keep the previoulsy estimated value
        if (is_valid(reflectivity)) {
            ref_filt[idx] = reflectivity;
        }
        if (is_valid(interferometric_phase)) {
            phi_filt[idx] = correct_phase_range(interferometric_phase);
        }
        if (is_valid(coherence)) {
            coh_filt[idx] = correct_coherence_range(coherence);
        }
    }
}
