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

__kernel void patches_pack(__global float* interf_real_unpacked, 
                           __global float* interf_imag_unpacked,
                           __global float* interf_real_packed,
                           __global float* interf_imag_packed,
                           const int height_unpacked,
                           const int width_unpacked,
                           const int patch_size,
                           const int overlap,
                           const int offset_x,
                           const int offset_y)
{
    const int width_packed  = width_unpacked  / patch_size * (patch_size-2*overlap);
    const int height_packed = height_unpacked / patch_size * (patch_size-2*overlap);

    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    // factor of two is for striding, so that we take only every second patch
    const int patch_idx = 2*(tx/patch_size) + offset_x/patch_size;
    const int patch_idy = 2*(ty/patch_size) + offset_y/patch_size;

    const int rel_tx = tx % patch_size;
    const int rel_ty = ty % patch_size;

    const int in_idx = patch_idx*patch_size + rel_tx;
    const int in_idy = patch_idy*patch_size + rel_ty;

    const int out_idx = patch_idx*(patch_size-2*overlap) + (rel_tx - overlap);
    const int out_idy = patch_idy*(patch_size-2*overlap) + (rel_ty - overlap);

    float scaling_factor = 1.0f;
    if(rel_tx < 2*overlap || rel_tx >= patch_size - 2*overlap) {
        scaling_factor *= 0.5f;
    }
    if(rel_ty < 2*overlap || rel_ty >= patch_size - 2*overlap) {
        scaling_factor *= 0.5f;
    }
    if(out_idx < overlap || out_idx >= width_packed - overlap) {
        scaling_factor *= 2.0f;
    }
    if(out_idy < overlap || out_idy >= height_packed - overlap) {
        scaling_factor *= 2.0f;
    }
    
    const float val_real = scaling_factor * interf_real_unpacked[in_idy*width_unpacked + in_idx];
    const float val_imag = scaling_factor * interf_imag_unpacked[in_idy*width_unpacked + in_idx];

    // no check necessary since we assume that the unpacked dimensions are fixed multiples of the block size
    if (out_idx >= 0 && out_idx < width_packed && \
        out_idy >= 0 && out_idy < height_packed) {
        interf_real_packed[out_idy*width_packed + out_idx] += val_real;
        interf_imag_packed[out_idy*width_packed + out_idx] += val_imag;
    }
}
