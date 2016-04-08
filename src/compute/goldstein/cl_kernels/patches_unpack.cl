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

__kernel void patches_unpack(__global float* interf_real_packed,
                             __global float* interf_imag_packed,
                             __global float* interf_real_unpacked, 
                             __global float* interf_imag_unpacked,
                             const int height_unpacked,
                             const int width_unpacked,
                             const int patch_size,
                             const int overlap)
{
    const int width_packed  = (width_unpacked  / patch_size) * (patch_size-2*overlap);
    const int height_packed = (height_unpacked / patch_size) * (patch_size-2*overlap);

    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    const int patch_idx = tx / patch_size;
    const int patch_idy = ty / patch_size;

    const int rel_tx = tx % patch_size;
    const int rel_ty = ty % patch_size;

    const int idx_packed = min(width_packed  - 1, max(0, patch_idx*(patch_size-2*overlap) + (rel_tx - overlap)));
    const int idy_packed = min(height_packed - 1, max(0, patch_idy*(patch_size-2*overlap) + (rel_ty - overlap)));

    // no check necessary since we assume that the unpacked dimensions are fixed multiples of the block size
    interf_real_unpacked[ty*width_unpacked + tx] = interf_real_packed[idy_packed*width_packed + idx_packed];
    interf_imag_unpacked[ty*width_unpacked + tx] = interf_imag_packed[idy_packed*width_packed + idx_packed];
} 
