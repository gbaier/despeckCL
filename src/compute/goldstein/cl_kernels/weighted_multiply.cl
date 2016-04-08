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

__kernel void weighted_multiply (__global float* interf_real,
                                 __global float* interf_imag,
                                 const int height,
                                 const int width,
                                 const float alpha)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    const int idx = tx*width + ty;

    if(tx < height && ty < width) {
        const float psd = sqrt(pow(interf_real[idx], 2.0f) + pow(interf_imag[idx], 2.0f));
        interf_real[idx] = pow(psd, alpha)*interf_real[idx];
        interf_imag[idx] = pow(psd, alpha)*interf_imag[idx];
    }
} 
