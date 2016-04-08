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

__kernel void slc2real(__global float* interf_real, 
                       __global float* interf_imag,
                       __global float* ampl,
                       __global float* dphase,
                       const int height,
                       const int width)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    const int idx = tx*width + ty;

    if(tx < height && ty < width) {
        ampl[idx]   = fabs(sqrt(pow(interf_real[idx], 2.0f) + pow(interf_imag[idx], 2.0f)));
        dphase[idx] = atan2(interf_imag[idx], interf_real[idx]);
    }
} 
