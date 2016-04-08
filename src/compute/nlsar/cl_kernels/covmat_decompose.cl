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

__kernel void covmat_decompose (__global float* covmat,
                                __global float* ampl_filt,
                                __global float* dphase_filt,
                                __global float* coh_filt,
                                const int height,
                                const int width)
{
    const int h = get_group_id(0) * get_local_size(0) + get_local_id(0);
    const int w = get_group_id(1) * get_local_size(1) + get_local_id(1);

    if ((h < height) && (w < width)) {
        const int idx = h*width + w;

        const float ampl_master = sqrt(covmat[idx]);
        const float ampl_slave  = sqrt(covmat[idx + 6*height*width]);

        ampl_filt   [idx] = ampl_master;
        dphase_filt [idx] =    atan2(covmat[idx + 3*height*width],              covmat[idx + 2*height*width]);
        coh_filt    [idx] = sqrt(pow(covmat[idx + 3*height*width], 2.0f) +  pow(covmat[idx + 2*height*width], 2.0f)) / (ampl_master*ampl_slave);
    }
}
