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

__kernel void compute_weights (__global float * sims, __global float * kls, __global float * weights, const int size, const float h, const float T)
{
    const int tx = get_global_id(0);

    if( tx < size ) {
        float weight = exp(sims[tx] / h  +  kls[tx] / T);
        if (isnan(weight) || isinf(weight)) {
            weight = 0.0f;
        }
        weights[tx] = weight;
    }
}
