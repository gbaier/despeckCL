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

float enl_nobias(float enl, float alpha, float wsum) // Eq. 20 in Deledalle's NLSAR paper
{
    return enl/(pow(1.0f-alpha, 2.0f) + (pow(alpha, 2.0f) + 2.0f*alpha*(1.0f-alpha)/wsum)*enl);
    //return enl/(pow(1.0f-alpha, 2.0f) + enl*pow((1-alpha)/wsum+alpha, 2.0f));
}

__kernel void compute_enls_nobias (__global float * enls,
                                   __global float * alphas,
                                   __global float * wsums,
                                   __global float * enls_nobias,
                                   const int height_ori,
                                   const int width_ori)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    if (tx < height_ori && ty < width_ori) {
        const int idx = tx * width_ori + ty;
        enls_nobias[idx] = enl_nobias( enls[idx], alphas[idx], wsums[idx] );
    }
}
