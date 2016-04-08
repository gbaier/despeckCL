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

__kernel void compute_alphas (__global float * intensities_nl,
                              __global float * weighted_variances,
                              __global float * alphas,
                              const int height_ori,
                              const int width_ori,
                              const int dimension,
                              const int nlooks) // multilooking unrelated to nonlocal filtering
{
    const int tx = get_group_id(0) * get_local_size(0) + get_local_id(0);
    const int ty = get_group_id(1) * get_local_size(1) + get_local_id(1);

    float alpha = 0.0f;

    if (tx < height_ori && ty < width_ori) {
        const int idx = tx * width_ori + ty;
        for(int d=0; d<dimension; d++) {
            const float var    = weighted_variances [d*height_ori*width_ori + idx];
            const float int_nl = intensities_nl     [d*height_ori*width_ori + idx];
            alpha = max(alpha, max(0.0f, (var - int_nl*int_nl/nlooks)/var));
            //float alpha_new = fabs(var-int_nl*int_nl/nlooks);
            //alpha_new = alpha_new / (alpha_new + int_nl*int_nl/nlooks);
            //alpha = max(alpha, alpha_new);
        }
        alphas[tx*width_ori + ty] = alpha;
    }
}
