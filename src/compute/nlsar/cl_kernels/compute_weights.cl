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

__kernel void compute_weights (__global float * patch_similarities,
                               __global float * weights,
                               const int height_symm,
                               const int width_symm,
                               const int search_window_size,
                               const int patch_size,
                               const float h,
                               const float c,
                               __constant float * dissims2relidx,
                               __constant float * chi2cdf_inv,
                               const int lut_size,
                               const float dissims_min,
                               const float dissims_max)
{
    const int tx = get_global_id(0);

    const int wsh = (search_window_size-1)/2;

    if( tx < (search_window_size*wsh + wsh)*height_symm*width_symm) {
        float dissim = patch_similarities[tx];

        if (dissim > dissims_max) {
            weights[tx] = 0.0f;
        } else  {
            dissim = max(dissim, dissims_min);
            dissim = min(dissim, dissims_max);

            // map dissimilarities to lookup table index
            const float mapped_idx = (dissim-dissims_min)/(dissims_max - dissims_min)*(lut_size-1);

            const float quantile = dissims2relidx[ (unsigned int) mapped_idx];
            const float x        = chi2cdf_inv[(unsigned int) (quantile * (lut_size-1))];
            const float weight = exp(-fabs(x-c)/h);

            weights[tx] = weight;
        }
    }
}
