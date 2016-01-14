__kernel void compute_weights (__global float * patch_similarities,
                               __global float * weights,
                               const int height_symm,
                               const int width_symm,
                               const int search_window_size,
                               const int patch_size,
                               __constant float * dissims2relidx,
                               __constant float * chi2cdf_inv,
                               const int lut_size,
                               const float dissims_min,
                               const float dissims_max)
{
    const float h = 16.0f;
    const float c = 49.0f;

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
