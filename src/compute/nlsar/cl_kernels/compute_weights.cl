__kernel void compute_weights (__global float * patch_similarities,
                               __global float * weights,
                               const int height_ori,
                               const int width_ori,
                               const int search_window_size,
                               const int patch_size,
                               __constant float * dissims2relidx,
                               __constant float * chi2cdf_inv,
                               const int lut_size,
                               const float dissims_min,
                               const float dissims_max)
{
    //const float h = 0.33333f;
    const float h = 4.0f;
    const float c = 49.0f;

    const int tx = get_global_id(0);

    if( tx < height_ori*width_ori*search_window_size*search_window_size ) {
        float dissim = patch_similarities[tx];

        dissim = max(dissim, dissims_min);
        dissim = min(dissim, dissims_max);

        // map dissimilarities to lookup table index
        const float mapped_idx = (dissim-dissims_min)/(dissims_max - dissims_min)*lut_size;

        const float quantile = dissims2relidx[ (unsigned int) mapped_idx];
        const float x        = chi2cdf_inv[(unsigned int) (quantile * lut_size)];

        weights[tx] = max(0.000001f, exp(-fabs(x-c)/h));
    }
}
