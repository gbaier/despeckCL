// computes the following three figures:
// 1) nonlocal means intensities (the diagonal element of the covariance matrix in Eq. 14 
// 2) the weighted variances as in Eq. 16
// 3) the sum over all weights in the search window since they are reused later in Eq. 17
__kernel void compute_nl_statistics (__global float * covmat_ori,
                                     __global float * weights,
                                     __global float * intensities_nl,
                                     __global float * weighted_variances,
                                     __global float * weights_sums,
                                     const int height_ori,
                                     const int width_ori,
                                     const int search_window_size,
                                     const int patch_size_max,
                                     const int scale_width)
{
    const int height_overlap_avg = height_ori + search_window_size + patch_size_max + scale_width - 3;
    const int width_overlap_avg  = width_ori  + search_window_size + patch_size_max + scale_width - 3;

    const int wsh = (search_window_size - 1)/2;
    const int psh = (patch_size_max     - 1)/2;
    const int swh = (scale_width        - 1)/2;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    const int out_x = get_group_id(0) * get_local_size(0) + tx;
    const int out_y = get_group_id(1) * get_local_size(1) + ty;

    // step to always land at the real part of a diagonal element
    const int step_diag = DIMENSION*2 + 2;

    __local float intensities_local         [DIMENSION] [BLOCK_SIZE+SEARCH_WINDOW_SIZE] [BLOCK_SIZE+SEARCH_WINDOW_SIZE];
    __local float intensities_local_squared [DIMENSION] [BLOCK_SIZE+SEARCH_WINDOW_SIZE] [BLOCK_SIZE+SEARCH_WINDOW_SIZE];

    for(int x = 0; x<BLOCK_SIZE+SEARCH_WINDOW_SIZE; x += get_local_size(0)) {
        for(int y = 0; y<BLOCK_SIZE+SEARCH_WINDOW_SIZE; y += get_local_size(1)) {
            if ( (tx+x) < (BLOCK_SIZE+SEARCH_WINDOW_SIZE)
              && (ty+y) < (BLOCK_SIZE+SEARCH_WINDOW_SIZE)
              && (out_x+x) < height_overlap_avg
              && (out_y+y) < width_overlap_avg ) {
                for(int d = 0; d<DIMENSION; d++) {
                    const float intensity = covmat_ori[d*step_diag*height_overlap_avg*width_overlap_avg \
                                                                  + (out_x+x+psh+swh)*width_overlap_avg \
                                                                                    + (out_y+y+psh+swh) ];
                    intensities_local         [d][tx+x][ty+y] = intensity;
                    intensities_local_squared [d][tx+x][ty+y] = intensity*intensity;
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (out_x < height_ori && out_y < width_ori) {
        __private float weighted_variances_private [DIMENSION] = {0};
        __private float intensities_nl_private     [DIMENSION] = {0};
        __private float weights_sum_private = 0.0f;
        for(int x = 0; x<SEARCH_WINDOW_SIZE; x++ ) {
            for(int y = 0; y<SEARCH_WINDOW_SIZE; y++ ) {
                const float weight = weights[x * SEARCH_WINDOW_SIZE * height_ori * width_ori \
                                                                + y * height_ori * width_ori \
                                                                         + out_x * width_ori \
                                                                                     + out_y];
                weights_sum_private += weight;
                for(int d = 0; d<DIMENSION; d++) {
                    intensities_nl_private     [d] += weight * intensities_local         [d][tx+x][ty+y];
                    weighted_variances_private [d] += weight * intensities_local_squared [d][tx+x][ty+y];
                }

            }
        }
        weights_sums[out_x*width_ori + out_y] = weights_sum_private;
        for(int d = 0; d<DIMENSION; d++) {
            intensities_nl_private     [d] /= weights_sum_private;
            weighted_variances_private [d] /= weights_sum_private;
            const int idx = d*height_ori*width_ori \
                                 + out_x*width_ori \
                                           + out_y;
            weighted_variances [idx] = weighted_variances_private[d] - pow(intensities_nl_private[d], 2.0f);
            intensities_nl     [idx] = intensities_nl_private[d];
        }
    }
}
