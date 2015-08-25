__kernel void weighted_means (__global float * covmat_in,
                              __global float * covmat_out,
                              __global float * weights,
                              const int height_ori,
                              const int width_ori,
                              const int search_window_size,
                              const int patch_size,
                              const int window_width)
{
    const int height_overlap = height_ori + search_window_size + patch_size - 2;
    const int width_overlap  = width_ori  + search_window_size + patch_size - 2;

    const int height_overlap_avg = height_overlap + window_width - 1;
    const int width_overlap_avg  = width_overlap  + window_width - 1;

    const int wsh = (search_window_size - 1)/2;
    const int psh = (patch_size - 1)/2;
    const int wwh = (window_width -1)/2;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    const int out_x = get_group_id(0) * get_local_size(0) + tx;
    const int out_y = get_group_id(1) * get_local_size(1) + ty;

    __local float covmat_local [2*DIMENSION*DIMENSION] [BLOCK_SIZE+SEARCH_WINDOW_SIZE] [BLOCK_SIZE+SEARCH_WINDOW_SIZE];

    for(int x = 0; x<BLOCK_SIZE+SEARCH_WINDOW_SIZE; x += get_local_size(0)) {
        for(int y = 0; y<BLOCK_SIZE+SEARCH_WINDOW_SIZE; y += get_local_size(1)) {
            if ( (tx+x) < (BLOCK_SIZE+SEARCH_WINDOW_SIZE)
              && (ty+y) < (BLOCK_SIZE+SEARCH_WINDOW_SIZE)
              && (out_x+x) < height_overlap
              && (out_y+y) < width_overlap ) {
                for(int d = 0; d<2*DIMENSION*DIMENSION; d++) {
                    covmat_local [d][tx+x][ty+y] = covmat_in[d*height_overlap_avg*width_overlap_avg \
                                                                + (out_x+x+2*psh)*width_overlap_avg \
                                                                                 + (out_y+y+2*psh) ];
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (out_x < height_ori && out_y < width_ori) {
        float covmat_new[2*DIMENSION*DIMENSION] = {0};
        float weight_sum = 0.0f;
        for(int x = 0; x<SEARCH_WINDOW_SIZE; x++ ) {
            for(int y = 0; y<SEARCH_WINDOW_SIZE; y++ ) {
                // FIXME maybe there is a way to make use of coalesce memory access
                const float weight = weights[ out_x * width_ori * SEARCH_WINDOW_SIZE * SEARCH_WINDOW_SIZE \
                                                        + out_y * SEARCH_WINDOW_SIZE * SEARCH_WINDOW_SIZE \
                                                                                 + x * SEARCH_WINDOW_SIZE \
                                                                                                      + y ];
                weight_sum += weight;
                for(int d = 0; d<2*DIMENSION*DIMENSION; d++) {
                    covmat_new[d] += weight * covmat_local[d][tx+x][ty+y];
                }

            }
        }
        for(int d = 0; d<2*DIMENSION*DIMENSION; d++) {
            covmat_out[d*height_overlap_avg*width_overlap_avg \
                      + (wsh+psh+wwh+out_x)*width_overlap_avg \
                                         + (wsh+psh+wwh+out_y)] = covmat_new[d]/weight_sum;
        }
    }
}
