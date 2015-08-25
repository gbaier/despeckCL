__kernel void covmat_spatial_avg (__global float * covmat_in,
                                  __global float * covmat_out, 
                                  const int dimension,
                                  const int height_overlap,
                                  const int width_overlap)
{
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    const int in_x = get_group_id(0) * OUTPUT_BLOCK_SIZE + tx;
    const int in_y = get_group_id(1) * OUTPUT_BLOCK_SIZE + ty;

    const int height_overlap_avg = height_overlap + WINDOW_WIDTH - 1;
    const int width_overlap_avg  = width_overlap  + WINDOW_WIDTH - 1;

    __local float local_data [BLOCK_SIZE][BLOCK_SIZE];

    for(int i = 0; i < 2*dimension*dimension; i++) {
        if ( (in_x < height_overlap_avg) && (in_y < width_overlap_avg) ) {
            local_data [tx][ty] = covmat_in [i*height_overlap_avg*width_overlap_avg + in_x*width_overlap_avg + in_y];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        float sum = 0;

        if ((tx < OUTPUT_BLOCK_SIZE) && (ty < OUTPUT_BLOCK_SIZE)) {
            for(int kx = 0; kx < WINDOW_WIDTH; kx++) {
                for(int ky = 0; ky < WINDOW_WIDTH; ky++) {
                    sum += local_data[tx + kx][ty + ky];
                }
            }
            if (in_x < height_overlap &&  in_y < width_overlap) {
                covmat_out[i*height_overlap*width_overlap + in_x*width_overlap + in_y] = sum/(WINDOW_WIDTH*WINDOW_WIDTH);
            }
        }
    }
}
