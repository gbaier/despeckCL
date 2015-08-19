__kernel void covmat_spatial_avg (__global float * covmat_in,
                                  __global float * covmat_out, 
                                  const int dimension,
                                  const int height,
                                  const int width)
{
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    const int in_x = get_group_id(0) * OUTPUT_BLOCK_SIZE + tx;
    const int in_y = get_group_id(1) * OUTPUT_BLOCK_SIZE + ty;

    const int height_new = height - WINDOW_WIDTH + 1;
    const int width_new  = width  - WINDOW_WIDTH + 1;

    __local float local_data [BLOCK_SIZE][BLOCK_SIZE];

    for(int i = 0; i < (dimension * (dimension+1))/2; i++) {
        if ( (in_x < height) && (in_y < width) ) {
            local_data [tx][ty] = covmat_in [i*height*width + in_x*width + in_y];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        float sum = 0;

        if ((tx < OUTPUT_BLOCK_SIZE) && (ty < OUTPUT_BLOCK_SIZE)) {
            for(int kx = 0; kx < WINDOW_WIDTH; kx++) {
                for(int ky = 0; ky < WINDOW_WIDTH; ky++) {
                    sum += local_data[tx + kx][ty + ky];
                }
            }
            if (in_x < height_new &&  in_y < width_new) {
                covmat_out[i*height_new*width_new + in_x*width_new + in_y] = sum/(WINDOW_WIDTH*WINDOW_WIDTH);
            }
        }
    }
}
