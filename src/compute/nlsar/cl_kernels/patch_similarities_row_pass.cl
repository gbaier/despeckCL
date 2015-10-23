__kernel void patch_similarities_row_pass (__global float * pixel_similarities,
                                           __global float * intermed_row_avg,
                                           const int height_sim,
                                           const int width_sim,
                                           const int patch_size,
                                           const int patch_size_max,
                                           __local float * pixel_similarities_local)
{
    const int offset = (patch_size_max - patch_size) / 2;

    const int height_ori = height_sim - patch_size_max + 1;
    const int width_ori  = width_sim  - patch_size_max + 1;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tz = get_global_id(2);

    const int z_idx_c = tz*height_sim*width_sim;
    const int y_idx_c = (offset+get_global_id(1))*width_sim;
    const int x_idx_c = offset+get_group_id(0)*BLOCK_SIZE_X*STEPS_ROW;

    // load data into local memory
    // -1 and +1 are due to halo
    for(int b = -1; b < STEPS_ROW + 1; b++) {
        const int put_idx = ty*(BLOCK_SIZE_X*STEPS_ROW+2) + BLOCK_SIZE_X*(b+1) + tx;
        const int x_idx = x_idx_c + BLOCK_SIZE_X*b + tx;
        if(x_idx < 0 || x_idx >= width_sim || y_idx_c >= height_sim) { //halo or indices outside valid data
            pixel_similarities_local[put_idx] =  0;
        } else {
            pixel_similarities_local[put_idx] = pixel_similarities[z_idx_c + y_idx_c + x_idx];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int psh = (patch_size-1)/2;
    for(int b = 0; b < STEPS_ROW; b++) {
        float sum = 0;
        for(int k = -psh; k <= psh; k++) {
            const int idx = ty*(BLOCK_SIZE_X*STEPS_ROW+2) + BLOCK_SIZE_X*(b+1) + tx + k;
            sum += pixel_similarities_local[idx];
        }
        if (get_global_id(1) < height_sim && get_group_id(0)*BLOCK_SIZE_X*STEPS_ROW + BLOCK_SIZE_X*b + tx < width_ori) {
            const int idx = tz*height_ori*width_ori + \
                            get_global_id(1)*width_ori + \
                            get_group_id(0)*BLOCK_SIZE_X*STEPS_ROW + BLOCK_SIZE_X*b + tx;
            intermed_row_avg[idx] = sum;
        }
    }
}
