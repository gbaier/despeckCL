__kernel void patch_similarities_row_pass (__global float * pixel_similarities,
                                           __global float * intermed_row_avg,
                                           const int height_sim,
                                           const int width_sim,
                                           const int patch_size,
                                           const int patch_size_max,
                                           __local float * pixel_similarities_local)
{
    const int offset = (patch_size_max - patch_size) / 2;
    const int ori_offset = (patch_size_max - 1) / 2;

    const int height_ori = height_sim - patch_size_max + 1;
    const int width_ori  = width_sim  - patch_size_max + 1;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tz = get_global_id(2);

    const int z_idx_c = tz*height_sim*width_sim;
    const int y_idx_c = offset+get_global_id(1);
    const int x_idx_c = ori_offset+get_group_id(0)*BLOCK_SIZE_X*STEPS_ROW;

    // load data into local memory
    // +2 is for the left and right halo
    for(int b = 0; b < STEPS_ROW + 2; b++) {
        const int put_idx = ty*(BLOCK_SIZE_X*STEPS_ROW+2) + BLOCK_SIZE_X*b + tx;
        const int x_idx = x_idx_c + BLOCK_SIZE_X*(b-1) + tx;
        // FIXME there should be a test for the maximum dimension of x_idx: x_idx >= width_sim
        if(x_idx < 0 || y_idx_c >= height_sim-offset) { //halo or indices outside valid data
            pixel_similarities_local[put_idx] = 0;
        } else {
            pixel_similarities_local[put_idx] = pixel_similarities[z_idx_c + y_idx_c*width_sim + x_idx];
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
        if (get_global_id(1) < height_sim-2*offset && get_group_id(0)*BLOCK_SIZE_X*STEPS_ROW + BLOCK_SIZE_X*b + tx < width_ori) {
            const int idx = tz*(height_sim-2*offset)*width_ori + \
                            get_global_id(1)*width_ori + \
                            get_group_id(0)*BLOCK_SIZE_X*STEPS_ROW + BLOCK_SIZE_X*b + tx;
            intermed_row_avg[idx] = sum;
        }
    }
}
