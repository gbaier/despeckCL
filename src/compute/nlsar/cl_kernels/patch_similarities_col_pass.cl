__kernel void patch_similarities_col_pass (__global float * intermed_row_avg,
                                           __global float * patch_similarities,
                                           const int height_sim,
                                           const int width_sim,
                                           const int patch_size,
                                           const int patch_size_max,
                                           __local float * intermed_row_avg_local)
{
    const int ori_offset = (patch_size_max - 1) / 2;
    const int offset = (patch_size_max - patch_size) / 2;

    const int height_ori = height_sim - patch_size_max + 1;
    const int width_ori  = width_sim  - patch_size_max + 1;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tz = get_global_id(2);

    const int z_idx_c = tz*(height_sim-2*offset)*width_ori;
    const int y_idx_c = ori_offset+get_group_id(1)*BLOCK_SIZE_Y*STEPS_COL;
    const int x_idx_c = get_global_id(0);

    // load data into local memory
    // -2 and +2 are due to halo
    for(int b = -2; b < STEPS_COL + 2; b++) {
        const int put_idx = (ty+(b+2)*BLOCK_SIZE_Y)*BLOCK_SIZE_X + tx;
        const int y_idx = y_idx_c + BLOCK_SIZE_Y*b + ty;
        if(y_idx < 0 || y_idx >= height_sim || x_idx_c >= width_ori) { //halo or indices outside valid data
            intermed_row_avg_local[put_idx] = 0;
        } else {
            intermed_row_avg_local[put_idx] = intermed_row_avg[z_idx_c + y_idx*width_ori + x_idx_c];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int psh = (patch_size-1)/2;
    for(int b = 0; b < STEPS_COL; b++) {
        float sum = 0;
        for(int k = -psh; k <= psh; k++) {
            const int idx = (ty+k+(b+2)*BLOCK_SIZE_Y)*BLOCK_SIZE_X + tx;
            sum += intermed_row_avg_local[idx];
        }
        if (get_global_id(0) < width_ori && get_group_id(1)*BLOCK_SIZE_Y*STEPS_COL + BLOCK_SIZE_Y*b + ty < height_ori) {
            const int idx = tz*height_ori*width_ori + \
                            (get_group_id(1)*BLOCK_SIZE_Y*STEPS_COL + BLOCK_SIZE_Y*b + ty)*width_ori + \
                            get_global_id(0);
            patch_similarities[idx] = sum;
        }
    }
}
