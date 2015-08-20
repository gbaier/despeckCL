__kernel void compute_patch_similarities (__global float * pixel_similarities,
                                          __global float * patch_similarities,
                                          const int height_sim,
                                          const int width_sim,
                                          const int patch_size)
{
    const int height_ori = height_sim - patch_size + 1;
    const int width_ori  = width_sim  - patch_size + 1;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tz = get_global_id(2);

    const int out_x = get_group_id(0) * OUTPUT_BLOCK_SIZE + tx;
    const int out_y = get_group_id(1) * OUTPUT_BLOCK_SIZE + ty;

    __local float pixel_similarities_local [BLOCK_SIZE][BLOCK_SIZE];
    __local float intermed_local           [BLOCK_SIZE][OUTPUT_BLOCK_SIZE];

    if ( (out_x < height_sim) && (out_y < width_sim) ) {
        pixel_similarities_local[tx][ty] = pixel_similarities[tz*height_sim*width_sim + out_x*width_sim + out_y];
    } else {
        pixel_similarities_local[tx][ty] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;

    // average over 2nd dimension
    if (ty < OUTPUT_BLOCK_SIZE) {
        for(int ky = 0; ky < patch_size; ky++) {
            sum += pixel_similarities_local[tx][ty + ky];
        }
        intermed_local[tx][ty] = sum;
    }
    // use the intermediate results for the 2nd dimension to average over the 1st
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tx < OUTPUT_BLOCK_SIZE) {
        for(int kx = 1; kx < patch_size; kx++) {
            sum += intermed_local[tx+kx][ty];
        }
    }

    if ((tx < OUTPUT_BLOCK_SIZE) && (ty < OUTPUT_BLOCK_SIZE)) {
        if (out_x < height_ori && out_y < width_ori) {
            patch_similarities[tz*height_ori*width_ori + out_x*width_ori + out_y] = sum;
        }
    }
}
