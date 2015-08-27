__kernel void compute_patch_similarities (__global float * pixel_similarities,
                                          __global float * patch_similarities,
                                          const int height_sim,
                                          const int width_sim,
                                          const int patch_size,
                                          __local float * pixel_similarities_local,
                                          __local float * cache)
{
    const int height_ori = height_sim - patch_size + 1;
    const int width_ori  = width_sim  - patch_size + 1;

    const int output_block_size = get_local_size(0) - patch_size + 1;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tz = get_global_id(2);

    const int out_x = get_group_id(0) * output_block_size + tx;
    const int out_y = get_group_id(1) * output_block_size + ty;

    if ( (out_x < height_sim) && (out_y < width_sim) ) {
        pixel_similarities_local[tx*get_local_size(1) + ty] = pixel_similarities[tz*height_sim*width_sim + out_x*width_sim + out_y];
    } else {
        pixel_similarities_local[tx*get_local_size(1) + ty] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;

    // average over 2nd dimension
    if (ty < output_block_size) {
        for(int ky = 0; ky < patch_size; ky++) {
            sum += pixel_similarities_local[tx*get_local_size(1) + ty + ky];
        }
        cache[tx*output_block_size+ty] = sum;
    }
    // use the intermediate results for the 2nd dimension to average over the 1st
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tx < output_block_size) {
        for(int kx = 1; kx < patch_size; kx++) {
            sum += cache[(tx+kx)*output_block_size + ty];
        }
    }

    if ((tx < output_block_size) && (ty < output_block_size)) {
        if (out_x < height_ori && out_y < width_ori) {
            patch_similarities[tz*height_ori*width_ori + out_x*width_ori + out_y] = sum;
        }
    }
}
