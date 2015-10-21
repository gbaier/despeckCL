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

    const int output_block_size = get_local_size(0) - patch_size + 1;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tz = get_global_id(2);

    const int out_x = get_group_id(0) * output_block_size + tx;
    const int out_y = get_group_id(1) * output_block_size + ty;

    // loading data into local memory

    if ( (out_x < height_sim) && (out_y < width_sim) ) {
        pixel_similarities_local[tx*get_local_size(1) + ty] = pixel_similarities[tz*height_sim*width_sim + (offset+out_x)*width_sim + offset + out_y];
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
    if ((tx < output_block_size) && (ty < output_block_size)) {
        if (out_x < height_ori && out_y < width_ori) {
            patch_similarities[tz*height_ori*width_ori + out_x*width_ori + out_y] = sum;
        }
    }
}
