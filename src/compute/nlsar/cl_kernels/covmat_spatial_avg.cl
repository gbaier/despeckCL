__kernel void covmat_spatial_avg (__global float * covmat_in,
                                  __global float * covmat_out, 
                                  const int dimension,
                                  const int height_overlap,
                                  const int width_overlap,
                                  const int scale_size,
                                  __local float * local_data)
{
    const int block_size = get_local_size(0);
    const int output_block_size = get_local_size(0) - scale_size + 1;

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    const int in_x = get_group_id(0) * output_block_size + tx;
    const int in_y = get_group_id(1) * output_block_size + ty;

    const int height_overlap_avg = height_overlap + scale_size - 1;
    const int width_overlap_avg  = width_overlap  + scale_size - 1;

    for(int i = 0; i < 2*dimension*dimension; i++) {
        if ( (in_x < height_overlap_avg) && (in_y < width_overlap_avg) ) {
            local_data [tx*block_size + ty] = covmat_in [i*height_overlap_avg*width_overlap_avg + in_x*width_overlap_avg + in_y];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        float sum = 0;

        if ((tx < output_block_size) && (ty < output_block_size)) {
            for(int kx = 0; kx < scale_size; kx++) {
                for(int ky = 0; ky < scale_size; ky++) {
                    sum += local_data[(tx + kx)*block_size + ty + ky];
                }
            }
            if (in_x < height_overlap &&  in_y < width_overlap) {
                covmat_out[i*height_overlap*width_overlap + in_x*width_overlap + in_y] = sum/(scale_size*scale_size);
            }
        }
    }
}
