__kernel void covmat_rescale (__global float * covmat,
                              const int dimension,
                              const int nlooks,
                              const int height,
                              const int width)
{
    const float LD    = ((float) nlooks) / dimension;
    const float gamma = pow(min(LD, 1.0f), 0.33333f);

    const int tx = get_group_id(0) * get_local_size(0) + get_local_id(0);
    const int ty = get_group_id(1) * get_local_size(1) + get_local_id(1);

    if ((tx < height) && (ty < width)) {
        for(int row_idx = 0; row_idx < dimension; row_idx++) {
            for(int col_idx = 0; col_idx < dimension; col_idx++) {
                if (row_idx != col_idx) {
                    covmat[  2*(row_idx * dimension + col_idx)     *height*width + tx*width + ty ] *= gamma;
                    covmat[ (2*(row_idx * dimension + col_idx) + 1)*height*width + tx*width + ty ] *= gamma;
                }
            }
        }
    }
}
