__kernel void transpose(__global float * matrix, __global float * matrix_trans,
                        const int height, const int width)
{
    const int h = get_group_id(0) * THREAD_SIZE_COL + get_local_id(0);
    const int w = get_group_id(1) * THREAD_SIZE_COL + get_local_id(1);

    __local float block[THREAD_SIZE_COL][THREAD_SIZE_COL];

    for (int i = 0; i < THREAD_SIZE_COL; i += THREAD_SIZE_ROW) {
        if( (h+i) < height && w < width ) {
            block[get_local_id(0)+i][get_local_id(1)] = matrix[(h+i)*width + w];
        } else { // this should be unnecessary, since this block will not be copied back to matrix_trans
            block[get_local_id(0)+i][get_local_id(1)] = 0.0f;
        }
    }   

    barrier(CLK_LOCAL_MEM_FENCE);

    const int ht = get_group_id(1) * THREAD_SIZE_COL + get_local_id(0);
    const int wt = get_group_id(0) * THREAD_SIZE_COL + get_local_id(1);

    const int height_t = width;
    const int width_t = height;

    for (int i = 0; i < THREAD_SIZE_COL; i += THREAD_SIZE_ROW) {
        if( (ht+i) < height_t && wt < width_t ) {
            matrix_trans[(ht+i)*width_t + wt] = block[get_local_id(1)][get_local_id(0) + i];
        }
    }
}
