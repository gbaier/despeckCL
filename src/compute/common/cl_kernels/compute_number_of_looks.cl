__kernel void compute_number_of_looks (__global float * weights,
                                       __global float * nols,
                                       const int height_ori,
                                       const int width_ori,
                                       const int search_window_size)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    const int swsh = (search_window_size-1)/2;

    float wsum  = 0.0f;
    float wsum2 = 0.0f;

    if (tx < height_ori && ty < width_ori) {
        for(int k = 0; k < search_window_size*search_window_size; k++) {
            const int idx = k * height_ori * width_ori + tx*width_ori + ty;
            const float weight = weights[idx];
            wsum += weight;
            wsum2 += weight*weight;
        }
        nols[tx*width_ori + ty] = wsum*wsum/wsum2;
    }
}
