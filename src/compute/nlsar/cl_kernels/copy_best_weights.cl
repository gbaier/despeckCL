__kernel void copy_best_weights (__global float * all_weights,
                                 __global int * best_params,
                                 __global float * best_weights,
                                 const int height_ori,
                                 const int width_ori,
                                 const int search_window_size)
{
    const int tx = get_global_id(0);
    const int weights_size = search_window_size*search_window_size*height_ori*width_ori;

    if (tx < height_ori * width_ori) {
        const int best_idx = best_params[tx];
        for(int i = 0; i < search_window_size*search_window_size; i++) {
            //if neighbouring pixels share the same parameters the memory access will be coalesced
            best_weights[i*height_ori*width_ori + tx] = all_weights[best_idx*weights_size + i*height_ori*width_ori + tx];
        }
    }
}
