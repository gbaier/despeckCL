inline const int symm_idx(const int idx,    // center pixel position
                          const int sw_idx, // pixel position in search window
                          const int wsh)    // half width of search window
{
   return idx + sw_idx - wsh;
}

inline const int symm_sw_idx(const int sw_idx,             // pixel position in search window
                             const int search_window_size) // half width of search window
{
   return search_window_size - sw_idx - 1;
}


__kernel void copy_symm_weights(__global float * weights_symm,
                                __global float * weights_full,
                                const int height_ori,
                                const int width_ori,
                                const int search_window_size)
{
    const int wsh = (search_window_size - 1)/2;

    const int h = get_global_id(0);
    const int w = get_global_id(1);

    const int h_symm = h;
    const int w_symm = w + wsh;

    const int width_symm  = width_ori  + 2*wsh;
    const int height_symm = height_ori +   wsh;

    if( h < height_ori && w < width_ori) {
        // copy precomputed values
        for(int i = 0; i < wsh*search_window_size + wsh; i++) {
            const int idx = i * height_ori * width_ori + h * width_ori + w;
            weights_full[idx] = weights_symm[i*height_symm*width_symm + h_symm*width_symm + w_symm];
        }
        // copy symmetric values
        for(int hh = wsh; hh < search_window_size; hh++) {
            int ww_start = 0;
            if (hh == wsh) {
                ww_start = wsh+1;
            }
            for(int ww = ww_start; ww < search_window_size; ww++) {
                const int idx = (hh * search_window_size + ww) * height_ori * width_ori \
                                                                        + h * width_ori \
                                                                                    + w;
                const int hs =       symm_idx(h, hh, wsh);
                const int ws = wsh + symm_idx(w, ww, wsh);
                const int hhs = symm_sw_idx(hh, search_window_size);
                const int wws = symm_sw_idx(ww, search_window_size);
                const int sidx = hhs * search_window_size * height_symm * width_symm \
                                                    + wws * height_symm * width_symm \
                                                                   + hs * width_symm \
                                                                                + ws;
                weights_full[idx] = weights_symm[sidx];
            }
        }
    }
}
