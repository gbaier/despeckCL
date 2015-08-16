__kernel void precompute_filter_values (__global float * ampl_master, __global float * ampl_slave,  __global float * dphase,
                                        __global float * filter_data_a, __global float * filter_data_x_real, __global float * filter_data_x_imag,
                                        const int height_overlap, const int width_overlap, const int patch_size)
{
    const int width_sws  = width_overlap  - patch_size + 1;
    const int height_sws = height_overlap - patch_size + 1;
    const int psh = (patch_size-1)/2;

    const int tx = get_group_id(0) * get_local_size(0) + get_local_id(0);
    const int ty = get_group_id(1) * get_local_size(1) + get_local_id(1);

    if (tx < height_sws && ty < width_sws) {
        const float a1 = ampl_master[ (tx+psh)*width_overlap + (ty+psh) ];
        const float a2 = ampl_slave [ (tx+psh)*width_overlap + (ty+psh) ];
        const float dp = dphase     [ (tx+psh)*width_overlap + (ty+psh) ];

        filter_data_a      [tx*width_sws + ty] = 0.5f * ( pow(a1, 2.0f) + pow(a2, 2.0f) );
        filter_data_x_real [tx*width_sws + ty] = a1 * a2 * cos(-dp);
        filter_data_x_imag [tx*width_sws + ty] = a1 * a2 * sin(-dp);
    }
}
