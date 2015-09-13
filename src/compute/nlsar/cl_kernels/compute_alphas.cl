__kernel void compute_alphas (__global float * intensities_nl,
                              __global float * weighted_variances,
                              __global float * alphas,
                              const int height_ori,
                              const int width_ori,
                              const int dimension,
                              const int nlooks) // multilooking unrelated to nonlocal filtering
{
    const int tx = get_group_id(0) * get_local_size(0) + get_local_id(0);
    const int ty = get_group_id(1) * get_local_size(1) + get_local_id(1);

    float alpha = 0.0f;

    if (tx < height_ori && ty < width_ori) {
        const int idx = tx + width_ori + ty;
        for(int d=0; d<dimension; d++) {
            const float var    = weighted_variances [d*height_ori*width_ori + idx];
            const float int_nl = intensities_nl     [d*height_ori*width_ori + idx];
            alpha = max(alpha, max(0.0f, (var - int_nl*int_nl/nlooks)/var));
        }
        alphas[tx*width_ori + ty] = alpha;
    }
}
