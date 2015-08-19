__kernel void covmat_create (__global float* ampl_master,
                             __global float* ampl_slave,
                             __global float* dphase,
                             __global float* covmat,
                             const int height,
                             const int width)
{
    const int h = get_group_id(0) * get_local_size(0) + get_local_id(0);
    const int w = get_group_id(1) * get_local_size(1) + get_local_id(1);

    if ((h < height) && (w < width)) {
        const int idx = h*width + w;

        const float el_00real = ampl_master[idx] * ampl_master[idx];
        const float el_00imag = 0.0f;

        const float el_01real = ampl_master[idx] * ampl_slave [idx] * cos(dphase[idx]);
        const float el_01imag = ampl_master[idx] * ampl_slave [idx] * sin(dphase[idx]);

        const float el_10real =   el_01real;
        const float el_10imag = - el_01imag;

        const float el_11real = ampl_slave[idx] * ampl_slave[idx];
        const float el_11imag = 0.0f;

        covmat[idx]                  = el_00real;
        covmat[idx + 1*height*width] = el_00imag;

        covmat[idx + 2*height*width] = el_01real;
        covmat[idx + 3*height*width] = el_01imag;

        covmat[idx + 4*height*width] = el_10real;
        covmat[idx + 5*height*width] = el_10imag;

        covmat[idx + 6*height*width] = el_11real;
        covmat[idx + 7*height*width] = el_11imag;
    }
}
