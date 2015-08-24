__kernel void covmat_decompose (__global float* covmat,
                                __global float* ampl_filt,
                                __global float* dphase_filt,
                                __global float* coh_filt,
                                const int height,
                                const int width)
{
    const int h = get_group_id(0) * get_local_size(0) + get_local_id(0);
    const int w = get_group_id(1) * get_local_size(1) + get_local_id(1);

    if ((h < height) && (w < width)) {
        const int idx = h*width + w;

        const float ampl_master = sqrt(covmat[idx]);
        const float ampl_slave  = sqrt(covmat[idx + 6*height*width]);

        ampl_filt   [idx] = ampl_master;
        dphase_filt [idx] =    atan2(covmat[idx + 3*height*width],              covmat[idx + 2*height*width]);
        coh_filt    [idx] = sqrt(pow(covmat[idx + 3*height*width], 2.0f) +  pow(covmat[idx + 2*height*width], 2.0f)) / (ampl_master*ampl_slave);
    }
}
