__kernel void slc2real(__global float* interf_real, 
                       __global float* interf_imag,
                       __global float* ampl,
                       __global float* dphase,
                       const int height,
                       const int width)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    const int idx = tx*width + ty;

    if(tx < height && ty < width) {
        ampl[idx]   = fabs(sqrt(pow(interf_real[idx], 2.0f) + pow(interf_imag[idx], 2.0f)));
        dphase[idx] = atan2(interf_imag[idx], interf_real[idx]);
    }
} 
