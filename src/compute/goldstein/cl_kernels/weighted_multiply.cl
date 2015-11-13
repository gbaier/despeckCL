__kernel void weighted_multiply (__global float* interf_real,
                                 __global float* interf_imag,
                                 const int height,
                                 const int width,
                                 const float alpha)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    const int idx = tx*width + ty;

    if(tx < height && ty < width) {
        const float psd = sqrt(pow(interf_real[idx], 2.0f) + pow(interf_imag[idx], 2.0f));
        interf_real[idx] = pow(psd, alpha)*interf_real[idx];
        interf_imag[idx] = pow(psd, alpha)*interf_imag[idx];
    }
} 
