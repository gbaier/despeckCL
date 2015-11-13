__kernel void weighted_multiply (__global float* interf_real_in,
                                 __global float* interf_imag_in,
                                 __global float* interf_real_out, 
                                 __global float* interf_imag_out,
                                 const int height,
                                 const int width,
                                 const float alpha)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    const int idx = tx*width + ty;

    if(tx < height && ty < width) {
        const float psd = sqrt(pow(interf_real_in[idx], 2.0f) + pow(interf_imag_in[idx], 2.0f));
        interf_real_out[idx] = pow(psd, alpha)*interf_real_in[idx];
        interf_imag_out[idx] = pow(psd, alpha)*interf_imag_in[idx];
    }
} 
