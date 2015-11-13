__kernel void raw_interferogram(__global float* ampl_master,
                                __global float* ampl_slave,
                                __global float* dphase,
                                __global float* interf_real, 
                                __global float* interf_imag,
                                const int height,
                                const int width)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    const int idx = tx*width + ty;

    if(tx < height && ty < width) {
        interf_real[idx] = 0.5f*(ampl_master[idx] + ampl_slave[idx])*cos(dphase[idx]);
        interf_imag[idx] = 0.5f*(ampl_master[idx] + ampl_slave[idx])*sin(dphase[idx]);
    }
} 
