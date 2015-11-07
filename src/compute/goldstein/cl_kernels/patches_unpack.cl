__kernel void patches_unpack(__global float* interf_real_packed,
                             __global float* interf_imag_packed,
                             __global float* interf_real_unpacked, 
                             __global float* interf_imag_unpacked,
                             const int height_unpacked,
                             const int width_unpacked,
                             const int patch_size,
                             const int overlap)
{
    const int width_packed = width_unpacked / patch_size * (patch_size-2*overlap);
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    const int patch_idx = tx / patch_size;
    const int patch_idy = ty / patch_size;

    // indices of input for unpacking
    const int pixel_idx = max(0, tx - (patch_idx * (patch_size-2*overlap)) - overlap);
    const int pixel_idy = max(0, ty - (patch_idy * (patch_size-2*overlap)) - overlap);

    // no check necessary since we assume that the unpacked dimensions are fixed multiples of the block size
    interf_real_unpacked[ty*width_unpacked + tx] = interf_real_packed[pixel_idy*width_packed + pixel_idx];
    interf_imag_unpacked[ty*width_unpacked + tx] = interf_imag_packed[pixel_idy*width_packed + pixel_idx];
} 
