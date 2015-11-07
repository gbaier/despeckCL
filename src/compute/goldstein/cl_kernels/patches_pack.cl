__kernel void patches_pack(__global float* interf_real_unpacked, 
                           __global float* interf_imag_unpacked,
                           __global float* interf_real_packed,
                           __global float* interf_imag_packed,
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

    const int rel_tx = tx - patch_idx * patch_size;
    const int rel_ty = ty - patch_idy * patch_size;

    float scaling_factor = 1.0f;
    if(rel_tx < overlap || rel_tx > patch_size - overlap) {
        scaling_factor *= 0.5f;
    }
    if(rel_ty < overlap || rel_ty > patch_size - overlap) {
        scaling_factor *= 0.5f;
    }

    // indices of input for unpacking
    const int pixel_idx = max(0, tx - (patch_idx * (patch_size-2*overlap)) - overlap);
    const int pixel_idy = max(0, ty - (patch_idy * (patch_size-2*overlap)) - overlap);

    const float val_real = scaling_factor * interf_real_packed[ty*width_unpacked + tx];
    const float val_imag = scaling_factor * interf_imag_packed[ty*width_unpacked + tx];

    // no check necessary since we assume that the unpacked dimensions are fixed multiples of the block size
    interf_real_packed[pixel_idy*width_packed + pixel_idx] += val_real;
    interf_imag_packed[pixel_idy*width_packed + pixel_idx] += val_imag;
} 
