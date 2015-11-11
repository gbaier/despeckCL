__kernel void patches_unpack(__global float* interf_real_packed,
                             __global float* interf_imag_packed,
                             __global float* interf_real_unpacked, 
                             __global float* interf_imag_unpacked,
                             const int height_unpacked,
                             const int width_unpacked,
                             const int patch_size,
                             const int overlap)
{
    const int width_packed  = (width_unpacked  / patch_size) * (patch_size-2*overlap);
    const int height_packed = (height_unpacked / patch_size) * (patch_size-2*overlap);

    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    const int patch_idx = tx / patch_size;
    const int patch_idy = ty / patch_size;

    const int rel_tx = tx % patch_size;
    const int rel_ty = ty % patch_size;

    const int idx_packed = min(width_packed  - 1, max(0, patch_idx*(patch_size-2*overlap) + (rel_tx - overlap)));
    const int idy_packed = min(height_packed - 1, max(0, patch_idy*(patch_size-2*overlap) + (rel_ty - overlap)));

    // no check necessary since we assume that the unpacked dimensions are fixed multiples of the block size
    interf_real_unpacked[ty*width_unpacked + tx] = interf_real_packed[idy_packed*width_packed + idx_packed];
    interf_imag_unpacked[ty*width_unpacked + tx] = interf_imag_packed[idy_packed*width_packed + idx_packed];
} 
