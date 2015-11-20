#include "tile_size.h"

#include "easylogging++.h"

int goldstein::round_down(const int num, const int multiple)
{
     int remainder = num % multiple;
     return num - remainder;
}

int goldstein::round_up(const int num, const int multiple)
{
     return round_down(num, multiple) + multiple;
}

int goldstein::tile_size(cl::Context context,
                         const int patch_size,
                         const int overlap)
{
    const int multiple = patch_size - 2*overlap;

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::Device dev = devices[0];

    int long max_mem_alloc_size;
    dev.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_mem_alloc_size);
    VLOG(0) << "maximum memory allocation size = " << max_mem_alloc_size;

    /*
     * the following buffer will take up space
     * 1) raw data, i.e. amplitude of master and slave image and their interferometric phase 
     *    => factor of 3
     * 2) real and complex value of the interferogram
     *    => factor of 2
     * 3) real and complex value of the patched interferogram
     *    => factor of 2
     * which leads to total factor of 7 times a factor of 4 as floats are stored in the buffers.
     *
     */

    const int n_pixels = max_mem_alloc_size / (4*7);

    const int tile_size_fit         = std::sqrt(n_pixels);
    const int tile_size_fit_rounded = round_down(tile_size_fit, multiple);

    VLOG(0) << "tile_size_fit = "         << tile_size_fit;
    VLOG(0) << "tile_size_fit_rounded = " << tile_size_fit_rounded;

    const float safety_factor = 0.5;
    int safe_tile_size = 0;
    if (float(tile_size_fit_rounded)/tile_size_fit < safety_factor) {
        safe_tile_size = tile_size_fit_rounded;
    } else {
        safe_tile_size = tile_size_fit_rounded - multiple;
    }
    VLOG(0) << "safe_tile_size = " << safe_tile_size;

    return safe_tile_size;
}
