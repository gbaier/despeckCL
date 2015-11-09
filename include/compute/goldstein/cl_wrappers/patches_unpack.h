#ifndef PATCHES_UNPACK_H
#define PATCHES_UNPACK_H

#include "clcfg.h"

#include <string>

namespace goldstein
    {
    class patches_unpack : public kernel_env<patches_unpack>
    {
        public:
            using kernel_env::kernel_env;

            static constexpr const char* kernel_source =
    "__kernel void patches_unpack(__global float* interf_real_packed,\n"
"                             __global float* interf_imag_packed,\n"
"                             __global float* interf_real_unpacked, \n"
"                             __global float* interf_imag_unpacked,\n"
"                             const int height_unpacked,\n"
"                             const int width_unpacked,\n"
"                             const int patch_size,\n"
"                             const int overlap)\n"
"{\n"
"    const int width_packed = width_unpacked / patch_size * (patch_size-2*overlap);\n"
"    const int tx = get_global_id(0);\n"
"    const int ty = get_global_id(1);\n"
"\n"
"    const int patch_idx = tx / patch_size;\n"
"    const int patch_idy = ty / patch_size;\n"
"\n"
"    // indices of input for unpacking\n"
"    const int pixel_idx = max(0, tx - (patch_idx * (patch_size-2*overlap)) - overlap);\n"
"    const int pixel_idy = max(0, ty - (patch_idy * (patch_size-2*overlap)) - overlap);\n"
"\n"
"    // no check necessary since we assume that the unpacked dimensions are fixed multiples of the block size\n"
"    interf_real_unpacked[ty*width_unpacked + tx] = interf_real_packed[pixel_idy*width_packed + pixel_idx];\n"
"    interf_imag_unpacked[ty*width_unpacked + tx] = interf_imag_packed[pixel_idy*width_packed + pixel_idx];\n"
"} \n"
"\n"
    ;

            static constexpr const char* routine_name {"patches_unpack"};

            void run(cl::CommandQueue cmd_queue,
                     cl::Buffer interf_real, 
                     cl::Buffer interf_img,
                     cl::Buffer interf_real_unpacked, 
                     cl::Buffer interf_imag_unpacked,
                     const int height_unpacked,
                     const int width_unpacked,
                     const int patch_size,
                     const int overlap);
    };
}
#endif
