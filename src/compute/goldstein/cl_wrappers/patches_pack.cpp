#include "patches_pack.h"

void goldstein::patches_pack::run(cl::CommandQueue cmd_queue,
                                  cl::Buffer interf_real_unpacked, 
                                  cl::Buffer interf_imag_unpacked,
                                  cl::Buffer interf_real, 
                                  cl::Buffer interf_imag,
                                  const int height_unpacked,
                                  const int width_unpacked,
                                  const int patch_size,
                                  const int overlap)
{
    kernel.setArg( 0, interf_real_unpacked);
    kernel.setArg( 1, interf_imag_unpacked);
    kernel.setArg( 2, interf_real);
    kernel.setArg( 3, interf_imag);
    kernel.setArg( 4, height_unpacked);
    kernel.setArg( 5, width_unpacked);
    kernel.setArg( 6, patch_size);
    kernel.setArg( 7, overlap);

    cl::NDRange global_size {(size_t) block_size*( (width_unpacked  - 1)/block_size + 1),
                             (size_t) block_size*( (height_unpacked - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
