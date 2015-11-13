#include "weighted_multiply.h"

void goldstein::weighted_multiply::run(cl::CommandQueue cmd_queue,
                                       cl::Buffer interf_real,
                                       cl::Buffer interf_imag,
                                       const int height,
                                       const int width,
                                       const float alpha)
{
    kernel.setArg( 0, interf_real);
    kernel.setArg( 1, interf_imag);
    kernel.setArg( 2, height);
    kernel.setArg( 3, width);
    kernel.setArg( 4, alpha);

    cl::NDRange global_size {(size_t) block_size*( (width  - 1)/block_size + 1),
                             (size_t) block_size*( (height - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
