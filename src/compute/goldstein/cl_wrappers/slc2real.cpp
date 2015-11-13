#include "slc2real.h"

void goldstein::slc2real::run(cl::CommandQueue cmd_queue,
                              cl::Buffer interf_real,
                              cl::Buffer interf_imag,
                              cl::Buffer ampl,
                              cl::Buffer dphase,
                              const int height,
                              const int width)
{
    kernel.setArg( 0, interf_real);
    kernel.setArg( 1, interf_imag);
    kernel.setArg( 2, ampl);
    kernel.setArg( 3, dphase);
    kernel.setArg( 4, height);
    kernel.setArg( 5, width);

    cl::NDRange global_size {(size_t) block_size*( (width  - 1)/block_size + 1),
                             (size_t) block_size*( (height - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
