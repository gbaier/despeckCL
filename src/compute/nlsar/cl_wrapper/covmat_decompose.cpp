#include "covmat_decompose.h"

void nlsar::covmat_decompose::run(cl::CommandQueue cmd_queue,
                                  cl::Buffer covmat,
                                  cl::Buffer amplitude,
                                  cl::Buffer dphase,
                                  cl::Buffer coherence,
                                  const int height,
                                  const int width)
{
    kernel.setArg(0, covmat);
    kernel.setArg(1, amplitude);
    kernel.setArg(2, dphase);
    kernel.setArg(3, coherence);
    kernel.setArg(4, height);
    kernel.setArg(5, width);

    cl::NDRange global_size {(size_t) block_size*( (height - 1)/block_size + 1), \
                             (size_t) block_size*( (width  - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
