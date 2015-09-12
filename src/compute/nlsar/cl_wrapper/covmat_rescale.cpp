#include "covmat_rescale.h"

void nlsar::covmat_rescale::run(cl::CommandQueue cmd_queue,
                                cl::Buffer covmat,
                                const int dimension,
                                const int nlooks,
                                const int height,
                                const int width)
{
    kernel.setArg(0, covmat);
    kernel.setArg(1, dimension);
    kernel.setArg(2, nlooks);
    kernel.setArg(3, height);
    kernel.setArg(4, width);

    cl::NDRange global_size {(size_t) block_size*( (height - 1)/block_size + 1), \
                             (size_t) block_size*( (width  - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
