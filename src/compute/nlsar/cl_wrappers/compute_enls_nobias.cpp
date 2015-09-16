#include "compute_enls_nobias.h"

void nlsar::compute_enls_nobias::run(cl::CommandQueue cmd_queue,
                                     cl::Buffer enls,
                                     cl::Buffer alphas,
                                     cl::Buffer wsums,
                                     cl::Buffer enls_nobias,
                                     const int height_ori,
                                     const int width_ori)
{
    kernel.setArg(0, enls);
    kernel.setArg(1, alphas);
    kernel.setArg(2, wsums);
    kernel.setArg(3, enls_nobias);
    kernel.setArg(4, height_ori);
    kernel.setArg(5, width_ori);

    cl::NDRange global_size {(size_t) block_size*( (height_ori - 1)/block_size + 1), \
                             (size_t) block_size*( (width_ori  - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
