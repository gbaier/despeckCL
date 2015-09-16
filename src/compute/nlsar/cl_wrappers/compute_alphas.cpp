#include "compute_alphas.h"

void nlsar::compute_alphas::run(cl::CommandQueue cmd_queue,
                                cl::Buffer intensities_nl,
                                cl::Buffer weighted_variances,
                                cl::Buffer alphas,
                                const int height_ori,
                                const int width_ori,
                                const int dimensions,
                                const int nlooks) // multilooking unrelated to nonlocal filtering
{
    kernel.setArg(0, intensities_nl);
    kernel.setArg(1, weighted_variances);
    kernel.setArg(2, alphas);
    kernel.setArg(3, height_ori);
    kernel.setArg(4, width_ori);
    kernel.setArg(5, dimensions);
    kernel.setArg(6, nlooks);

    cl::NDRange global_size {(size_t) block_size*( (height_ori - 1)/block_size + 1), \
                             (size_t) block_size*( (width_ori  - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
