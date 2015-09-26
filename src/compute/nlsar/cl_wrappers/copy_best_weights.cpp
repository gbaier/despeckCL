#include "copy_best_weights.h"

void nlsar::copy_best_weights::run(cl::CommandQueue cmd_queue,
                                   cl::Buffer all_weights,
                                   cl::Buffer best_params,
                                   cl::Buffer best_weights,
                                   const int height_ori,
                                   const int width_ori,
                                   const int search_window_size)
{
    kernel.setArg( 0, all_weights);
    kernel.setArg( 1, best_params);
    kernel.setArg( 2, best_weights);
    kernel.setArg( 3, height_ori);
    kernel.setArg( 4, width_ori);
    kernel.setArg( 5, search_window_size);

    cl::NDRange global_size {(size_t) block_size*( (height_ori*width_ori-1)/block_size + 1)};
    cl::NDRange local_size  {block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
