#include "copy_symm_weights.h"

void nlsar::copy_symm_weights::run(cl::CommandQueue cmd_queue,
                                   cl::Buffer weights_symm,
                                   cl::Buffer weights_full,
                                   const int height_ori,
                                   const int width_ori,
                                   const int search_window_size)
{
    kernel.setArg(0, weights_symm);
    kernel.setArg(1, weights_full);
    kernel.setArg(2, height_ori);
    kernel.setArg(3, width_ori);
    kernel.setArg(4, search_window_size);

    cl::NDRange global_size {(size_t) width_ori, (size_t) height_ori};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, cl::NullRange, NULL, NULL);
}
