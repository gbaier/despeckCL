#include "compute_weights.h"

void nlsar::compute_weights::run(cl::CommandQueue cmd_queue,
                                 cl::Buffer patch_similarities,
                                 cl::Buffer weights,
                                 const int height_ori,
                                 const int width_ori,
                                 const int search_window_size,
                                 const int patch_size,
                                 cl::Buffer dissims2relidx,
                                 cl::Buffer chi2cdf_inv,
                                 const int lut_size,
                                 const float dissims_min,
                                 const float dissims_max)
{
    kernel.setArg( 0, patch_similarities);
    kernel.setArg( 1, weights);
    kernel.setArg( 2, height_ori);
    kernel.setArg( 3, width_ori);
    kernel.setArg( 4, search_window_size);
    kernel.setArg( 5, patch_size);
    kernel.setArg( 6, dissims2relidx);
    kernel.setArg( 7, chi2cdf_inv);
    kernel.setArg( 8, lut_size);
    kernel.setArg( 9, dissims_min);
    kernel.setArg(10, dissims_max);

    cl::NDRange global_size {(size_t) block_size*( (height_ori*width_ori*search_window_size*search_window_size - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
