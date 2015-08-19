#include "compute_pixel_similarities_2x2.h"

void compute_pixel_similarities_2x2::run(cl::CommandQueue cmd_queue,
                                         cl::Buffer covmat,
                                         cl::Buffer pixel_similarities,
                                         const int height_overlap,
                                         const int width_overlap,
                                         const int search_window_size)
{
    kernel.setArg(0, covmat);
    kernel.setArg(1, pixel_similarities);
    kernel.setArg(2, height_overlap);
    kernel.setArg(3, width_overlap);
    kernel.setArg(4, search_window_size);

    cl::NDRange global_size {(size_t) block_size*( (height_overlap - 1)/block_size + 1), \
                             (size_t) block_size*( (width_overlap  - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
