#include "compute_number_of_looks.h"

void compute_number_of_looks::run(cl::CommandQueue cmd_queue,
                                  cl::Buffer weights,
                                  cl::Buffer nols,
                                  const int height_ori,
                                  const int width_ori,
                                  const int search_window_size)
{
    kernel.setArg(0, weights);
    kernel.setArg(1, nols);
    kernel.setArg(2, height_ori);
    kernel.setArg(3, width_ori);
    kernel.setArg(4, search_window_size);

    cl::NDRange global_size {(size_t) block_size*((height_ori-1)/block_size+1), (size_t) block_size*((width_ori-1)/block_size+1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);

    cmd_queue.finish();
}
