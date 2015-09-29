#include "compute_patch_similarities.h"

int nlsar::compute_patch_similarities::get_output_block_size(const int patch_size)
{
    return block_size - patch_size + 1;
}

void nlsar::compute_patch_similarities::run(cl::CommandQueue cmd_queue,
                                            cl::Buffer pixel_similarities,
                                            cl::Buffer patch_similarities,
                                            const int height_sim,
                                            const int width_sim,
                                            const int search_window_size,
                                            const int patch_size,
                                            const int patch_size_max)
{
    const int output_block_size = get_output_block_size(patch_size);

    const int offset = (patch_size_max - patch_size) / 2;

    kernel.setArg(0, pixel_similarities);
    kernel.setArg(1, patch_similarities);
    kernel.setArg(2, height_sim);
    kernel.setArg(3, width_sim);
    kernel.setArg(4, patch_size);
    kernel.setArg(5, patch_size_max);
    kernel.setArg(6, cl::Local(block_size*block_size*sizeof(float)));
    kernel.setArg(7, cl::Local(block_size*output_block_size*sizeof(float)));

    cl::NDRange global_size {(size_t) block_size*( (height_sim - 2*offset - 1)/output_block_size + 1), \
                             (size_t) block_size*( (width_sim  - 2*offset - 1)/output_block_size + 1), \
                             (size_t) search_window_size*search_window_size };
    cl::NDRange local_size  {block_size, block_size, 1};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
