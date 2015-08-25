#include "compute_patch_similarities.h"

compute_patch_similarities::compute_patch_similarities(const size_t block_size,
                                                       cl::Context context,
                                                       const int patch_size) : patch_size(patch_size),
                                                                               output_block_size(block_size - patch_size + 1)
{
    kernel_env::block_size = block_size;
    kernel_env::context    = context;
    build_program(return_build_options());
    build_kernel();
}

compute_patch_similarities::compute_patch_similarities(const compute_patch_similarities& other) : patch_size(other.patch_size),
                                                                                                  output_block_size(other.output_block_size)
{
    kernel_env::block_size = other.block_size;
    kernel_env::context    = other.context;
    program = other.program;
    build_kernel();
}

std::string compute_patch_similarities::return_build_options(void)
{
    std::ostringstream out;
    out << " -D PATCH_SIZE=" << patch_size << " -D BLOCK_SIZE=" << block_size << " -D OUTPUT_BLOCK_SIZE=" << output_block_size;
    return kernel_env::return_build_options() + out.str();
}

void compute_patch_similarities::run(cl::CommandQueue cmd_queue,
                                     cl::Buffer pixel_similarities,
                                     cl::Buffer patch_similarities,
                                     const int height_sim,
                                     const int width_sim,
                                     const int search_window_size,
                                     const int patch_size)
{
    kernel.setArg(0, pixel_similarities);
    kernel.setArg(1, patch_similarities);
    kernel.setArg(2, height_sim);
    kernel.setArg(3, width_sim);
    kernel.setArg(4, patch_size);

    cl::NDRange global_size {(size_t) block_size*( (height_sim - 1)/output_block_size + 1), \
                             (size_t) block_size*( (width_sim  - 1)/output_block_size + 1), \
                             (size_t) search_window_size*search_window_size };
    cl::NDRange local_size  {block_size, block_size, 1};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
