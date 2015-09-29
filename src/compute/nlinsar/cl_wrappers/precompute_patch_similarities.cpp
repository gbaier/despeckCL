#include "precompute_patch_similarities.h"

#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>

nlinsar::precompute_patch_similarities::precompute_patch_similarities(const size_t block_size,
                                                                      cl::Context context,
                                                                      const int window_width) : kernel_env<precompute_patch_similarities>(block_size,
                                                                                                                                          context,
                                                                                                               return_build_options(window_width,
                                                                                                                                    block_size,
                                                                                                                                    block_size - window_width + 1)),
                                                                                                window_width(window_width),
                                                                                                output_block_size(block_size - window_width + 1) {}

nlinsar::precompute_patch_similarities::precompute_patch_similarities(const precompute_patch_similarities& other) : kernel_env<precompute_patch_similarities>(other),
                                                                                                                    window_width(other.window_width),
                                                                                                                    output_block_size(other.output_block_size) {}

std::string nlinsar::precompute_patch_similarities::return_build_options(const int window_width, const int block_size, const int output_block_size)
{
    std::ostringstream out;
    out << " -D WINDOW_WIDTH=" << window_width << " -D BLOCK_SIZE=" << block_size << " -D OUTPUT_BLOCK_SIZE=" << output_block_size;
    return return_default_build_opts() + out.str();
}

void nlinsar::precompute_patch_similarities::run(cl::CommandQueue cmd_queue,
                                                 cl::Buffer similarities,
                                                 cl::Buffer kullback_leiblers,
                                                 const int height_sim,
                                                 const int width_sim,
                                                 const int search_window_size,
                                                 const int patch_size,
                                                 cl::Buffer patch_similarities,
                                                 cl::Buffer patch_kullback_leiblers)
{
    const int height_ori = height_sim - patch_size + 1;
    const int width_ori  = width_sim  - patch_size + 1;

    kernel.setArg(2, height_ori);
    kernel.setArg(3, width_ori);
    kernel.setArg(4, patch_size);

    cl::NDRange global_size {(size_t) block_size*( (height_ori - 1)/output_block_size + 1), \
                             (size_t) block_size*( (width_ori  - 1)/output_block_size + 1), \
                             (size_t) search_window_size*search_window_size };
    cl::NDRange local_size  {block_size, block_size, 1};
    
    std::vector<std::pair<cl::Buffer, cl::Buffer>> src_dest_pairs {{similarities, patch_similarities}, {kullback_leiblers, patch_kullback_leiblers}};

    for(auto src_dest : src_dest_pairs) {
        cl::Buffer cl_buffer_in   = src_dest.first;
        cl::Buffer cl_buffer_out  = src_dest.second;

        kernel.setArg(0, cl_buffer_in);
        kernel.setArg(1, cl_buffer_out);

        cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
    }
}
