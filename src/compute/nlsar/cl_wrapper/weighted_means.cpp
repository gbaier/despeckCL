#include "weighted_means.h"

#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>

weighted_means::weighted_means(const size_t block_size,
                               cl::Context context,
                               const int search_window_size,
                               const int dimension) : search_window_size(search_window_size),
                                                      dimension(dimension)
{
    kernel_env::block_size = block_size;
    kernel_env::context = context;
    build_program(return_build_options());
    build_kernel();
}

weighted_means::weighted_means(const weighted_means& other) : search_window_size(other.search_window_size),
                                                              dimension(other.dimension)
{
    kernel_env::block_size = other.block_size;
    kernel_env::context = other.context;
    program = other.program;
    build_kernel();
}

std::string weighted_means::return_build_options(void)
{
    std::ostringstream out;
    out << " -D SEARCH_WINDOW_SIZE=" << search_window_size << " -D BLOCK_SIZE=" << block_size << " -D DIMENSION=" << dimension;
    return kernel_env::return_build_options() + out.str();
}

void weighted_means::run(cl::CommandQueue cmd_queue,
                         cl::Buffer covmat_in,
                         cl::Buffer covmat_out,
                         cl::Buffer weights,
                         const int height_ori,
                         const int width_ori,
                         const int search_window_size,
                         const int patch_size,
                         const int window_width)
{
    kernel.setArg(0, covmat_in);
    kernel.setArg(1, covmat_out);
    kernel.setArg(2, weights);
    kernel.setArg(3, height_ori);
    kernel.setArg(4, width_ori);
    kernel.setArg(5, search_window_size);
    kernel.setArg(6, patch_size);
    kernel.setArg(7, window_width);

    cl::NDRange global_size {(size_t) block_size*( (height_ori - 1)/block_size + 1), \
                             (size_t) block_size*( (width_ori  - 1)/block_size + 1)};

    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
