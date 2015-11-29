#include "weighted_means.h"

#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>

nlsar::weighted_means::weighted_means(const size_t block_size,
                                      cl::Context context,
                                      const int search_window_size,
                                      const int dimension) : kernel_env_single<weighted_means>(block_size, context),
                                                             search_window_size(search_window_size),
                                                             dimension(dimension)
{
    program = build_program(build_opts(), kernel_source);
    kernel  = build_kernel(program, routine_name);
}

nlsar::weighted_means::weighted_means(const weighted_means& other) : kernel_env_single<weighted_means>(other),
                                                                     search_window_size(other.search_window_size),
                                                                     dimension(other.dimension)
{
    program = other.program;
    kernel  = build_kernel(program, routine_name);
}

std::string nlsar::weighted_means::build_opts()
{
    std::ostringstream out;
    out << " -D SEARCH_WINDOW_SIZE=" << search_window_size << " -D BLOCK_SIZE=" << block_size << " -D DIMENSION=" << dimension;
    return default_build_opts() + out.str();
}

void nlsar::weighted_means::run(cl::CommandQueue cmd_queue,
                                cl::Buffer covmat_in,
                                cl::Buffer covmat_out,
                                cl::Buffer weights,
                                cl::Buffer alphas,
                                const int height_ori,
                                const int width_ori,
                                const int search_window_size,
                                const int patch_size,
                                const int window_width)
{
    kernel.setArg(0, covmat_in);
    kernel.setArg(1, covmat_out);
    kernel.setArg(2, weights);
    kernel.setArg(3, alphas);
    kernel.setArg(4, height_ori);
    kernel.setArg(5, width_ori);
    kernel.setArg(6, search_window_size);
    kernel.setArg(7, patch_size);
    kernel.setArg(8, window_width);

    cl::NDRange global_size {(size_t) block_size*( (height_ori - 1)/block_size + 1), \
                             (size_t) block_size*( (width_ori  - 1)/block_size + 1)};

    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
