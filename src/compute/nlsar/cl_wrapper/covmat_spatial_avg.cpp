#include "covmat_spatial_avg.h"

#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>

covmat_spatial_avg::covmat_spatial_avg(const size_t block_size,
                                       cl::Context context,
                                       const int window_width) : window_width(window_width),
                                                                 output_block_size(block_size - window_width + 1)
{
    kernel_env::block_size = block_size;
    kernel_env::context = context;
    build_program(return_build_options());
    build_kernel();
}

covmat_spatial_avg::covmat_spatial_avg(const covmat_spatial_avg& other) : window_width(other.window_width),
                                                                          output_block_size(other.output_block_size)
{
    kernel_env::block_size = other.block_size;
    kernel_env::context = other.context;
    program = other.program;
    build_kernel();
}

std::string covmat_spatial_avg::return_build_options(void)
{
    std::ostringstream out;
    out << " -D WINDOW_WIDTH=" << window_width << " -D BLOCK_SIZE=" << block_size << " -D OUTPUT_BLOCK_SIZE=" << output_block_size;
    return "-Werror -cl-std=CL1.1" + out.str();
}

void covmat_spatial_avg::run(cl::CommandQueue cmd_queue,
                             cl::Buffer covmat_in,
                             cl::Buffer covmat_out,
                             const int dimension,
                             const int height,
                             const int width)
{
    kernel.setArg(0, covmat_in);
    kernel.setArg(1, covmat_out);
    kernel.setArg(2, dimension);
    kernel.setArg(3, height);
    kernel.setArg(4, width);

    cl::NDRange global_size {(size_t) block_size*( (height - 1)/output_block_size + 1), \
                             (size_t) block_size*( (width  - 1)/output_block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
