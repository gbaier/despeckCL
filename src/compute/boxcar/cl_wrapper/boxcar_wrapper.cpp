#include "boxcar_wrapper.h"

#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>

boxcar_wrapper::boxcar_wrapper(const size_t block_size,
                               cl::Context context,
                               const int window_width) : window_width(window_width),
                                                         output_block_size(block_size - window_width + 1)
{
    kernel_env::block_size = block_size;
    kernel_env::context    = context;
    build_program(return_build_options());
    build_kernel();
}

boxcar_wrapper::boxcar_wrapper(const boxcar_wrapper& other) : window_width(other.window_width),
                                                              output_block_size(other.output_block_size)
{
    kernel_env::block_size = other.block_size;
    kernel_env::context    = other.context;
    program                = other.program;
    build_kernel();
}

std::string boxcar_wrapper::return_build_options(void)
{
    std::ostringstream out;
    out << " -D WINDOW_WIDTH=" << window_width << " -D BLOCK_SIZE=" << block_size << " -D OUTPUT_BLOCK_SIZE=" << output_block_size;
    return "-Werror -cl-std=CL1.1" + out.str();
}

void boxcar_wrapper::run(cl::CommandQueue cmd_queue,
                         cl::Buffer ampl_master,
                         cl::Buffer ampl_slave,
                         cl::Buffer dphase,
                         cl::Buffer ampl_filt,
                         cl::Buffer dphase_filt,
                         cl::Buffer coh_filt,
                         const int height,
                         const int width)
{
    kernel.setArg(0, ampl_master);
    kernel.setArg(1, ampl_slave);
    kernel.setArg(2, dphase);
    kernel.setArg(3, ampl_filt);
    kernel.setArg(4, dphase_filt);
    kernel.setArg(5, coh_filt);
    kernel.setArg(6, height);
    kernel.setArg(7, width);

    cl::NDRange global_size {(size_t) block_size*( (height - 1)/output_block_size + 1), \
                             (size_t) block_size*( (width  - 1)/output_block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
