/* Copyright 2015, 2016 Gerald Baier
 *
 * This file is part of despeckCL.
 *
 * despeckCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * despeckCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with despeckCL. If not, see <http://www.gnu.org/licenses/>.
 */

#include "boxcar_wrapper.h"

#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>

boxcar_wrapper::boxcar_wrapper(const size_t block_size,
                               cl::Context context,
                               const int window_width) : kernel_env_single<boxcar_wrapper>(block_size, context),
                                                         window_width(window_width),
                                                         output_block_size(block_size - window_width + 1)
{
    program = build_program(build_opts(), kernel_source);
    kernel  = build_kernel(program, routine_name);
}

boxcar_wrapper::boxcar_wrapper(const boxcar_wrapper& other) : kernel_env_single<boxcar_wrapper>(other),
                                                              window_width(other.window_width),
                                                              output_block_size(other.output_block_size)
{
    program = other.program;
    kernel = build_kernel(program, routine_name);
}

std::string boxcar_wrapper::build_opts()
{
    std::ostringstream out;
    out << " -D WINDOW_WIDTH=" << window_width << " -D BLOCK_SIZE=" << block_size << " -D OUTPUT_BLOCK_SIZE=" << output_block_size;
    return default_build_opts() + out.str();
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
