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

constexpr const char* boxcar::cl_wrappers::routine_name;
constexpr const char* boxcar::cl_wrappers::kernel_source;

boxcar::cl_wrappers::cl_wrappers(cl::Context context,
                                 boxcar::kernel_params kp) : kernel_env_single<cl_wrappers>(kp.block_size, context),
                                                             window_width(kp.window_width),
                                                             output_block_size(kp.block_size - kp.window_width + 1)
{
    program = build_program(build_opts(), kernel_source);
    kernel  = build_kernel(program, routine_name);
}

boxcar::cl_wrappers::cl_wrappers(const cl_wrappers& other) : kernel_env_single<cl_wrappers>(other),
                                                                      window_width(other.window_width),
                                                                      output_block_size(other.output_block_size)
{
    program = other.program;
    kernel = build_kernel(program, routine_name);
}

std::string boxcar::cl_wrappers::build_opts()
{
    std::ostringstream out;
    out << " -D WINDOW_WIDTH=" << window_width << " -D BLOCK_SIZE=" << block_size << " -D OUTPUT_BLOCK_SIZE=" << output_block_size;
    return default_build_opts() + out.str();
}

void boxcar::cl_wrappers::run(cl::CommandQueue cmd_queue,
                              cl::Buffer ampl_master,
                              cl::Buffer ampl_slave,
                              cl::Buffer phase,
                              cl::Buffer ref_filt,
                              cl::Buffer phase_filt,
                              cl::Buffer coh_filt,
                              const int height,
                              const int width)
{
    kernel.setArg(0, ampl_master);
    kernel.setArg(1, ampl_slave);
    kernel.setArg(2, phase);
    kernel.setArg(3, ref_filt);
    kernel.setArg(4, phase_filt);
    kernel.setArg(5, coh_filt);
    kernel.setArg(6, height);
    kernel.setArg(7, width);

    cl::NDRange global_size {(size_t) block_size*( (height - 1)/output_block_size + 1), \
                             (size_t) block_size*( (width  - 1)/output_block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}

boxcar::cl_wrappers boxcar::get_cl_wrappers(cl::Context cl_context, boxcar::kernel_params pm) {
    return boxcar::cl_wrappers(cl_context, pm);
}
