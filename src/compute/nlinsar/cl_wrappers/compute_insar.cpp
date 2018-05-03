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

#include "compute_insar.h"

#include <stdio.h>
#include <sstream>

constexpr const char* nlinsar::compute_insar::routine_name;
constexpr const char* nlinsar::compute_insar::kernel_source;

nlinsar::compute_insar::compute_insar(const size_t block_size,
                                      cl::Context context,
                                      const int search_window_size) : kernel_env_single<compute_insar>(block_size,
                                                                                                context),
                                                                      search_window_size(search_window_size)
{
    program = build_program(build_opts(), kernel_source);
    kernel  = build_kernel(program, routine_name);
}

nlinsar::compute_insar::compute_insar(const compute_insar& other) : kernel_env_single<compute_insar>(other),
                                                                    search_window_size(other.search_window_size)
{
    program = other.program;
    kernel  = build_kernel(program, routine_name);
}

std::string nlinsar::compute_insar::build_opts()
{
    std::ostringstream out;
    out << " -D BLOCK_SIZE=" << block_size << " -D SEARCH_WINDOW_SIZE=" << search_window_size;
    return default_build_opts() + out.str();
}

void nlinsar::compute_insar::run(cl::CommandQueue cmd_queue,
                                 cl::Buffer device_filter_values_a,
                                 cl::Buffer device_filter_values_x_real,
                                 cl::Buffer device_filter_values_x_imag,
                                 cl::Buffer device_ref_filt,
                                 cl::Buffer device_phi_filt,
                                 cl::Buffer device_coh_filt,
                                 const int height_overlap,
                                 const int width_overlap,
                                 cl::Buffer device_weights,
                                 const int search_window_size,
                                 const int patch_size)
{
    const int height_ori = height_overlap - search_window_size - patch_size + 2;
    const int width_ori  = width_overlap  - search_window_size - patch_size + 2;

    kernel.setArg(0,  device_filter_values_a);
    kernel.setArg(1,  device_filter_values_x_real);
    kernel.setArg(2,  device_filter_values_x_imag);

    kernel.setArg(3,  device_ref_filt);
    kernel.setArg(4,  device_phi_filt);
    kernel.setArg(5,  device_coh_filt);
    
    kernel.setArg(6,  device_weights);
    kernel.setArg(7,  height_ori);
    kernel.setArg(8,  width_ori);
    
    kernel.setArg(9,  search_window_size);
    kernel.setArg(10, patch_size);

    cl::NDRange global_size {(size_t) block_size*((height_ori-1)/block_size+1), (size_t) block_size*((width_ori-1)/block_size+1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);

    cmd_queue.finish();
}
