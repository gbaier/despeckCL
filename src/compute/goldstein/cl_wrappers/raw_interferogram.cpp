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

#include "raw_interferogram.h"

constexpr const char* goldstein::raw_interferogram::routine_name;
constexpr const char* goldstein::raw_interferogram::kernel_source;

void goldstein::raw_interferogram::run(cl::CommandQueue cmd_queue,
                                       cl::Buffer ampl_master,
                                       cl::Buffer ampl_slave,
                                       cl::Buffer phase,
                                       cl::Buffer interf_real,
                                       cl::Buffer interf_imag,
                                       const int height,
                                       const int width)
{
    kernel.setArg( 0, ampl_master);
    kernel.setArg( 1, ampl_slave);
    kernel.setArg( 2, phase);
    kernel.setArg( 3, interf_real);
    kernel.setArg( 4, interf_imag);
    kernel.setArg( 5, height);
    kernel.setArg( 6, width);

    cl::NDRange global_size {(size_t) block_size*( (width  - 1)/block_size + 1),
                             (size_t) block_size*( (height - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
