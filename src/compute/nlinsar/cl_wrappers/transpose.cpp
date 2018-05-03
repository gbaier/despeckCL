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

#include "transpose.h"

#include <sstream>

constexpr const char* nlinsar::transpose::routine_name;
constexpr const char* nlinsar::transpose::kernel_source;

nlinsar::transpose::transpose(const size_t block_size,
                              cl::Context context,
                              const size_t thread_size_row,
                              const size_t thread_size_col) : kernel_env_single<transpose>(block_size, context),
                                                              thread_size_row(thread_size_row),
                                                              thread_size_col(thread_size_col)
{
    program = build_program(build_opts(), kernel_source);
    kernel  = build_kernel(program, routine_name);
}

nlinsar::transpose::transpose(const transpose& other) : kernel_env_single<transpose>(other),
                                                        thread_size_row(other.thread_size_row),
                                                        thread_size_col(other.thread_size_col)
{
    program = other.program;
    kernel  = build_kernel(program, routine_name);
}

std::string nlinsar::transpose::build_opts()
{
    std::ostringstream out;
    out << " -D THREAD_SIZE_ROW=" << thread_size_row << " -D THREAD_SIZE_COL=" << thread_size_col;
    return default_build_opts() + out.str();
}

void nlinsar::transpose::run(cl::CommandQueue cmd_queue,
                             cl::Buffer matrix,
                             const int height,
                             const int width)
{
    cl::Buffer matrix_trans{context, CL_MEM_READ_WRITE, height * width * sizeof(float), NULL, NULL};
    
    kernel.setArg(0, matrix);
    kernel.setArg(1, matrix_trans);
    kernel.setArg(2, height);
    kernel.setArg(3, width);

    cl::NDRange global_size {(size_t) block_size * ( (height - 1)/block_size + 1), (size_t) block_size*((width - 1)/block_size + 1)};
    cl::NDRange local_size  {thread_size_row, thread_size_col};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
    cmd_queue.enqueueCopyBuffer(matrix_trans, matrix,
                                0, 0,
                                height*width*sizeof(float),
                                NULL, NULL);
}
