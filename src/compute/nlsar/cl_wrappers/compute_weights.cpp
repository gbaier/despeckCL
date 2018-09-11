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

#include "compute_weights.h"

constexpr const char* nlsar::compute_weights::routine_name;
constexpr const char* nlsar::compute_weights::kernel_source;

nlsar::compute_weights::compute_weights(const size_t block_size,
                                        cl::Context context,
                                        const float h_param,
                                        const float c_param)
    : kernel_env_single<compute_weights>(block_size, context),
      h_param(h_param),
      c_param(c_param)
{
  program = build_program(build_opts(), kernel_source);
  kernel  = build_kernel(program, routine_name);
}

nlsar::compute_weights::compute_weights(const compute_weights& other)
    : kernel_env_single<compute_weights>(other),
      h_param(other.h_param),
      c_param(other.c_param)
{
  program = other.program;
  kernel  = build_kernel(program, routine_name);
}

void nlsar::compute_weights::run(cl::CommandQueue cmd_queue,
                                 cl::Buffer patch_similarities,
                                 cl::Buffer weights,
                                 const int height_symm,
                                 const int width_symm,
                                 const int search_window_size,
                                 const int patch_size,
                                 cl::Buffer dissims2relidx,
                                 cl::Buffer chi2cdf_inv,
                                 const int lut_size,
                                 const float dissims_min,
                                 const float dissims_max)
{
    kernel.setArg( 0, patch_similarities);
    kernel.setArg( 1, weights);
    kernel.setArg( 2, height_symm);
    kernel.setArg( 3, width_symm);
    kernel.setArg( 4, search_window_size);
    kernel.setArg( 5, patch_size);
    kernel.setArg( 6, this->h_param);
    kernel.setArg( 7, this->c_param);
    kernel.setArg( 8, dissims2relidx);
    kernel.setArg( 9, chi2cdf_inv);
    kernel.setArg(10, lut_size);
    kernel.setArg(11, dissims_min);
    kernel.setArg(12, dissims_max);

    const int wsh = (search_window_size-1)/2;

    cl::NDRange global_size {(size_t) block_size*( ((search_window_size*wsh + wsh)*height_symm*width_symm - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
