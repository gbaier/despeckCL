/* Copyright 2018 Gerald Baier
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

#include "data.h"
#include "cl_wrappers.h"
#include "clcfg.h"
#include "logging.h"
#include "tile_size.h"

namespace nlsar {

// Various overloads for SAR, InSAR or PolSAR data

cl::Buffer data_to_covmat(const insar_data& sub_insar_data,
                          const cl::Context& context,
                          const cl::CommandQueue& cmd_queue,
                          covmat_create& cl_routine,
                          const buffer_sizes& buf_sizes);

cl::Buffer data_to_covmat(const ampl_data& sub_data,
                          const cl::Context& context,
                          const cl::CommandQueue& cmd_queue,
                          const buffer_sizes& buf_sizes);

void covmat_to_data(const cl::Buffer& covmat_filt,
                    insar_data& sub_insar_data,
                    const cl::Context& context,
                    const cl::CommandQueue& cmd_queue,
                    covmat_decompose& cl_routine,
                    const buffer_sizes& buf_sizes);

void covmat_to_data(const cl::Buffer& covmat_filt,
                    ampl_data& sub_data,
                    const cl::CommandQueue& cmd_queue,
                    const buffer_sizes& buf_sizes);
}  // namespace nlsar
