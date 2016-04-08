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

#ifndef GOLDSTEIN_PATCH_FT_H
#define GOLDSTEIN_PATCH_FT_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <clFFT.h>

void goldstein_patch_ft(cl::CommandQueue &cmd_queue,
                        clfftPlanHandle &plan_handle,
                        cl::Buffer interf_real,
                        cl::Buffer interf_imag,
                        const int height,
                        const int width,
                        const int patch_size,
                        clfftDirection dir);

#endif
