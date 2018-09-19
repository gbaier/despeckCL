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

#ifndef GOLDSTEIN_CL_WRAPPERS_H
#define GOLDSTEIN_CL_WRAPPERS_H

#include <CL/cl.h>

#include "patches_pack.h"
#include "patches_unpack.h"
#include "raw_interferogram.h"
#include "weighted_multiply.h"
#include "slc2real.h"

namespace goldstein {
    struct kernel_params {
        const int block_size = 16;
    };

    struct cl_wrappers {
        raw_interferogram raw_interferogram_routine;
        patches_unpack    patches_unpack_routine;
        weighted_multiply weighted_multiply_routine;
        patches_pack      patches_pack_routine;
        slc2real          slc2real_routine;

        cl_wrappers(cl::Context context,
                    kernel_params kp) : raw_interferogram_routine (kp.block_size, context),
                                        patches_unpack_routine    (kp.block_size, context),
                                        weighted_multiply_routine (kp.block_size, context),
                                        patches_pack_routine      (kp.block_size, context),
                                        slc2real_routine          (kp.block_size, context) {};

        cl_wrappers(const cl_wrappers& other) : raw_interferogram_routine (other.raw_interferogram_routine),
                                                patches_unpack_routine    (other.patches_unpack_routine),
                                                weighted_multiply_routine (other.weighted_multiply_routine),
                                                patches_pack_routine      (other.patches_pack_routine),
                                                slc2real_routine          (other.slc2real_routine) {};
    };

    cl_wrappers get_cl_wrappers(cl::Context cl_context, kernel_params pm);
}

#endif
