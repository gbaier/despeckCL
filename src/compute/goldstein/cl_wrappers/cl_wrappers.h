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

#ifndef CL_WRAPPERS_H
#define CL_WRAPPERS_H

#include <CL/cl.h>

#include "patches_pack.h"
#include "patches_unpack.h"
#include "raw_interferogram.h"
#include "weighted_multiply.h"
#include "slc2real.h"

namespace goldstein {
    struct cl_wrappers {
        raw_interferogram raw_interferogram_routine;
        patches_unpack    patches_unpack_routine;
        weighted_multiply weighted_multiply_routine;
        patches_pack      patches_pack_routine;
        slc2real          slc2real_routine;

        cl_wrappers(cl::Context context,
                    const int block_size = 16);

        cl_wrappers(const cl_wrappers& other);
    };
}

#endif
