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

#ifndef BOXCAR_SUB_IMAGE_H
#define BOXCAR_SUB_IMAGE_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "insar_data.h"
#include "cl_wrapper/boxcar_wrapper.h"

int boxcar_sub_image(cl::Context context,
                     boxcar_wrapper boxcar_routine,
                     insar_data& sub_insar_data);

#endif
