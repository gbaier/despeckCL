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

#ifndef GET_DISSIMS_H
#define GET_DISSIMS_H

#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "data.h"
#include "cl_wrappers.h"

namespace nlsar {
    namespace training {
        std::vector<float> get_dissims(cl::Context context,
                                       nlsar::cl_wrappers nlsar_cl_wrappers,
                                       const insar_data& sub_insar_data,
                                       const int patch_size,
                                       const int window_width);
    }
}

#endif
