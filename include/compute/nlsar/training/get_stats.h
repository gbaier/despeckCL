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

#ifndef GET_STATS_H
#define GET_STATS_H

#include <vector>
#include <map>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "insar_data.h"
#include "stats.h"
#include "parameters.h"
#include "cl_wrappers.h"

namespace nlsar {
    namespace training {
        std::map<nlsar::params, nlsar::stats> get_stats (const std::vector<int> patch_sizes,
                                                         const std::vector<int> scale_sizes,
                                                         const insar_data training_data,
                                                         cl::Context context,
                                                         nlsar::cl_wrappers nlsar_cl_wrappers);
    }
}


#endif
