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

#ifndef NLSAR_SUB_IMAGE_H
#define NLSAR_SUB_IMAGE_H

#include "cl_wrappers.h"
#include "data.h"
#include "stats.h"
#include "best_params.h"
#include "timings.h"
#include "cl_wrappers.h"

namespace nlsar {
    timings::map filter_sub_image(cl::Context context,
                                  cl_wrappers nlsar_cl_wrappers,
                                  insar_data& sub_insar_data,
                                  const int search_window_size,
                                  const std::vector<int> patch_sizes,
                                  const std::vector<int> scale_sizes,
                                  const int dimension,
                                  std::map<params, stats> &dissim_stats);
}

#endif
