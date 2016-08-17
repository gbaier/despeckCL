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

#ifndef ROUTINES_H
#define ROUTINES_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "stats.h"
#include "cl_wrappers.h"
#include "timings.h"

using nlsar::cl_wrappers;

namespace nlsar {
    namespace routines {
        timings::map get_pixel_similarities (cl::Context context,
                                             cl::Buffer& covmat_rescaled,
                                             cl::Buffer& device_pixel_similarities,
                                             const int height_overlap,
                                             const int width_overlap,
                                             const int dimensions,
                                             const int nlooks,
                                             const int search_window_size,
                                             const int scale_size,
                                             const int scale_size_max,
                                             cl_wrappers& nl_routines);

        timings::map get_weights (cl::Context context,
                                  cl::Buffer& pixel_similarities,
                                  cl::Buffer& weights,
                                  const int height_sim,
                                  const int width_ori,
                                  const int search_window_size,
                                  const int patch_size,
                                  const int patch_size_max,
                                  stats& parameter_stats,
                                  cl::Buffer& lut_dissims2relidx,
                                  cl::Buffer& lut_chi2cdf_inv,
                                  cl_wrappers& nl_routines);

        timings::map get_enls_nobias_and_alphas (cl::Context context,
                                                 cl::Buffer& device_weights,
                                                 cl::Buffer& device_covmat_ori,
                                                 cl::Buffer& device_enls_nobias,
                                                 cl::Buffer& device_alphas,
                                                 const int height_ori,
                                                 const int width_ori,
                                                 const int search_window_size,
                                                 const int patch_size,
                                                 const int scale_size_max,
                                                 const int nlooks,
                                                 const int dimension,
                                                 cl_wrappers& nl_routines);
    }
}

#endif
