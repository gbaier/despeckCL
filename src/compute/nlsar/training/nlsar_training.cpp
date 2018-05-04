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

#include "despeckcl.h"

#include "logging.h"
#include "clcfg.h"
#include "insar_data.h"
#include "get_stats.h"

#include "cl_wrappers.h"

std::map<nlsar::params, nlsar::stats>
despeckcl::nlsar_training(float *ampl_master,
                          float *ampl_slave,
                          float *phase,
                          const int height,
                          const int width,
                          const std::vector<int> patch_sizes,
                          const std::vector<int> scale_sizes,
                          std::vector<std::string> enabled_log_levels)
{
  logging_setup(enabled_log_levels);

  float *dummy = (float*) malloc(height*width*sizeof(float));
  insar_data training_data{ampl_master,
                           ampl_slave,
                           phase,
                           dummy,
                           dummy,
                           dummy,
                           height,
                           width};
  free(dummy);

  size_t dummy_search_window_size = 21;
  size_t dimension = 2;

  cl::Context context = opencl_setup();

  nlsar::cl_wrappers nlsar_cl_wrappers(
      context, dummy_search_window_size, dimension);

  VLOG(0) << "Training weighting kernels";
  return nlsar::training::get_stats(
      patch_sizes, scale_sizes, training_data, context, nlsar_cl_wrappers);
}
