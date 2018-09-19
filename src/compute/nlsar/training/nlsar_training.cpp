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
#include "data.h"
#include "get_stats.h"

#include "cl_wrappers.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

using nlsar_stats_collection = std::map<nlsar::params, nlsar::stats>;

nlsar_stats_collection
nlsar_training(covmat_data& training_data,
               const std::vector<int> patch_sizes,
               const std::vector<int> scale_sizes,
               std::vector<std::string> enabled_log_levels)
{
  logging_setup(enabled_log_levels);
  auto cl_devs = get_platform_devs(0);
  cl::Context cl_context (cl_devs);

  size_t dummy_search_window_size = 21;
  size_t dummy_h_param = 5.0f;
  size_t dummy_c_param = 49.0f;
  nlsar::kernel_params nkp{dummy_search_window_size, training_data.dim(), dummy_h_param, dummy_c_param, 16};
  nlsar::cl_wrappers nlsar_cl_wrappers(cl_context, nkp);

  VLOG(0) << "Training weighting kernels";
  return nlsar::training::get_stats(
      patch_sizes, scale_sizes, training_data, cl_context, nlsar_cl_wrappers);
}


nlsar_stats_collection
despeckcl::nlsar_training(float *ampl_master,
                          float *ampl_slave,
                          float *phase,
                          const int height,
                          const int width,
                          const std::vector<int> patch_sizes,
                          const std::vector<int> scale_sizes,
                          std::vector<std::string> enabled_log_levels)
{
  auto dummy = std::make_unique<float[]>(height*width);
  covmat_data training_data{insar_data{ampl_master,
                                       ampl_slave,
                                       phase,
                                       dummy.get(),
                                       dummy.get(),
                                       dummy.get(),
                                       height,
                                       width}};

  return nlsar_training(
      training_data, patch_sizes, scale_sizes, enabled_log_levels);
}


nlsar_stats_collection
despeckcl::nlsar_training(float *covmat,
                          const int height,
                          const int width,
                          const int dim,
                          const std::vector<int> patch_sizes,
                          const std::vector<int> scale_sizes,
                          std::vector<std::string> enabled_log_levels)
{
  auto dummy = std::make_unique<float[]>(2*dim*dim*height*width);
  covmat_data training_data{covmat, dummy.get(), height, width, dim};

  return nlsar_training(
      training_data, patch_sizes, scale_sizes, enabled_log_levels);
}


void despeckcl::store_nlsar_stats_collection(nlsar_stats_collection nsc, std::string filename)
{
    std::ofstream ofs(filename);
    boost::archive::text_oarchive oa(ofs);
    oa << nsc;
}

nlsar_stats_collection despeckcl::load_nlsar_stats_collection(std::string filename)
{
    std::map<nlsar::params, nlsar::stats> nsc;
    std::ifstream ifs(filename);
    boost::archive::text_iarchive ia(ifs);
    ia >> nsc;
    return nsc;
}
