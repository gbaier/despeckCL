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

#ifndef DESPECKCL_H
#define DESPECKCL_H

#include <vector>
#include <string>
#include <map>

#include "nlsar/parameters.h"
#include "nlsar/stats.h"

using nlsar_stats_collection = std::map<nlsar::params, nlsar::stats>;

namespace despeckcl {
int boxcar(float* ampl_master,
           float* ampl_slave,
           float* phase,
           float* ref_filt,
           float* phase_filt,
           float* coh_filt,
           const int height,
           const int width,
           const int window_width,
           std::vector<std::string> enabled_log_levels);

int nlsar(float* ampl_master,
          float* ampl_slave,
          float* phase,
          float* ref_filt,
          float* phase_filt,
          float* coh_filt,
          const int height,
          const int width,
          const int search_window_size,
          const std::vector<int> patch_sizes,
          const std::vector<int> scale_sizes,
          std::map<nlsar::params, nlsar::stats> nlsar_stats,
          std::vector<std::string> enabled_log_levels);


int nlsar(float* covmat_raw,
          float* covmat_filt,
          const int height,
          const int width,
          const int dim,
          const int search_window_size,
          const std::vector<int> patch_sizes,
          const std::vector<int> scale_sizes,
          std::map<nlsar::params, nlsar::stats> nlsar_stats,
          std::vector<std::string> enabled_log_levels);

nlsar_stats_collection
nlsar_training(float *ampl_master,
               float *ampl_slave,
               float *phase,
               const int height,
               const int width,
               const std::vector<int> patch_sizes,
               const std::vector<int> scale_sizes,
               std::vector<std::string> enabled_log_levels);

void store_nlsar_stats_collection(nlsar_stats_collection nsc, std::string filename);
nlsar_stats_collection load_nlsar_stats_collection(std::string filename);

int nlinsar(float* ampl_master,
            float* ampl_slave,
            float* phase,
            float* ref_filt,
            float* phase_filt,
            float* coh_filt,
            const int height,
            const int width,
            const int search_window_size,
            const int patch_size,
            const int niter,
            const int lmin,
            std::vector<std::string> enabled_log_levels);

int goldstein(float* ampl_master,
              float* ampl_slave,
              float* phase,
              float* ref_filt,
              float* phase_filt,
              float* coh_filt,
              const unsigned int height,
              const unsigned int width,
              const unsigned int patch_size,
              const unsigned int overlap,
              const float alpha,
              std::vector<std::string> enabled_log_levels);
}

#endif
