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

namespace despeckcl {
void boxcar(float* ampl_master,
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

std::map<nlsar::params, nlsar::stats>
nlsar_training(float *ampl_master,
               float *ampl_slave,
               float *phase,
               const int height,
               const int width,
               const std::vector<int> patch_sizes,
               const std::vector<int> scale_sizes);

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
