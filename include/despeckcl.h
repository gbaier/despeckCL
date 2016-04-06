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
            float* dphase,
            float* ampl_filt,
            float* dphase_filt,
            float* coh_filt,
            const int height,
            const int width,
            const int window_width,
            std::vector<std::string> enabled_log_levels);

int nlsar(float* ampl_master,
          float* ampl_slave,
          float* dphase,
          float* ampl_filt,
          float* dphase_filt,
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
               float *dphase,
               const int height,
               const int width,
               const std::vector<int> patch_sizes,
               const std::vector<int> scale_sizes);

int nlinsar(float* ampl_master,
            float* ampl_slave,
            float* dphase,
            float* ampl_filt,
            float* dphase_filt,
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
              float* dphase,
              float* ampl_filt,
              float* dphase_filt,
              float* coh_filt,
              const unsigned int height,
              const unsigned int width,
              const unsigned int patch_size,
              const unsigned int overlap,
              const float alpha,
              std::vector<std::string> enabled_log_levels);
}

#endif
