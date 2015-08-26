#ifndef NLSAR_H
#define NLSAR_H

#include "utils/easylogging++.h"

int nlsar(float* master_amplitude, float* slave_amplitude, float* dphase,
          float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered,
          const int height, const int width,
          const int search_window_size,
          const int patch_size,
          std::vector<el::Level> enabled_log_levels);

#endif
