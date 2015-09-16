#ifndef NLSAR_H
#define NLSAR_H

#include <vector>
#include <string>

namespace nlsar {
    int nlsar(float* master_amplitude, float* slave_amplitude, float* dphase,
              float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered,
              const int height, const int width,
              const int search_window_size,
              const std::vector<int> patch_sizes,
              const std::vector<int> scale_sizes,
              std::vector<std::string> enabled_log_levels);
}

#endif
