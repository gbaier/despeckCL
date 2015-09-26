#ifndef NLINSAR_OPENCL_H
#define NLINSAR_OPENCL_H

#include <vector>
#include <string>

namespace nlinsar {

    int nlinsar(float* master_amplitude, float* slave_amplitude, float* dphase,
                float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered,
                const int height,
                const int width,
                const int search_window_size,
                const int patch_size,
                const int niter,
                const int lmin,
                std::vector<std::string> enabled_log_levels);
}

#endif
