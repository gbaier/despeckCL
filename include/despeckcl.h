#ifndef DESPECKCL_H
#define DESPECKCL_H

#include <vector>
#include <string>

void boxcar(float* master_amplitude,
            float* slave_amplitude,
            float* dphase,
            float* ampl_filt,
            float* dphase_filt,
            float* coh_filt,
            const int height,
            const int width,
            const int window_width,
            std::vector<std::string> enabled_log_levels);

namespace nlsar {
    int nlsar(float* master_amplitude, float* slave_amplitude, float* dphase,
              float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered,
              const int height, const int width,
              const int search_window_size,
              const std::vector<int> patch_sizes,
              const std::vector<int> scale_sizes,
              std::vector<std::string> enabled_log_levels);
}

namespace nlinsar {
    struct routines {
        nlinsar::precompute_similarities_1st_pass* precompute_similarities_1st_pass_routine;
        nlinsar::precompute_similarities_2nd_pass* precompute_similarities_2nd_pass_routine;
        nlinsar::precompute_patch_similarities*    precompute_patch_similarities_routine;
        nlinsar::compute_weights*                  compute_weights_routine;
        compute_number_of_looks*                   compute_number_of_looks_routine;
        nlinsar::transpose*                        transpose_routine;
        nlinsar::precompute_filter_values*         precompute_filter_values_routine;
        nlinsar::compute_insar*                    compute_insar_routine;
        nlinsar::smoothing*                        smoothing_routine;
    };

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
