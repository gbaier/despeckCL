#ifndef NLINSAR_OPENCL_H
#define NLINSAR_OPENCL_H

#include "utils/easylogging++.h"

#include "precompute_similarities.h"
#include "precompute_patch_similarities.h"
#include "compute_weights.h"
#include "compute_number_of_looks.h"
#include "transpose.h"
#include "precompute_filter_values.h"
#include "compute_insar.h"
#include "smoothing.h"


struct nlinsar_routines {
    precompute_similarities_1st_pass* precompute_similarities_1st_pass_routine;
    precompute_similarities_2nd_pass* precompute_similarities_2nd_pass_routine;
    precompute_patch_similarities*    precompute_patch_similarities_routine;
    compute_weights*                  compute_weights_routine;
    compute_number_of_looks*          compute_number_of_looks_routine;
    transpose*                        transpose_routine;
    precompute_filter_values*         precompute_filter_values_routine;
    compute_insar*                    compute_insar_routine;
    smoothing*                        smoothing_routine;
};

int nlinsar(float* master_amplitude, float* slave_amplitude, float* dphase,
            float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered,
            const int height,
            const int width,
            const int search_window_size,
            const int patch_size,
            const int niter,
            const int lmin,
            std::vector<el::Level> enabled_log_levels);
#endif
