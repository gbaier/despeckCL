#ifndef NLSAR_H
#define NLSAR_H

#include "utils/easylogging++.h"

#include "covmat_create.h"
#include "covmat_rescale.h"
#include "covmat_spatial_avg.h"
#include "compute_pixel_similarities_2x2.h"
#include "compute_patch_similarities.h"
#include "weighted_means.h"

struct nlsar_routines {
    covmat_create*                  covmat_create_routine;
    covmat_rescale*                 covmat_rescale_routine;
    covmat_spatial_avg*             covmat_spatial_avg_routine;
    compute_pixel_similarities_2x2* compute_pixel_similarities_2x2_routine;
    compute_patch_similarities*     compute_patch_similarities_routine;
    weighted_means*                 weighted_means_routine;
};

int nlsar(float* master_amplitude, float* slave_amplitude, float* dphase,
          float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered,
          const int height, const int width,
          const int search_window_size,
          const int patch_size,
          std::vector<el::Level> enabled_log_levels);

#endif
