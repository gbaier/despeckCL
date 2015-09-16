#ifndef NLSAR_ROUTINES_H
#define NLSAR_ROUTINES_H

#include <CL/cl.h>

#include "covmat_create.h"
#include "covmat_rescale.h"
#include "covmat_spatial_avg.h"
#include "compute_pixel_similarities_2x2.h"
#include "compute_patch_similarities.h"
#include "compute_weights.h"
#include "compute_number_of_looks.h"
#include "compute_nl_statistics.h"
#include "compute_alphas.h"
#include "compute_enls_nobias.h"
#include "covmat_decompose.h"
#include "weighted_means.h"

namespace nlsar {
    struct routines {
        covmat_create                  covmat_create_routine;
        covmat_rescale                 covmat_rescale_routine;
        covmat_spatial_avg             covmat_spatial_avg_routine;
        compute_pixel_similarities_2x2 compute_pixel_similarities_2x2_routine;
        compute_patch_similarities     compute_patch_similarities_routine;
        compute_weights                compute_weights_routine;
        compute_number_of_looks        compute_number_of_looks_routine;
        compute_nl_statistics          compute_nl_statistics_routine;
        compute_alphas                 compute_alphas_routine;
        compute_enls_nobias            compute_enls_nobias_routine;
        covmat_decompose               covmat_decompose_routine;
        weighted_means                 weighted_means_routine;

        routines(cl::Context context,
                       const int search_window_size,
                       const int dimension,
                       const int block_size = 16);

        routines(const routines& other);
    };
}

#endif
