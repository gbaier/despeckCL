#ifndef NLSAR_ROUTINES_H
#define NLSAR_ROUTINES_H

#include <CL/cl.h>

#include "covmat_create.h"
#include "covmat_rescale.h"
#include "covmat_spatial_avg.h"
#include "compute_pixel_similarities_2x2.h"
#include "compute_patch_similarities.h"
#include "covmat_decompose.h"
#include "weighted_means.h"

struct nlsar_routines {
    covmat_create                  covmat_create_routine;
    covmat_rescale                 covmat_rescale_routine;
    covmat_spatial_avg             covmat_spatial_avg_routine;
    compute_pixel_similarities_2x2 compute_pixel_similarities_2x2_routine;
    compute_patch_similarities     compute_patch_similarities_routine;
    covmat_decompose               covmat_decompose_routine;
    weighted_means                 weighted_means_routine;

    nlsar_routines(cl::Context context,
                   const int search_window_size,
                   const int patch_size,
                   const int window_width,
                   const int dimension,
                   const int block_size = 16);

    nlsar_routines(const nlsar_routines& other);
};

#endif
