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

#ifndef NLSAR_CL_WRAPPERS_H
#define NLSAR_CL_WRAPPERS_H

#include <CL/cl.h>

#include "covmat_create.h"
#include "covmat_rescale.h"
#include "covmat_spatial_avg.h"
#include "compute_pixel_similarities_2x2.h"
#include "compute_pixel_similarities_3x3.h"
#include "compute_patch_similarities.h"
#include "compute_weights.h"
#include "compute_number_of_looks.h"
#include "compute_nl_statistics.h"
#include "compute_alphas.h"
#include "compute_enls_nobias.h"
#include "copy_best_weights.h"
#include "copy_symm_weights.h"
#include "covmat_decompose.h"
#include "weighted_means.h"

namespace nlsar {

    struct kernel_params {
        const int search_window_size;
        const int dimension;
        const float h_param = 15.0f;
        const float c_param = 49.0f;
        const int block_size = 16;
    };

    struct cl_wrappers {
        covmat_create                  covmat_create_routine;
        covmat_rescale                 covmat_rescale_routine;
        covmat_spatial_avg             covmat_spatial_avg_routine;
        compute_pixel_similarities_2x2 compute_pixel_similarities_2x2_routine;
        compute_pixel_similarities_3x3 compute_pixel_similarities_3x3_routine;
        compute_patch_similarities     compute_patch_similarities_routine;
        compute_weights                compute_weights_routine;
        compute_number_of_looks        compute_number_of_looks_routine;
        compute_nl_statistics          compute_nl_statistics_routine;
        compute_alphas                 compute_alphas_routine;
        compute_enls_nobias            compute_enls_nobias_routine;
        copy_best_weights              copy_best_weights_routine;
        copy_symm_weights              copy_symm_weights_routine;
        covmat_decompose               covmat_decompose_routine;
        weighted_means                 weighted_means_routine;

        cl_wrappers(cl::Context context,
                    const kernel_params kp) : covmat_create_routine                  (kp.block_size, context),
                                              covmat_rescale_routine                 (kp.block_size, context),
                                              covmat_spatial_avg_routine             (kp.block_size, context),
                                              compute_pixel_similarities_2x2_routine (kp.block_size, context),
                                              compute_pixel_similarities_3x3_routine (kp.block_size, context),
                                              compute_patch_similarities_routine     (context, 16, 4, 4, 4),
                                              compute_weights_routine                (64, context, kp.h_param, kp.c_param),
                                              compute_number_of_looks_routine        (kp.block_size, context),
                                              compute_nl_statistics_routine          (kp.block_size, context, kp.search_window_size, kp.dimension),
                                              compute_alphas_routine                 (kp.block_size, context),
                                              compute_enls_nobias_routine            (kp.block_size, context),
                                              copy_best_weights_routine              (64, context),
                                              copy_symm_weights_routine              (-1, context),
                                              covmat_decompose_routine               (kp.block_size, context),
                                              weighted_means_routine                 (kp.block_size, context, kp.search_window_size, kp.dimension) {};

    };

    cl_wrappers get_cl_wrappers(cl::Context cl_context, kernel_params pm);
}

#endif
