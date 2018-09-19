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

#ifndef NLINSAR_CL_WRAPPERS_H
#define NLINSAR_CL_WRAPPERS_H

#include <CL/cl.h>

#include "precompute_similarities_1st_pass.h"
#include "precompute_similarities_2nd_pass.h"
#include "precompute_patch_similarities.h"
#include "compute_weights.h"
#include "compute_number_of_looks.h"
#include "transpose.h"
#include "precompute_filter_values.h"
#include "compute_insar.h"
#include "smoothing.h"

namespace nlinsar {
    struct kernel_params {
        const int search_window_size;
        const int patch_size;
        const int block_size;
    };

    struct cl_wrappers {
        precompute_similarities_1st_pass precompute_similarities_1st_pass_routine;
        precompute_similarities_2nd_pass precompute_similarities_2nd_pass_routine;
        precompute_patch_similarities    precompute_patch_similarities_routine;
        compute_weights                  compute_weights_routine;
        compute_number_of_looks          compute_number_of_looks_routine;
        transpose                        transpose_routine;
        smoothing                        smoothing_routine;
        precompute_filter_values         precompute_filter_values_routine;
        compute_insar                    compute_insar_routine;

        cl_wrappers(cl::Context context,
                    const kernel_params kp) : precompute_similarities_1st_pass_routine (kp.block_size, context),
                                              precompute_similarities_2nd_pass_routine (kp.block_size, context),
                                              precompute_patch_similarities_routine    (14,            context, kp.patch_size),
                                              compute_weights_routine                  (64,            context),
                                              compute_number_of_looks_routine          (kp.block_size, context),
                                              transpose_routine                        (32,            context, 8, 32),
                                              precompute_filter_values_routine         (kp.block_size, context),
                                              compute_insar_routine                    (14,            context, kp.search_window_size) {};

        cl_wrappers(const cl_wrappers& other) : precompute_similarities_1st_pass_routine (other.precompute_similarities_1st_pass_routine),
                                                precompute_similarities_2nd_pass_routine (other.precompute_similarities_2nd_pass_routine),
                                                precompute_patch_similarities_routine    (other.precompute_patch_similarities_routine),
                                                compute_weights_routine                  (other.compute_weights_routine),
                                                compute_number_of_looks_routine          (other.compute_number_of_looks_routine),
                                                transpose_routine                        (other.transpose_routine),
                                                precompute_filter_values_routine         (other.precompute_filter_values_routine),
                                                compute_insar_routine                    (other.compute_insar_routine) {};
    };

    cl_wrappers get_cl_wrappers(cl::Context cl_context, kernel_params pm);
}

#endif
