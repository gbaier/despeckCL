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

#include "cl_wrappers.h"

nlinsar::cl_wrappers::cl_wrappers(cl::Context context,
                                  const int search_window_size,
                                  const int patch_size,
                                  const int block_size) : precompute_similarities_1st_pass_routine (block_size, context),
                                                          precompute_similarities_2nd_pass_routine (block_size, context),
                                                          precompute_patch_similarities_routine    (14,         context, patch_size),
                                                          compute_weights_routine                  (64,         context),
                                                          compute_number_of_looks_routine          (block_size, context),
                                                          transpose_routine                        (32,         context, 8, 32),
                                                          precompute_filter_values_routine         (block_size, context),
                                                          compute_insar_routine                    (14,         context, search_window_size)
{
}

nlinsar::cl_wrappers::cl_wrappers(const cl_wrappers& other) : precompute_similarities_1st_pass_routine (other.precompute_similarities_1st_pass_routine),
                                                              precompute_similarities_2nd_pass_routine (other.precompute_similarities_2nd_pass_routine),
                                                              precompute_patch_similarities_routine    (other.precompute_patch_similarities_routine),
                                                              compute_weights_routine                  (other.compute_weights_routine),
                                                              compute_number_of_looks_routine          (other.compute_number_of_looks_routine),
                                                              transpose_routine                        (other.transpose_routine),
                                                              precompute_filter_values_routine         (other.precompute_filter_values_routine),
                                                              compute_insar_routine                    (other.compute_insar_routine)
{
}
