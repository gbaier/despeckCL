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

goldstein::cl_wrappers::cl_wrappers(cl::Context context,
                                const int block_size) : raw_interferogram_routine (block_size, context),
                                                        patches_unpack_routine    (block_size, context),
                                                        weighted_multiply_routine (block_size, context),
                                                        patches_pack_routine      (block_size, context),
                                                        slc2real_routine          (block_size, context)
{
}

goldstein::cl_wrappers::cl_wrappers(const cl_wrappers& other) : raw_interferogram_routine (other.raw_interferogram_routine),
                                                                patches_unpack_routine    (other.patches_unpack_routine),
                                                                weighted_multiply_routine (other.weighted_multiply_routine),
                                                                patches_pack_routine      (other.patches_pack_routine),
                                                                slc2real_routine          (other.slc2real_routine)
{
}

