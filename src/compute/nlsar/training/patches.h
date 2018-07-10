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

#ifndef PATCHES_H
#define PATCHES_H

#include <vector>
#include <cstdint>

#include "data.h"

namespace nlsar
{
namespace training
{
covmat_data get_patch(const covmat_data& training_data,
                      int upper_h,
                      int left_w,
                      int patch_size);

std::vector<covmat_data> get_all_patches(const covmat_data& training_data,
                                         int patch_size);

float dissimilarity(const covmat_data& first, const covmat_data& second);
}  // namespace training
}  // namespace nlsar

#endif
