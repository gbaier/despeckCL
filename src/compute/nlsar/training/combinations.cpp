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

#include "combinations.h"

#include <vector>
#include <algorithm>

#include "patches.h"

std::vector<float> nlsar::training::get_all_dissim_combs(std::vector<data> patches, std::vector<float> acc)
{
    if (patches.size() == 1) {
        return acc;
    } else {
        data head = patches.back();
        patches.pop_back();
        std::vector<float> dissims {};
        std::transform(patches.begin(),
                       patches.end(),
                       std::back_inserter(dissims),
                       [&head] (data& x) { return dissimilarity(head, x); });
        acc.insert(acc.end(), dissims.begin(), dissims.end());
        return get_all_dissim_combs(patches, acc);
    }
}
