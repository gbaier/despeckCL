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

#include "best_params.h"

#include <algorithm>
#include <numeric>
#include <iterator>     // std::back_inserter

using nlsar::params;

std::vector<params> nlsar::get_best_params(const std::map<params, std::vector<float>> &enl)
{
  size_t nelem_ori = (*enl.begin()).second.size();

  std::vector<float> best_enl(nelem_ori, -1.0f);
  std::vector<params> best_parameters(nelem_ori, params{0, 0});

  // can possibly be written more succinctly using std::reduce, which needs
  // C++17
  for (auto const& param_and_enl : enl) {
    std::vector<bool> mask;
    std::transform(std::begin(best_enl),
                   std::end(best_enl),
                   std::begin(param_and_enl.second),
                   std::back_inserter(mask),
                   [](float enl1, float enl2) { return enl2 > enl1; });

    auto cbp     = std::begin(best_parameters);
    auto mask_it = std::begin(mask);

    while (cbp != std::end(best_parameters)) {
      if (*mask_it++) {
        *cbp++ = param_and_enl.first;
      }
    }
  }
  return best_parameters;
}
