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

using nlsar::params;

void nlsar::get_best_params::run(std::map<params, std::vector<float>> &enl,
                                 std::vector<params>* best_parameters,
                                 const int height,
                                 const int width)
{
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            std::pair<params, float> best_param {{0,0}, -1.0f};
            // get best parameter for a single pixel
            for(auto const& param_and_enl : enl) {
                if(param_and_enl.second[h*width+w] > best_param.second) {
                    best_param.first  = param_and_enl.first;
                    best_param.second = param_and_enl.second[h*width+w];
                }
            }
            best_parameters->push_back(best_param.first);
        }
    }
}
