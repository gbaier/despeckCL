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

#include "best_weights_copy.h"

std::vector<float> nlsar::best_weights_copy(std::map<params, std::vector<float>> &weights,
                                            std::vector<params> best_parameters,
                                            const int height,
                                            const int width,
                                            const int search_window_size)
{
    std::vector<float> best_weights (search_window_size*search_window_size*height*width);
    std::map<params, float*> mapptr;
    for(auto& x : weights) {
        mapptr[x.first] = x.second.data();
    }
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            const params best_params = best_parameters[h*width + w];
            float* best_match = mapptr.at(best_params);
            for(int i = 0; i < search_window_size*search_window_size; i++) {
                const int idx = i*height*width + h*width + w;
                best_weights[idx] = best_match[idx];
            }
        }
    }
    return best_weights;
}
