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

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "unit_test_helper.h"

#include "best_weights_copy.h"

#include <random>

using namespace nlsar;
using testing::Pointwise;

TEST(beist_weights_copy, random) {

    // data setup
    const int height = 20;
    const int width = 20;
    const int search_window_size = 21;

    const std::vector<params> ps {
                                  {1,1},
                                  {2,1},
                                  {3,1}
                                 };
    const int size = ps.size();
    // simulate coherence value
    static std::default_random_engine rand_eng{};
    static std::discrete_distribution<> dist_disc(0, size);
    std::vector<params> best_params;
    for(int i = 0; i < height*width; i++) {
       best_params.push_back( ps[dist_disc(rand_eng)] ); 
    }

    static std::uniform_real_distribution<float> dist_float {0, 1};
    std::vector<float> weights1             (search_window_size*search_window_size*height*width);
    std::vector<float> weights2             (search_window_size*search_window_size*height*width);
    std::vector<float> weights3             (search_window_size*search_window_size*height*width);
    for(int i = 0; i < height*width*search_window_size*search_window_size; i++) {
        weights1[i] = dist_float(rand_eng);
        weights2[i] = dist_float(rand_eng);
        weights3[i] = dist_float(rand_eng);
    }
    std::map<params, std::vector<float>> weights;
    weights[ps[0]] = weights1;
    weights[ps[1]] = weights2;
    weights[ps[2]] = weights3;

    std::vector<float> desired_best_weights (search_window_size*search_window_size*height*width);
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            const params best = best_params[h*width + w];
            const int offset = height*width;
            for(int i = 0; i < search_window_size*search_window_size; i++) {
                desired_best_weights[i*offset + h*width + w] = weights[best][i*offset + h*width + w];
            }
        }
    }
    
    std::vector<float> best_weights = best_weights_copy(weights,
                                                        best_params,
                                                        height,
                                                        width,
                                                        search_window_size);

    ASSERT_THAT(best_weights, Pointwise(FloatNearPointwise(1e-4), desired_best_weights));
}
