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

#include "best_params.h"

#include <random>

using namespace nlsar;
using testing::Pointwise;

TEST(best_params, random) {

    // data setup
    const int height = 10;
    const int width = 10;

    const params p1 {1,1};
    const params p2 {2,1};

    std::vector<float> enl1 (height*width);
    std::vector<float> enl2 (height*width);
    
    // simulate coherence value
    static std::default_random_engine rand_eng{};
    static std::uniform_real_distribution<float> dist {0, 1};

    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            const float rn = dist(rand_eng);
            enl1[h*width + w] = rn;
            enl2[h*width + w] = 2*rn;
        }
    }

    std::map<params, std::vector<float>> enl;
    enl[p1] = enl1;
    enl[p2] = enl2;

    std::vector<params> best_ps;
    get_best_params{}.run(enl, &best_ps, height, width);
    std::vector<params> desired_best_ps(height*width, p2);

    ASSERT_THAT(best_ps, Pointwise(testing::Eq(), desired_best_ps));
}
