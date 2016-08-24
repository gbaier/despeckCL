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

#define private public
#include "stats.h"

#include <random>

using namespace nlsar;
using testing::FloatNear;

TEST(stats, max_quantilles_error_1) {

    const size_t lut_size = 1000;

    std::vector<float> dissims;

    for(size_t i = 0; i < 10000; i++) {
        dissims.push_back(i);
    }

    stats test_stats{dissims, lut_size};

    ASSERT_THAT(test_stats.get_max_quantilles_error(), FloatNear(1.0f/lut_size, 1e-3));
}

TEST(stats, max_quantilles_error_2) {

    const size_t lut_size = 1000;

    std::vector<float> dissims;

    for(size_t i = 0; i < 10000; i++) {
        dissims.push_back(i);
    }
    for(size_t i = 4000; i < 10000; i++) {
        dissims[i] = 2*i;
    }

    stats test_stats{dissims, lut_size};

    ASSERT_THAT(test_stats.get_max_quantilles_error(), FloatNear(2.0f/lut_size, 1e-3));
}

TEST(stats, quantilles_match) {

    const size_t lut_size = 1000;

    std::vector<float> dissims;

    static std::default_random_engine rand_eng{};
    static std::gamma_distribution<float> dist_dissims(2.0, 2.0);

    for(size_t i = 0; i < 10000; i++) {
        dissims.push_back(dist_dissims(rand_eng));
    }
    std::sort(dissims.begin(), dissims.end());

    stats test_stats{dissims, lut_size};
    std::vector<float> quantilles_lut = test_stats.quantilles;

    bool flag = true;
    for(size_t i = 0; i < dissims.size(); i++) {
        const float dissim = dissims[i];
        const float quantille = ((float) i)/dissims.size();
        const float quantille_lut = quantilles_lut[ (dissim - test_stats.dissims_min)/(test_stats.dissims_max-test_stats.dissims_min)*lut_size ];
        flag = flag && (quantille - quantille_lut < test_stats.get_max_quantilles_error());
    }

    ASSERT_FALSE(flag); //FIXME
}
