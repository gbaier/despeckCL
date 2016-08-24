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

#include <complex>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "unit_test_helper.h"

#include "sim_measures.h"

using testing::FloatEq;

TEST(sim_measures, 2x2_mat_determinant) {

        std::complex<float> el00{2,  0};
        std::complex<float> el01{4,  3};
        std::complex<float> el10{4, -3};
        std::complex<float> el11{2,  0};

        std::complex<float> det = el00*el11 - el01*el10;

        ASSERT_THAT(det_covmat_2x2(1, 0, 0, 1), FloatEq(1.0f));
        ASSERT_THAT(std::imag(det), FloatEq(0.0f));
        ASSERT_THAT(std::real(det), FloatEq(det_covmat_2x2(std::real(el00), std::real(el01), std::imag(el01), std::real(el11))));
}
