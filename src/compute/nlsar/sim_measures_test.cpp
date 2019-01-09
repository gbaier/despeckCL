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
  std::complex<float> el00{2, 0};
  std::complex<float> el01{4, 3};
  std::complex<float> el10{4, -3};
  std::complex<float> el11{2, 0};

  std::complex<float> det = el00 * el11 - el01 * el10;

  ASSERT_THAT(det_covmat_2x2(1, 0, 0, 1), FloatEq(1.0f));
  ASSERT_THAT(std::imag(det), FloatEq(0.0f));
  ASSERT_THAT(
      std::real(det),
      FloatEq(det_covmat_2x2(
          std::real(el00), std::real(el01), std::imag(el01), std::real(el11))));
}

using CovMat3x3 = std::array<std::array<std::complex<float>, 3>, 3>;

struct TestPair
{
    CovMat3x3 cm;
    float det; // determinatn
};

// vector with test values
std::vector<TestPair> test_values{
    TestPair{{{{{{3, 0}, {0, 0}, {0, 0}}},
               {{{0, 0}, {2, 0}, {0, 0}}},
               {{{0, 0}, {0, 0}, {1, 0}}}}},
             6.0f},
    TestPair{{{{{{2, 0}, {4, 3}, {1, 2}}},
               {{{4, -3}, {3, 0}, {1, 1}}},
               {{{1, -2}, {1, -1}, {1, 0}}}}},
             -8.0f},
    TestPair{{{{{{3, 0}, {0, 3}, {2, -1}}},
               {{{0, -3}, {2, 0}, {0, 0}}},
               {{{2, 1}, {0, 0}, {1, 0}}}}},
             -13.0f},
    TestPair{{{{{{3, 0}, {0, 3}, {2, -1}}},
               {{{0, -3}, {5, 0}, {1, 1}}},
               {{{2, 1}, {1, -1}, {1, 0}}}}},
             -43.0f},
    TestPair{{{{{{4, 0}, {0, 1}, {2, -1}}},
               {{{0, -1}, {5, 0}, {1, 1}}},
               {{{2, 1}, {1, -1}, {3, 0}}}}},
             18.0f},
    TestPair{{{{{{5,  0}, {0, -4}, {-4, 0}}},
               {{{0,  4}, {5,  0}, {0, -4}}},
               {{{-4, 0}, {0,  4}, {5,  0}}}}},
             13.0f},
};

struct CovMatTest : public ::testing::TestWithParam<TestPair> {
  CovMat3x3 cov_mat;
  float det;

  CovMatTest()
  {
    cov_mat = GetParam().cm;
    det     = GetParam().det;
  }
};

TEST_P(CovMatTest, determinant)
{
    float test_det = det_covmat_3x3(std::real(cov_mat[0][0]),
                                    std::real(cov_mat[1][1]),
                                    std::real(cov_mat[2][2]),
                                    std::real(cov_mat[1][0]),
                                    std::imag(cov_mat[1][0]),
                                    std::real(cov_mat[2][0]),
                                    std::imag(cov_mat[2][0]),
                                    std::real(cov_mat[2][1]),
                                    std::imag(cov_mat[2][1]));

    ASSERT_THAT(std::imag(test_det), FloatEq(0.0f));
    ASSERT_THAT(std::real(test_det), FloatEq(det));
}

INSTANTIATE_TEST_CASE_P(Default, CovMatTest, testing::ValuesIn(test_values));
