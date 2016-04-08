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

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <complex>

#include "sim_measures.h"

TEST_CASE( "2x2 matrix determinant", "[similarity_measures]" ) {

        std::complex<float> el00{2,  0};
        std::complex<float> el01{4,  3};
        std::complex<float> el10{4, -3};
        std::complex<float> el11{2,  0};

        std::complex<float> det = el00*el11 - el01*el10;

        REQUIRE( ( det_covmat_2x2(1, 0, 0, 1) == 1 ) );
        REQUIRE( ( 0.0 == Approx(std::imag(det))) );
        REQUIRE( ( std::real(det) == det_covmat_2x2(std::real(el00), std::real(el01), std::imag(el01), std::real(el11)) ) );
}
