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
