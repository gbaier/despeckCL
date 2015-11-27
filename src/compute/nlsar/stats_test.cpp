#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#define private public
#include "stats.h"

using namespace nlsar;

TEST_CASE( "max_quantilles_error_1", "[stats]" ) {

    const size_t lut_size = 1000;

    std::vector<float> dissims;

    for(size_t i = 0; i < 10000; i++) {
        dissims.push_back(i);
    }

    stats test_stats{dissims, lut_size};

    REQUIRE( (1.0f/lut_size == Approx(test_stats.get_max_quantilles_error()).epsilon(0.0001)) );
}

TEST_CASE( "max_quantilles_error_2", "[stats]" ) {

    const size_t lut_size = 1000;

    std::vector<float> dissims;

    for(size_t i = 0; i < 10000; i++) {
        dissims.push_back(i);
    }
    for(size_t i = 4000; i < 10000; i++) {
        dissims[i] = 2*i;
    }

    stats test_stats{dissims, lut_size};

    REQUIRE( (2.0f/lut_size == Approx(test_stats.get_max_quantilles_error()).epsilon(0.0001)) );
}

TEST_CASE( "quantilles_match", "[stats]" ) {

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
    REQUIRE( (flag) );
}


