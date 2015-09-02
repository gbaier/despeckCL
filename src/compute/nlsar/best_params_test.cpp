#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "best_params.h"

TEST_CASE( "covmat_create", "[cl_kernels]" ) {

    // data setup
    const int height = 10;
    const int width = 10;

    const params p1 {1,1,1};
    const params p2 {2,1,1};

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

    std::vector<params> best_ps = best_params(enl, height, width);
    std::vector<params> desired_best_ps(height*width, p2);

    REQUIRE( ( best_ps == desired_best_ps ) );
}
