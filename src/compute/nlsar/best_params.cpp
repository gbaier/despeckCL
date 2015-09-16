#include "best_params.h"

#include <algorithm>

namespace nlsar {
    struct params_enl {
        const params p;
        const float enl;
    };

    params get_best_param(std::vector<params_enl> penl_map) {
        params_enl best = *std::max_element(penl_map.begin(), penl_map.end(), [] (params_enl e1, params_enl e2) { return e1.enl < e2.enl; });
        return best.p;
    }
}

using nlsar::params;

std::vector<params> nlsar::best_params(std::map<params, std::vector<float>> &enl,
                                       const int height,
                                       const int width)
{
    std::vector<params> best_ps;
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            std::vector<params_enl> penl_map;
            // get best parameter for a single pixel
            for(auto const& param : enl) {
                penl_map.push_back(params_enl{param.first, param.second[h*width + w]});
            }
            best_ps.push_back(get_best_param(penl_map));
        }
    }
    return best_ps;
}
