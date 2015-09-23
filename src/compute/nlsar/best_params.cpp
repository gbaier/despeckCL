#include "best_params.h"

#include <algorithm>

using nlsar::params;

void nlsar::get_best_params::run(std::map<params, std::vector<float>> &enl,
                                 std::vector<params>* best_parameters,
                                 const int height,
                                 const int width)
{
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            std::vector<std::pair<params, float>> params_enls;
            // get best parameter for a single pixel
            for(auto const& param : enl) {
                params_enls.push_back({param.first, param.second[h*width + w]});
            }
            best_parameters->push_back(get_best_pixel_params(params_enls));
        }
    }
}

params nlsar::get_best_params::get_best_pixel_params(std::vector<std::pair<params, float>> params_enls) {
        auto best = *std::max_element(params_enls.begin(), params_enls.end(), [] (auto e1, auto e2) { return e1.second < e2.second; });
        return best.first;
}
