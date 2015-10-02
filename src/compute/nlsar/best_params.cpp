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
            std::pair<params, float> best_param {{0,0}, -1.0f};
            // get best parameter for a single pixel
            for(auto const& param_and_enl : enl) {
                if(param_and_enl.second[h*width+w] > best_param.second) {
                    best_param.first  = param_and_enl.first;
                    best_param.second = param_and_enl.second[h*width+w];
                }
            }
            best_parameters->push_back(best_param.first);
        }
    }
}
