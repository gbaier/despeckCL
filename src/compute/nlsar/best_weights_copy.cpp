#include "best_weights_copy.h"

std::vector<float> nlsar::best_weights_copy(std::map<params, std::vector<float>> &weights,
                                            std::vector<params> best_parameters,
                                            const int height,
                                            const int width,
                                            const int search_window_size)
{
    std::vector<float> best_weights (search_window_size*search_window_size*height*width);
    std::map<params, float*> mapptr;
    for(auto& x : weights) {
        mapptr[x.first] = x.second.data();
    }
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            const params best_params = best_parameters[h*width + w];
            float* best_match = mapptr.at(best_params);
            for(int i = 0; i < search_window_size*search_window_size; i++) {
                const int idx = i*height*width + h*width + w;
                best_weights[idx] = best_match[idx];
            }
        }
    }
    return best_weights;
}
