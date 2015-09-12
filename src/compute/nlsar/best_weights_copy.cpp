#include "best_weights_copy.h"

std::vector<float> nlsar::best_weights_copy(std::map<params, std::vector<float>> &weights,
                                            std::vector<params> best_parameters,
                                            const int height,
                                            const int width,
                                            const int search_window_size)
{
    std::vector<float> best_weights (search_window_size*search_window_size*height*width);
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            const int offset = height*width;
            const params best_params = best_parameters[h*width + w];
            for(int i = 0; i < search_window_size*search_window_size; i++) {
                best_weights[i*offset + h*width + w] = weights[best_params][i*offset + h*width + w];
            }
        }
    }
    return best_weights;
}
