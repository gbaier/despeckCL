#include "best_alpha_copy.h"

std::vector<float> nlsar::best_alpha_copy(std::map<params, std::vector<float>> &alphas,
                                          std::vector<params> best_parameters,
                                          const int height,
                                          const int width)
{
    std::vector<float> best_alphas (height*width);
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            const params best_params = best_parameters[h*width + w];
            best_alphas[h*width + w] = alphas[best_params][h*width + w];
        }
    }
    return best_alphas;
}
