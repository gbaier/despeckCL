#ifndef BEST_ALPHAS_COPY_H
#define BEST_ALPHAS_COPY_H

#include <vector>
#include <map>

#include "best_params.h"

namespace nlsar {
    std::vector<float> best_alpha_copy(std::map<params, std::vector<float>> &alphas,
                                       std::vector<params> best_parameters,
                                       const int height,
                                       const int width);
}

#endif
