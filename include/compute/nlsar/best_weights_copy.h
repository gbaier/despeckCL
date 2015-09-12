#ifndef BEST_WEIGHTS_COPY_H
#define BEST_WEIGHTS_COPY_H

#include <vector>
#include <map>

#include "best_params.h"

namespace nlsar {
    std::vector<float> best_weights_copy(std::map<params, std::vector<float>> &weights,
                                         std::vector<params> best_parameters,
                                         const int height,
                                         const int width,
                                         const int search_window_size);
}

#endif
