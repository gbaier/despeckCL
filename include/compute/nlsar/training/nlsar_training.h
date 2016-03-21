#ifndef NLSAR_TRAINING_H
#define NLSAR_TRAINING_H

#include <vector>
#include <map>

#include "parameters.h"
#include "stats.h"

namespace nlsar {
std::map<params, stats>
nlsar_training(float *ampl_master,
               float *ampl_slave,
               float *dphase,
               const int height,
               const int width,
               const std::vector<int> patch_sizes,
               const std::vector<int> scale_sizes);
}

#endif
