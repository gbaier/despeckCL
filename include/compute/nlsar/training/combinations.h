#ifndef COMBINATIONS_H
#define COMBINATIONS_H

#include <vector>

#include "patches.h"

namespace nlsar {
    namespace training {
        std::vector<float> get_all_dissim_combs(std::vector<data> patches, std::vector<float> acc = {});
    }
}

#endif
