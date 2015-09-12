#ifndef STATS_H
#define STATS_H

#include <vector>
#include <map>

namespace nlsar {
    struct stats
    {
        std::vector<float> dissims2relidx;
        std::vector<float> chi2cdf_inv;
        const unsigned int lut_size;
        float dissims_min;
        float dissims_max;
        stats(std::vector<float> dissims, unsigned int patch_size, unsigned int lut_size);
    };
}

#endif
