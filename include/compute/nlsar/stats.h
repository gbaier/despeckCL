#ifndef STATS_H
#define STATS_H

#include <vector>
#include <map>

class stats
{
    private:
        const int patch_size;
        const float h = 0.333;
        const float c;
        float dissim_lookup(float dissim);
        float chi2cdf_inv_lookup(float idx);

    public:
        std::vector<float> dissims2relidx;
        std::vector<float> chi2cdf_inv;
        const unsigned int lut_size;
        float dissims_min;
        float dissims_max;
        stats(std::vector<float> dissims, unsigned int patch_size, unsigned int lut_size);
};

#endif
