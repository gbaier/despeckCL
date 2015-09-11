#ifndef STATS_H
#define STATS_H

#include <vector>
#include <map>

class stats
{
    private:
        const int patch_size;
        const unsigned int lut_size;
        const float h = 0.333;
        const float c;
        float dissims_min;
        float dissims_max;
        
        std::vector<float> dissims2relidx;
        std::vector<float> chi2cdf_inv;

        float dissim_lookup(float dissim);
        float chi2cdf_inv_lookup(float idx);

    public:
        stats(std::vector<float> dissims, unsigned int patch_size, unsigned int lut_size);
        float weight(float dissim);

};

#endif
