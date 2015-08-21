#ifndef STATS_H
#define STATS_H

#include <vector>

class stats
{
    private:
        const int patch_size;
        const float h = 0.333;
        const float size = 1000;
        const float c;
        
        std::vector<float> dissims;
        std::vector<float> chi2cdf_inv;

        float dissim_lookup(float dissim);
        float chi2cdf_inv_lookup(float idx);

    public:
        stats(std::vector<float> dissims, unsigned int patch_size);
        float weight(float dissim);

};

#endif
