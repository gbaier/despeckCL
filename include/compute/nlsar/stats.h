#ifndef STATS_H
#define STATS_H

#include <vector>

namespace nlsar {
    class stats
    {
        public:
            stats(std::vector<float> dissims, unsigned int lut_size);
            const unsigned int lut_size;
            float dissims_min;
            float dissims_max;
            std::vector<float> quantilles;
            std::vector<float> chi2cdf_inv;

        private:
            std::vector<float> get_quantilles(std::vector<float> &dissims);
            std::vector<float> get_chi2cdf_inv(void);
    };
}

#endif
