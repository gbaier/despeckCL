#ifndef INSARSIM_SIMU_H
#define INSARSIM_SIMU_H

#include <vector>
#include <tuple>

namespace nlinsar {
    namespace simu {

        float quantile(std::vector<float> vector, float alpha);
        
        std::tuple<float, float, float> insar_gen(void);

        float quantile_insar(int patch_size, float alpha);
    }
}

#endif
