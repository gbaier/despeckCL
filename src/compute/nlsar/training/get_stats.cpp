#include "get_stats.h"

#include "get_dissims.h"

std::map<nlsar::params, nlsar::stats> nlsar::training::get_stats (const std::vector<int> patch_sizes,
                                                                  const std::vector<int> scale_sizes,
                                                                  const insar_data training_data,
                                                                  cl::Context context)
{
    const int lut_size = 256;
    std::map<nlsar::params, nlsar::stats> nlsar_stats;
    for(int patch_size : patch_sizes) {
        for(int scale_size : scale_sizes) {
            std::vector<float> dissims  = get_dissims(context,
                                                      training_data,
                                                      patch_size,
                                                      scale_size);
            nlsar_stats.emplace(nlsar::params{patch_size, scale_size},
                                nlsar::stats(dissims, lut_size));
        }
    }
    return nlsar_stats;
}

