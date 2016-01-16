#include "get_stats.h"

#include "get_dissims.h"

std::map<nlsar::params, nlsar::stats> nlsar::training::get_stats (const std::vector<int> patch_sizes,
                                                                  const std::vector<int> scale_sizes,
                                                                  const insar_data training_data,
                                                                  cl::Context context)
{
    const int lut_size = 256;
    std::vector<nlsar::params> params;
    for(int patch_size : patch_sizes) {
        for(int scale_size : scale_sizes) {
            params.push_back(nlsar::params{patch_size, scale_size});
        }
    }

    std::map<nlsar::params, nlsar::stats> nlsar_stats;
    auto comp_stats = [&] (auto p) { return nlsar::stats(get_dissims(context, training_data, p.patch_size, p.scale_size), lut_size);};
    for(auto p : params) {
        nlsar_stats.emplace(p, comp_stats(p));
    }

    return nlsar_stats;
}

