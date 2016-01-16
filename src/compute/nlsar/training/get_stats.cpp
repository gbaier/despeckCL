#include "get_stats.h"

#include <future>

#include "get_dissims.h"

std::map<nlsar::params, nlsar::stats> nlsar::training::get_stats (const std::vector<int> patch_sizes,
                                                                  const std::vector<int> scale_sizes,
                                                                  const insar_data training_data,
                                                                  cl::Context context,
                                                                  nlsar::cl_wrappers nlsar_cl_wrappers)
{
    const int lut_size = 256;
    std::vector<nlsar::params> params;
    for(int patch_size : patch_sizes) {
        for(int scale_size : scale_sizes) {
            params.push_back(nlsar::params{patch_size, scale_size});
        }
    }

    std::map<nlsar::params, std::future<nlsar::stats>> futs;
    auto comp_stats = [context, nlsar_cl_wrappers, training_data, lut_size] (auto p) { return nlsar::stats(get_dissims(context,
                                                                                                                       nlsar_cl_wrappers,
                                                                                                                       training_data,
                                                                                                                       p.patch_size,
                                                                                                                       p.scale_size), lut_size);};
    for(auto p : params) {
        futs.emplace(p, std::async(std::launch::async, comp_stats, p));
    }

    std::map<nlsar::params, nlsar::stats> nlsar_stats;
    for(auto p : params) {
        nlsar_stats.emplace(p, futs[p].get());
    }

    return nlsar_stats;
}

