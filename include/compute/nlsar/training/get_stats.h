#ifndef GET_STATS_H
#define GET_STATS_H

#include <vector>
#include <map>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "insar_data.h"
#include "stats.h"
#include "parameters.h"
#include "cl_wrappers.h"

namespace nlsar {
    namespace training {
        std::map<nlsar::params, nlsar::stats> get_stats (const std::vector<int> patch_sizes,
                                                         const std::vector<int> scale_sizes,
                                                         const insar_data training_data,
                                                         cl::Context context,
                                                         nlsar::cl_wrappers nlsar_cl_wrappers);
    }
}


#endif
