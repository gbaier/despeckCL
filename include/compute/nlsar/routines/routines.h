#ifndef ROUTINES_H
#define ROUTINES_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "stats.h"
#include "cl_wrappers.h"

using nlsar::cl_wrappers;

namespace nlsar {
    namespace routines {
        cl::Buffer get_weights (cl::Buffer& pixel_similarities,
                                cl::Context context,
                                const int height_sim,
                                const int width_ori,
                                const int search_window_size,
                                const int patch_size,
                                const int patch_size_max,
                                stats& parameter_stats,
                                cl::Buffer& lut_dissims2relidx,
                                cl::Buffer& lut_chi2cdf_inv,
                                cl_wrappers& nl_routines);
    }
}

#endif
