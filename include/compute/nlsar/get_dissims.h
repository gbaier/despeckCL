#ifndef GET_DISSIMS_H
#define GET_DISSIMS_H

#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "insar_data.h"

namespace nlsar {
    std::vector<float> get_dissims(cl::Context context,
                                   const insar_data& sub_insar_data,
                                   const int patch_size,
                                   const int window_width);
}

#endif
