#ifndef NLINSAR_SUB_IMAGE_H
#define NLINSAR_SUB_IMAGE_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "insar_data.h"
#include "cl_wrappers.h"

namespace nlinsar {
    int nlinsar_sub_image(cl::Context context,
                          cl_wrappers nl_routines,
                          insar_data& sub_insar_data,
                          const int search_window_size,
                          const int patch_size,
                          const int lmin,
                          const float h_para,
                          const float T_para);

}
#endif
