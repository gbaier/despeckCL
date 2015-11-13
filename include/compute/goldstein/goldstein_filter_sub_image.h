#ifndef GOLDSTEIN_SUB_IMAGE_H
#define GOLDSTEIN_SUB_IMAGE_H

#include "cl_wrappers.h"
#include "insar_data.h"
#include "timings.h"

namespace goldstein {
    timings::map filter_sub_image(cl::Context context,
                                  cl_wrappers gs_routines,
                                  insar_data& sub_insar_data,
                                  const int patch_size,
                                  const int overlap,
                                  const float alpha);
}

#endif
