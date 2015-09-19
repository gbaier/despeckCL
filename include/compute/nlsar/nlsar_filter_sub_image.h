#ifndef NLSAR_SUB_IMAGE_H
#define NLSAR_SUB_IMAGE_H

#include "cl_wrappers.h"
#include "insar_data.h"
#include "stats.h"
#include "best_params.h"
#include "timings.h"

namespace nlsar {
    timings::map filter_sub_image(cl::Context context,
                                  cl_wrappers nlsar_cl_wrappers,
                                  insar_data& sub_insar_data,
                                  const int search_window_size,
                                  const int dimension,
                                  std::map<params, stats> &dissim_stats);
}

#endif
