#ifndef NLSAR_SUB_IMAGE_H
#define NLSAR_SUB_IMAGE_H

#include "nlsar_routines.h"
#include "insar_data.h"
#include "stats.h"
#include "best_params.h"

namespace nlsar {
    int filter_sub_image(cl::Context context,
                         routines nl_routines,
                         insar_data& sub_insar_data,
                         const int search_window_size,
                         const std::vector<int> patch_sizes,
                         const int dimension,
                         std::map<params, stats> &dissim_stats);
}

#endif
