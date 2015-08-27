#ifndef NLSAR_SUB_IMAGE_H
#define NLSAR_SUB_IMAGE_H

#include "nlsar_sub_image.h"

#include "nlsar_routines.h"
#include "insar_data.h"
#include "stats.h"

int nlsar_sub_image(cl::Context context,
                    nlsar_routines nl_routines,
                    insar_data& sub_insar_data,
                    const int search_window_size,
                    const std::vector<int> patch_sizes,
                    const int dimension,
                    stats dissim_stats);

#endif
