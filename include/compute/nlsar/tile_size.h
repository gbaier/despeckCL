#ifndef TILE_SIZE_H
#define TILE_SIZE_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <vector>

namespace nlsar {
    int round_down(const int num, const int multiple);

    int tile_size(cl::Context context,
                  const int search_window_size,
                  const std::vector<int>& patch_sizes,
                  const std::vector<int>& scale_sizes);
}

#endif
