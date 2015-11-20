#ifndef TILE_SIZE_H
#define TILE_SIZE_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace goldstein {
    int round_down(const int num, const int multiple);

    int tile_size(cl::Context context,
                  const int patch_size,
                  const int overlap);
}

#endif
