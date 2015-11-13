#ifndef CL_WRAPPERS_H
#define CL_WRAPPERS_H

#include <CL/cl.h>

#include "patches_pack.h"
#include "patches_unpack.h"
#include "raw_interferogram.h"
#include "weighted_multiply.h"
#include "slc2real.h"

namespace goldstein {
    struct cl_wrappers {
        raw_interferogram raw_interferogram_routine;
        patches_unpack    patches_unpack_routine;
        weighted_multiply weighted_multiply_routine;
        patches_pack      patches_pack_routine;
        slc2real          slc2real_routine;

        cl_wrappers(cl::Context context,
                    const int block_size = 16);

        cl_wrappers(const cl_wrappers& other);
    };
}

#endif
