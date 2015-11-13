#ifndef GOLDSTEIN_PATCH_FT_H
#define GOLDSTEIN_PATCH_FT_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <clFFT.h>

void goldstein_patch_ft(cl::CommandQueue &cmd_queue,
                        clfftPlanHandle &plan_handle,
                        cl::Buffer interf_real,
                        cl::Buffer interf_imag,
                        const int height,
                        const int width,
                        const int patch_size,
                        clfftDirection dir);

#endif
