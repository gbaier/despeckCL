#ifndef BOXCAR_SUB_IMAGE_H
#define BOXCAR_SUB_IMAGE_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "insar_data.h"
#include "cl_wrapper/boxcar_wrapper.h"

int boxcar_sub_image(cl::Context context,
                     boxcar_wrapper boxcar_routine,
                     insar_data& sub_insar_data,
                     const int window_width);

#endif
