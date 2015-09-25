#ifndef NLINSAR_SUB_IMAGE_H
#define NLINSAR_SUB_IMAGE_H

#include "nlinsar.h"
#include "insar_data.h"

#include <CL/cl.hpp>

int nlinsar_sub_image(cl::Context context,
                      nlinsar::routines nl_routines,
                      insar_data& sub_insar_data,
                      const int search_window_size,
                      const int patch_size,
                      const int lmin,
                      const float h_para,
                      const float T_para);

#endif
