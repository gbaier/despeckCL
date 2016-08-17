/* Copyright 2015, 2016 Gerald Baier
 *
 * This file is part of despeckCL.
 *
 * despeckCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * despeckCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with despeckCL. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SMOOTHING_H
#define SMOOTHING_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "../compute_env.h"

namespace nlinsar {

    class smoothing : public routine_env<smoothing>
    {
        std::string routine_name{"smoothing"};

        public:
            void run(cl::CommandQueue cmd_queue,
                     cl::Buffer device_weights,
                     cl::Buffer device_nols,
                     float * ampl_master,
                     const int height_ori,
                     const int width_ori,
                     const int search_window_size,
                     const int patch_size,
                     const int lmin);
    };

    void search_window_smoothing(const float * amplitude_master,
                                 float * weights,
                                 const int h,
                                 const int w,
                                 const int width,
                                 const int patch_size,
                                 const int search_window_size,
                                 const int lmin);
}
#endif
