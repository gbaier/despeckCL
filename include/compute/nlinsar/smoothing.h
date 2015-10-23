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
