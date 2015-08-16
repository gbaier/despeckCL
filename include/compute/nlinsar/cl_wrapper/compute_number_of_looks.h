#ifndef COMPUTE_NUMBER_OF_LOOKS_H
#define COMPUTE_NUMBER_OF_LOOKS_H

#include "clcfg.h"

class compute_number_of_looks : public kernel_env<compute_number_of_looks>
{
    public:
        static constexpr const char* routine_name {"compute_number_of_looks"};

        using kernel_env::kernel_env;

        void run(cl::CommandQueue cmd_queue,
                 cl::Buffer weights,
                 cl::Buffer nols,
                 const int height_ori,
                 const int width_ori,
                 const int search_window_size);
};

#endif
