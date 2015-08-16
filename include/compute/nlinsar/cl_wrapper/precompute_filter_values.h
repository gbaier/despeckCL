#ifndef PRECOMPUTE_FILTER_VALUES_H
#define PRECOMPUTE_FILTER_VALUES_H

#include "clcfg.h"

class precompute_filter_values : public kernel_env<precompute_filter_values>
{
    public:
        static constexpr const char* routine_name {"precompute_filter_values"};

        using kernel_env::kernel_env;

        void run(cl::CommandQueue cmd_queue,
                 cl::Buffer device_a1,
                 cl::Buffer device_a2,
                 cl::Buffer device_dp,
                 cl::Buffer device_filter_values_a,
                 cl::Buffer device_filter_values_x_real,
                 cl::Buffer device_filter_values_x_imag,
                 const int height_overlap,
                 const int width_overlap,
                 const int patch_size);
};

#endif
