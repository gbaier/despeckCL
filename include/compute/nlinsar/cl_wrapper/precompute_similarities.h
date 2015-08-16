#ifndef PRECOMPUTE_SIMILARITIES_H
#define PRECOMPUTE_SIMILARITIES_H

#include "clcfg.h"

class precompute_similarities_1st_pass : public kernel_env<precompute_similarities_1st_pass>
{
    public:
        static constexpr const char* routine_name {"precompute_similarities_1st_pass"};

        using kernel_env::kernel_env;

        void run(cl::CommandQueue cmd_queue,
                 cl::Buffer device_a1,
                 cl::Buffer device_a2,
                 cl::Buffer device_dp,
                 cl::Buffer device_amp_filt,
                 cl::Buffer device_phi_filt,
                 cl::Buffer device_coh_filt,
                 const int height_overlap,
                 const int width_overlap,
                 const int search_window_size,
                 cl::Buffer device_similarities,
                 cl::Buffer device_kullback_leiblers);
};

class precompute_similarities_2nd_pass : public kernel_env<precompute_similarities_2nd_pass>
{
    public:
        static constexpr const char* routine_name {"precompute_similarities_2nd_pass"};

        using kernel_env::kernel_env;

        void run(cl::CommandQueue cmd_queue,
                 const int height_overlap,
                 const int width_overlap,
                 const int search_window_size,
                 cl::Buffer device_similarities,
                 cl::Buffer device_kullback_leiblers);
};

#endif
