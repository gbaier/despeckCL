#ifndef BOXCAR_WRAPPER_H
#define BOXCAR_WRAPPER_H

#include <CL/cl.h>

#include "clcfg.h"

class boxcar_wrapper : public kernel_env<boxcar_wrapper>
{

    public:
        const int window_width;
        const int output_block_size;

        static constexpr const char* routine_name {"boxcar_kernel"};

        boxcar_wrapper(const size_t block_size,
               cl::Context context,
               const int window_width);

        boxcar_wrapper(const boxcar_wrapper& precompiled);

        std::string return_build_options(void);

        void run(cl::CommandQueue cmd_queue,
                 cl::Buffer ampl_master,
                 cl::Buffer ampl_slave,
                 cl::Buffer dphase,
                 cl::Buffer ampl_filt,
                 cl::Buffer dphase_filt,
                 cl::Buffer coh_filt,
                 const int height,
                 const int width);
};

#endif
