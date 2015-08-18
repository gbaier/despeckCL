#ifndef COVMAT_SPATIAL_AVG_H
#define COVMAT_SPATIAL_AVG_H

#include "clcfg.h"

class covmat_spatial_avg : public kernel_env<covmat_spatial_avg>
{
    public:
        const int window_width;
        const int output_block_size;

        static constexpr const char* kernel_source =
"__kernel void covmat_spatial_avg (__global float * covmat_in,\n"
"                                  __global float * covmat_out, \n"
"                                  const int dimension,\n"
"                                  const int height,\n"
"                                  const int width)\n"
"{\n"
"    const int window_radius = (WINDOW_WIDTH - 1) / 2;\n"
"    const int tx = get_local_id(0);\n"
"    const int ty = get_local_id(1);\n"
"\n"
"    const int in_x = get_group_id(0) * OUTPUT_BLOCK_SIZE + tx;\n"
"    const int in_y = get_group_id(1) * OUTPUT_BLOCK_SIZE + ty;\n"
"\n"
"    const int out_x = in_x + window_radius;\n"
"    const int out_y = in_y + window_radius;\n"
"\n"
"    __local float local_data [BLOCK_SIZE][BLOCK_SIZE];\n"
"\n"
"    for(int i = 0; i < (dimension * (dimension+1))/2; i++) {\n"
"        if ( (in_x < height) && (in_y < width) ) {\n"
"            local_data [tx][ty] = covmat_in [i*height*width + in_x*width + in_y];\n"
"        }\n"
"\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"        float sum = 0;\n"
"\n"
"        if ((tx < OUTPUT_BLOCK_SIZE) && (ty < OUTPUT_BLOCK_SIZE)) {\n"
"            for(int kx = 0; kx < WINDOW_WIDTH; kx++) {\n"
"                for(int ky = 0; ky < WINDOW_WIDTH; ky++) {\n"
"                    sum += local_data[tx + kx][ty + ky];\n"
"                }\n"
"            }\n"
"            if (out_x < height - window_radius && out_y < width - window_radius) {\n"
"                covmat_out[i*height*width + out_x*width + out_y] = sum/(WINDOW_WIDTH*WINDOW_WIDTH);\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
;

        static constexpr const char* routine_name {"covmat_spatial_avg"};

        covmat_spatial_avg(const size_t block_size,
                           cl::Context context,
                           const int window_width);

        covmat_spatial_avg(const covmat_spatial_avg& other);

        std::string return_build_options(void);

        void run(cl::CommandQueue cmd_queue,
                 cl::Buffer covmat_in,
                 cl::Buffer covmat_out,
                 const int dimension,
                 const int height,
                 const int width);
};

#endif
