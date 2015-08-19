#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include <string>
#include <iostream>

#include "covmat_spatial_avg.h"

TEST_CASE( "covmat_spatial_avg", "[cl_kernels]" ) {

        // data setup
        const int height = 30;
        const int width = 20;

        const int window_width = 5;

        std::vector<float> input          (height*width, 1.0);
        std::vector<float> output         ((height-window_width+1)*(width-window_width+1), 0.0);
        std::vector<float> desired_output ((height-window_width+1)*(width-window_width+1), 1.0);

        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        covmat_spatial_avg KUT{block_size, context, window_width};

        // allocate memory
        cl::Buffer device_input  {context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, height*width*sizeof(float),                                   input.data(), NULL};
        cl::Buffer device_output {context, CL_MEM_READ_WRITE,                        (height-window_width+1)*(width-window_width+1)*sizeof(float), NULL,         NULL};

        KUT.run(cmd_queue, 
                device_input,
                device_output,
                1,
                height,
                width);

        cmd_queue.enqueueReadBuffer(device_output, CL_TRUE, 0, (height-window_width+1)*(width-window_width+1)*sizeof(float), output.data(), NULL, NULL);

        REQUIRE( ( output == desired_output ) );
}
