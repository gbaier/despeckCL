#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include <string>
#include <iostream>

#include "covmat_spatial_avg.h"

using namespace nlsar;

TEST_CASE( "covmat_spatial_avg", "[cl_kernels]" ) {

        // data setup
        const int height = 30;
        const int width = 20;

        const int scale_size = 5;
        const int scale_size_max = scale_size;
        const int dimension = 2;

        std::vector<float> input          (2*dimension*dimension*height*width, 1.0);
        std::vector<float> output         (2*dimension*dimension*(height-scale_size+1)*(width-scale_size+1), 0.0);
        std::vector<float> desired_output (2*dimension*dimension*(height-scale_size+1)*(width-scale_size+1), 1.0);

        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        covmat_spatial_avg KUT{block_size, context};

        // allocate memory
        cl::Buffer device_input  {context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,            2*dimension*dimension*height*width*sizeof(float), input.data(), NULL};
        cl::Buffer device_output {context, CL_MEM_READ_WRITE, 2*dimension*dimension*(height-scale_size+1)*(width-scale_size+1)*sizeof(float), NULL,         NULL};

        KUT.run(cmd_queue, 
                device_input,
                device_output,
                dimension,
                height-scale_size+1,
                width-scale_size+1,
                scale_size,
                scale_size_max);

        cmd_queue.enqueueReadBuffer(device_output, CL_TRUE, 0, 2*dimension*dimension*(height-scale_size+1)*(width-scale_size+1)*sizeof(float), output.data(), NULL, NULL);

        REQUIRE( ( output == desired_output ) );
}
