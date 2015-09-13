#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include <string>
#include <iostream>

#include "compute_enls_nobias.h"

using namespace nlsar;

TEST_CASE( "compute_enls_nobias", "[cl_kernels]" ) {

        // data setup
        const int height = 10;
        const int width = 10;
        const int dimensions = 2;
        const int nlooks = 1;

        std::vector<float> enls                (height*width,  1.0);
        std::vector<float> alphas              (height*width,  0.5);
        std::vector<float> wsums               (height*width,  2.0);
        std::vector<float> enls_nobias         (height*width, -1.0);
        std::vector<float> desired_enls_nobias (height*width,  1.33333);

        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        compute_enls_nobias KUT{block_size, context};

        // allocate memory
        cl::Buffer device_enls        {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height*width*sizeof(float), enls.data(),   NULL};
        cl::Buffer device_alphas      {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height*width*sizeof(float), alphas.data(), NULL};
        cl::Buffer device_wsums       {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height*width*sizeof(float), wsums.data(),  NULL};
        cl::Buffer device_enls_nobias {context, CL_MEM_READ_WRITE,                       height*width*sizeof(float), NULL,          NULL};

        KUT.run(cmd_queue, 
                device_enls,
                device_alphas,
                device_wsums,
                device_enls_nobias,
                height,
                width);

        cmd_queue.enqueueReadBuffer(device_enls_nobias, CL_TRUE, 0, height*width*sizeof(float), enls_nobias.data(), NULL, NULL);
        
        for(auto x : enls_nobias) {
            std::cout << x << ",";
        }

        // workaround, since Approx does not work with vectors
        bool flag = true;
        for(int i = 0; i < height*width; i++) {
            flag = flag && (enls_nobias[i] == Approx(desired_enls_nobias[i]).epsilon( 0.0001 ));
        }
        REQUIRE( ( flag ) );
}
