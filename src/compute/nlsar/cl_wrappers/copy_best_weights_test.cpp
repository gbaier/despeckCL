#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <complex>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include "copy_best_weights.h"

using namespace nlsar;

TEST_CASE( "copy_best_weights", "[cl_kernels]" ) {

        // data setup
        const int height             = 30;
        const int width              = 50;
        const int nparams            = 3;
        const int search_window_size = 1;

        std::vector<float> all_weights          (nparams*height*width, -1.0);
        std::vector<int>   best_params          (        height*width, -1.0);
        std::vector<float> best_weights         (        height*width, -1.0);
        std::vector<float> desired_best_weights (        height*width, -1.0);

        for(int bp = 0; bp < nparams; bp++) {
            for(int i = 0; i < height*width; i++) {
                all_weights[bp*height*width + i] = bp;
            }
        }

        static std::default_random_engine rand_eng{};
        static std::uniform_int_distribution<int> dist_params(0, nparams-1);

        for(int i = 0; i < height*width; i++) {
            best_params[i] = dist_params(rand_eng);
            desired_best_weights[i] = best_params[i];
        }

        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        copy_best_weights KUT{block_size, context};

        // allocate memory
        cl::Buffer device_all_weights  {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nparams*height*width*sizeof(float), all_weights.data(), NULL};
        cl::Buffer device_best_params  {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,         height*width*sizeof(float), best_params.data(), NULL};
        cl::Buffer device_best_weights {context, CL_MEM_READ_WRITE,                               height*width*sizeof(float), NULL, NULL};

        KUT.run(cmd_queue, 
                device_all_weights,
                device_best_params,
                device_best_weights,
                height,
                width,
                search_window_size);

        cmd_queue.enqueueReadBuffer(device_best_weights, CL_TRUE, 0, height*width*sizeof(float), best_weights.data(), NULL, NULL);

        REQUIRE( ( best_weights == desired_best_weights ) );
}
