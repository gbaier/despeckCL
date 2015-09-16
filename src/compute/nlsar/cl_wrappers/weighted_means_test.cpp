#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include <string>
#include <iostream>

#include "weighted_means.h"

using namespace nlsar;

TEST_CASE( "weighted_means", "[cl_kernels]" ) {

        // data setup
        const int height_ori = 10;
        const int width_ori = 10;

        const int search_window_size = 11;
        const int patch_size = 3;
        const int window_width = 3;
        const int overlap = (patch_size-1)/2 + (search_window_size-1)/2;
        const int overlap_avg = overlap + (window_width-1)/2;
        const int dimension = 2;

        const int covmat_in_nelem  = (height_ori + 2*overlap_avg) * (width_ori + 2*overlap_avg) * dimension * dimension * 2;
        const int covmat_out_nelem = (height_ori + 2*overlap_avg) * (width_ori + 2*overlap_avg) * dimension * dimension * 2;
        const int weights_nelem    =  height_ori                  *  width_ori * search_window_size * search_window_size;

        std::vector<float> covmat_in          (covmat_in_nelem,    1.0);
        std::vector<float> covmat_out         (covmat_out_nelem,   0.0);
        std::vector<float> desired_covmat_out (covmat_out_nelem,   0.0);
        for(int h = overlap_avg; h < height_ori + overlap_avg; h++) {
            for(int w = overlap_avg; w < width_ori + overlap_avg; w++) {
                for(int d = 0; d < 2*dimension*dimension; d++) {
                    desired_covmat_out[d*(height_ori + 2*overlap_avg)*(width_ori + 2*overlap_avg) + h*(width_ori + 2*overlap_avg) + w] = 1.0;
                }
            }
        }
        std::vector<float> weights            (weights_nelem,      1.0);

        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        weighted_means KUT{block_size, context, search_window_size, dimension};

        // allocate memory
        cl::Buffer device_covmat_in  {context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, covmat_in_nelem  * sizeof(float), covmat_in.data(), NULL};
        cl::Buffer device_weights    {context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, weights_nelem    * sizeof(float), weights.data(),   NULL};
        cl::Buffer device_covmat_out {context, CL_MEM_READ_WRITE,                        covmat_out_nelem * sizeof(float), NULL,             NULL};

        KUT.run(cmd_queue, 
                device_covmat_in,
                device_covmat_out,
                device_weights,
                height_ori,
                width_ori,
                search_window_size,
                patch_size,
                window_width);

        cmd_queue.enqueueReadBuffer(device_covmat_out, CL_TRUE, 0, covmat_out_nelem * sizeof(float), covmat_out.data(), NULL, NULL);

        REQUIRE( ( covmat_out == desired_covmat_out ) );
}
