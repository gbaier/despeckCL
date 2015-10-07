#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include <string>
#include <iostream>
#include <vector>

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

TEST_CASE( "covmat_spatial_avg no averaging", "[cl_kernels]" ) {

        // data setup
        const int height = 30;
        const int width = 20;

        const int scale_size = 1;
        const int scale_size_max = 5;
        const int delta_scale = (scale_size_max - scale_size)/2;
        const int dimension = 2;

        // simulate coherence value
        static std::default_random_engine rand_eng{};
        static std::uniform_real_distribution<float> dist {0, 1};


        std::vector<float> input          (2*dimension*dimension*height*width, 0.0);
        const int n_elems_out = 2*dimension*dimension*(height-scale_size_max+1)*(width-scale_size_max+1);

        std::vector<float> output         (n_elems_out,  0.0);
        std::vector<float> desired_output (n_elems_out, -1.0);

        for(int d = 0; d < 2*dimension*dimension; d++) {
            for(int h = 0; h < height; h++) {
                for(int w = 0; w < width; w++) {
                    const float randn = dist(rand_eng);
                    input[d*height*width+h*width + w] = randn;
                    if ( h >= delta_scale && w >= delta_scale && h < height-delta_scale && w < width-delta_scale) {
                        const int hh = h - delta_scale;
                        const int ww = w - delta_scale;
                        desired_output[d*(height-scale_size_max+1)*(width-scale_size_max+1) \
                                                              + hh*(width-scale_size_max+1)
                                                                                      + ww] = randn;
                    }
                }
            }
        }

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
        cl::Buffer device_output {context, CL_MEM_READ_WRITE, 2*dimension*dimension*(height-scale_size_max+1)*(width-scale_size_max+1)*sizeof(float), NULL,         NULL};

        KUT.run(cmd_queue, 
                device_input,
                device_output,
                dimension,
                height-scale_size_max+1,
                width-scale_size_max+1,
                scale_size,
                scale_size_max);

        cmd_queue.enqueueReadBuffer(device_output, CL_TRUE, 0, 2*dimension*dimension*(height-scale_size_max+1)*(width-scale_size_max+1)*sizeof(float), output.data(), NULL, NULL);

        REQUIRE( ( output == desired_output ) );
}

TEST_CASE( "covmat_spatial_avg gauss", "[cl_kernels]" ) {

        const int scale_size = 5;
        cl::Context context = opencl_setup();
        const int block_size = 16;
        covmat_spatial_avg KUT{block_size, context};

        std::vector<float> gauss {KUT.gen_gauss(scale_size)};
        std::vector<float> gauss_rev = gauss;

        std::reverse(gauss_rev.begin(), gauss_rev.end());

        REQUIRE( gauss == gauss_rev );
        REQUIRE( gauss.size() == scale_size*scale_size );
        REQUIRE( Approx(std::accumulate(gauss.begin(), gauss.end(), 0.0f)) == 1.0f );
}
