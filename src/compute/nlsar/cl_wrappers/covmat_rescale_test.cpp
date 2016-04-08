/* Copyright 2015, 2016 Gerald Baier
 *
 * This file is part of despeckCL.
 *
 * despeckCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * despeckCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with despeckCL. If not, see <http://www.gnu.org/licenses/>.
 */

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include <string>
#include <iostream>

#include "covmat_rescale.h"

using namespace nlsar;

TEST_CASE( "covmat_rescale", "[cl_kernels]" ) {

        // data setup
        const int height = 10;
        const int width = 10;
        const int dimension = 2;
        const int nlooks = 1;

        const float gamma = std::pow(std::min(float(nlooks)/float(dimension), 1.0f), 0.333333);

        std::vector<float> inout_put      (2*height*width*dimension*dimension, 1.0);
        std::vector<float> desired_output (2*height*width*dimension*dimension, 1.0);

        for(int h = 0; h < height; h++) {
            for(int w = 0; w < width; w++) {
                for(int row_idx = 0; row_idx < dimension; row_idx++) {
                    for(int col_idx = 0; col_idx < dimension; col_idx++) {
                        if (row_idx != col_idx) {
                            const int idx = 2*(row_idx * dimension + col_idx)*height*width + h*width + w;
                            desired_output[idx]              = gamma * inout_put[idx];
                            desired_output[idx + height*width] = gamma * inout_put[idx + height*width];
                        }
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
        covmat_rescale KUT{block_size, context};

        // allocate memory
        cl::Buffer device_inout_put  {context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, 2*height*width*dimension*dimension*sizeof(float), inout_put.data(), NULL};

        KUT.run(cmd_queue, 
                device_inout_put,
                dimension,
                nlooks,
                height,
                width);

        cmd_queue.enqueueReadBuffer(device_inout_put, CL_TRUE, 0, 2*height*width*dimension*dimension*sizeof(float), inout_put.data(), NULL, NULL);

        // workaround, since Approx does not work with vectors
        bool flag = true;
        for(int i = 0; i < 2*height*width*dimension*dimension; i++) {
            flag = flag && (inout_put[i] == Approx(desired_output[i]).epsilon( 0.0001 ));
        }
        REQUIRE( ( flag ) );
}
