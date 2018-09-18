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

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "unit_test_helper.h"

#include "compute_enls_nobias.h"

using namespace nlsar;
using testing::FloatEq;
using testing::Each;

TEST(compute_enls_nobias, single_val_check) {

        // data setup
        const int height = 10;
        const int width = 10;

        std::vector<float> enls                (height*width,  1.0);
        std::vector<float> alphas              (height*width,  0.5);
        std::vector<float> wsums               (height*width,  2.0);
        std::vector<float> enls_nobias         (height*width, -1.0);
        const float desired_enl_nobias = 1.333333333;

        // opencl setup
        auto cl_devs = get_platform_devs(0);
        cl::Context context{cl_devs};
        cl::CommandQueue cmd_queue{context};

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
        
        ASSERT_THAT(enls_nobias, Each(FloatEq(desired_enl_nobias)));
}
