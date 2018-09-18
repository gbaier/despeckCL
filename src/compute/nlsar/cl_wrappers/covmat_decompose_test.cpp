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

#include <complex>
#include <cmath>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "unit_test_helper.h"

#include "covmat_decompose.h"
#include "covmat_create.h"

using namespace nlsar;
using testing::Pointwise;

TEST(covmat_decompose, random_vals) {

        // data setup
        const int height = 10;
        const int width  = 10;
        const int dimension = 2;

        const int nelems = height*width;

        std::vector<float> ampl_master (                      height*width, 1.0);
        std::vector<float> ampl_slave  (                      height*width, 1.0);
        std::vector<float> phase      (                      height*width, 1.0);
        std::vector<float> covmat      (2*dimension*dimension*height*width, 0.0);
        std::vector<float> ref_filt   (                      height*width, 0.0);
        std::vector<float> phase_filt (                      height*width, 0.0);
        
        // simulate coherence value
        static std::default_random_engine rand_eng{};
        static std::uniform_real_distribution<float> dist_phase {0, 1};
        // should be rayleigh
        static std::gamma_distribution<float> dist_ampl(2.0, 2.0);

        for(int i = 0; i<nelems; i++) {
            ampl_master[i] = dist_ampl(rand_eng);
            ampl_slave [i] = dist_ampl(rand_eng);
            phase[i]      = dist_phase(rand_eng);
        }

        // opencl setup
        auto cl_devs = get_platform_devs(0);
        cl::Context context{cl_devs};
        cl::CommandQueue cmd_queue{context};

        // kernel setup
        const int block_size = 16;
        covmat_create    KUT_create    {block_size, context};
        covmat_decompose KUT_decompose {block_size, context};

        // allocate memory
        cl::Buffer device_ampl_master {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), ampl_master.data(), NULL};
        cl::Buffer device_ampl_slave  {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), ampl_slave.data(),  NULL};
        cl::Buffer device_phase      {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), phase.data(),      NULL};
        cl::Buffer device_covmat      {context, CL_MEM_READ_WRITE,                       2*dimension*dimension*height*width*sizeof(float), NULL,               NULL};
        cl::Buffer device_ref_filt   {context, CL_MEM_READ_WRITE,                                             height*width*sizeof(float), NULL,               NULL};
        cl::Buffer device_phase_filt {context, CL_MEM_READ_WRITE,                                             height*width*sizeof(float), NULL,               NULL};
        cl::Buffer device_coh_filt    {context, CL_MEM_READ_WRITE,                                             height*width*sizeof(float), NULL,               NULL};

        KUT_create.run(cmd_queue, 
                       device_ampl_master,
                       device_ampl_slave,
                       device_phase,
                       device_covmat,
                       height,
                       width);

        KUT_decompose.run(cmd_queue, 
                          device_covmat,
                          device_ref_filt,
                          device_phase_filt,
                          device_coh_filt,
                          height,
                          width);

        cmd_queue.enqueueReadBuffer(device_ref_filt,   CL_TRUE, 0, height*width*sizeof(float), ref_filt.data(),   NULL, NULL);
        cmd_queue.enqueueReadBuffer(device_phase_filt, CL_TRUE, 0, height*width*sizeof(float), phase_filt.data(), NULL, NULL);
        
        // workaround, since Approx does not work with vectors
        std::vector<float> ref;
        for(unsigned int i = 0; i < ref_filt.size(); i++) {
            ref.push_back(std::pow(0.5f*(ampl_master[i]+ampl_slave[i]), 2.0f));
        }

        ASSERT_THAT(ref, Pointwise(FloatNearPointwise(1e-4), ref_filt));
        ASSERT_THAT(phase, Pointwise(FloatNearPointwise(1e-4), phase_filt));
}
