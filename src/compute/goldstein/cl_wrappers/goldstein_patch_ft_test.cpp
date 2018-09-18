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

#include <clFFT.h>

#include <complex>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "unit_test_helper.h"

//using namespace goldstein;

#include "clcfg.h"
#include "goldstein_patch_ft.h"

using testing::Pointwise;

TEST(goldstein_patch_ft, const2dirac) {

        // data setup
        const int height     = 64;
        const int width      = 64;
        const int patch_size = 32;

        std::vector<float> real_in  (height*width, 1.0);
        std::vector<float> imag_in  (height*width, 0.0);

        std::vector<float> real_out (height*width, -1.0);
        std::vector<float> imag_out (height*width, -1.0);

        // opencl setup
        auto cl_devs = get_platform_devs(0);
        cl::Context context{cl_devs};
        cl::CommandQueue cmd_queue{context};

        // io buffers
        cl::Buffer dev_real {context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), real_in.data(), NULL};
        cl::Buffer dev_imag {context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), imag_in.data(), NULL};

        /****************************************************************************
         *
         * clFFT setup
         *
         ******************************************************************/

        clfftPlanHandle plan_handle;
        clfftDim dim = CLFFT_2D;
        size_t cl_lengths[2] = {patch_size, patch_size};
        size_t in_strides [2] = {1, width};
        size_t out_strides[2] = {1, width};


        /* Setup clFFT. */
        clfftSetupData fft_setup;
        clfftInitSetupData(&fft_setup);
        clfftSetup(&fft_setup);

        /* Create a default plan for a complex FFT. */
        clfftCreateDefaultPlan(&plan_handle, context(), dim, cl_lengths);

        /* Set plan parameters. */
        clfftSetPlanPrecision(plan_handle, CLFFT_SINGLE);
        clfftSetLayout(plan_handle, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR); // separate arrays for real and complex data
        clfftSetResultLocation(plan_handle, CLFFT_INPLACE);
        clfftSetPlanInStride  (plan_handle, dim, in_strides);
        clfftSetPlanOutStride (plan_handle, dim, out_strides);
        clfftSetPlanBatchSize (plan_handle, width/patch_size);
        clfftSetPlanDistance  (plan_handle, patch_size, patch_size);

        /* Bake the plan. */
        clfftBakePlan(plan_handle, 1, &cmd_queue(), NULL, NULL);
        
        /****************************************************************************
         *
         * Processing
         *
         ******************************************************************/

        goldstein_patch_ft(cmd_queue,
                           plan_handle,
                           dev_real,
                           dev_imag,
                           height,
                           width,
                           patch_size,
                           CLFFT_FORWARD);

        /* Release the plan. */
        clfftDestroyPlan( &plan_handle );

        /* Release clFFT library. */
        clfftTeardown( );

        cmd_queue.enqueueReadBuffer(dev_real, CL_TRUE, 0, height*width*sizeof(float), real_out.data(), NULL, NULL);
        cmd_queue.enqueueReadBuffer(dev_imag, CL_TRUE, 0, height*width*sizeof(float), imag_out.data(), NULL, NULL);

        std::vector<float> desired_real_out (real_out.size(), 0.0f);
        std::vector<float> desired_imag_out (imag_out.size(), 0.0f);

        for(int h = 0; h < height; h++) {
            for(int w = 0; w < width; w++) {
                const int i = h*width+w;
                if ( w % patch_size == 0 && h % patch_size == 0) {
                    desired_real_out[i] = patch_size*patch_size;
                }
            }
        }
        ASSERT_THAT(real_out, Pointwise(FloatNearPointwise(1e-4), desired_real_out));
        ASSERT_THAT(imag_out, Pointwise(FloatNearPointwise(1e-4), desired_imag_out));
}

TEST(goldstein_patch_ft_rand,  ) {

        // data setup
        const int height     = 64;
        const int width      = 64;
        const int patch_size = 32;

        std::vector<float> real_in  (height*width, 1.0);
        std::vector<float> imag_in  (height*width, 0.0);

        std::vector<float> real_out (height*width, -1.0);
        std::vector<float> imag_out (height*width, -1.0);

        static std::default_random_engine rand_eng{};
        static std::uniform_int_distribution<int> dist_params(1, 9.0f);

        for(int i = 0; i < height * width; i++) {
            real_in[i] = dist_params(rand_eng);
            imag_in[i] = dist_params(rand_eng);
        }

        // opencl setup
        auto cl_devs = get_platform_devs(0);
        cl::Context context{cl_devs};
        cl::CommandQueue cmd_queue{context};

        // io buffers
        cl::Buffer dev_real {context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), real_in.data(), NULL};
        cl::Buffer dev_imag {context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), imag_in.data(), NULL};

        /****************************************************************************
         *
         * clFFT setup
         *
         ******************************************************************/

        clfftPlanHandle plan_handle;
        clfftDim dim = CLFFT_2D;
        size_t cl_lengths[2] = {patch_size, patch_size};
        size_t in_strides [2] = {1, width};
        size_t out_strides[2] = {1, width};


        /* Setup clFFT. */
        clfftSetupData fft_setup;
        clfftInitSetupData(&fft_setup);
        clfftSetup(&fft_setup);

        /* Create a default plan for a complex FFT. */
        clfftCreateDefaultPlan(&plan_handle, context(), dim, cl_lengths);

        /* Set plan parameters. */
        clfftSetPlanPrecision(plan_handle, CLFFT_SINGLE);
        clfftSetLayout(plan_handle, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR); // separate arrays for real and complex data
        clfftSetResultLocation(plan_handle, CLFFT_INPLACE);
        clfftSetPlanInStride  (plan_handle, dim, in_strides);
        clfftSetPlanOutStride (plan_handle, dim, out_strides);
        clfftSetPlanBatchSize (plan_handle, width/patch_size);
        clfftSetPlanDistance  (plan_handle, patch_size, patch_size);

        /* Bake the plan. */
        clfftBakePlan(plan_handle, 1, &cmd_queue(), NULL, NULL);
        
        /****************************************************************************
         *
         * Processing
         *
         ******************************************************************/

        goldstein_patch_ft(cmd_queue,
                           plan_handle,
                           dev_real,
                           dev_imag,
                           height,
                           width,
                           patch_size,
                           CLFFT_FORWARD);

        goldstein_patch_ft(cmd_queue,
                           plan_handle,
                           dev_real,
                           dev_imag,
                           height,
                           width,
                           patch_size,
                           CLFFT_BACKWARD);

        /* Release the plan. */
        clfftDestroyPlan( &plan_handle );

        /* Release clFFT library. */
        clfftTeardown( );

        cmd_queue.enqueueReadBuffer(dev_real, CL_TRUE, 0, height*width*sizeof(float), real_out.data(), NULL, NULL);
        cmd_queue.enqueueReadBuffer(dev_imag, CL_TRUE, 0, height*width*sizeof(float), imag_out.data(), NULL, NULL);

        ASSERT_THAT(real_in, Pointwise(FloatNearPointwise(1e-4), real_out));
        ASSERT_THAT(imag_in, Pointwise(FloatNearPointwise(1e-4), imag_out));
}
