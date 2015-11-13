#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <clFFT.h>

#include <complex>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

//using namespace goldstein;

#include "clcfg.h"
#include "goldstein_patch_ft.h"

TEST_CASE( "goldstein_patch_ft", "[cl_kernels]" ) {

        // data setup
        const int height     = 64;
        const int width      = 64;
        const int patch_size = 32;

        std::vector<float> real_in  (height*width, 1.0);
        std::vector<float> imag_in  (height*width, 0.0);

        std::vector<float> real_out (height*width, -1.0);
        std::vector<float> imag_out (height*width, -1.0);

        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // io buffers
        cl::Buffer dev_real_in {context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), real_in.data(), NULL};
        cl::Buffer dev_imag_in {context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), imag_in.data(), NULL};

        cl::Buffer dev_real_out {context, CL_MEM_READ_WRITE, height * width * sizeof(float), NULL, NULL};
        cl::Buffer dev_imag_out {context, CL_MEM_READ_WRITE, height * width * sizeof(float), NULL, NULL};

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
        clfftSetResultLocation(plan_handle, CLFFT_OUTOFPLACE);
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
                           dev_real_in,
                           dev_imag_in,
                           dev_real_out,
                           dev_imag_out,
                           height,
                           width,
                           patch_size,
                           CLFFT_FORWARD);

        /* Release the plan. */
        clfftDestroyPlan( &plan_handle );

        /* Release clFFT library. */
        clfftTeardown( );

        cmd_queue.enqueueReadBuffer(dev_real_out, CL_TRUE, 0, height*width*sizeof(float), real_out.data(), NULL, NULL);
        cmd_queue.enqueueReadBuffer(dev_imag_out, CL_TRUE, 0, height*width*sizeof(float), imag_out.data(), NULL, NULL);

        /*
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                std::cout << std::setw(2) << imag_out[y*width + x] << ",";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        */

        bool flag = true;
        for(int h = 0; h < height; h++) {
            for(int w = 0; w < width; w++) {
                const int i = h*width+w;
                if( w % patch_size || h % patch_size) {
                    flag = flag && (0.0f == Approx(real_out[i]).epsilon( 0.0001 ));
                } else {
                    flag = flag && (patch_size*patch_size == Approx(real_out[i]).epsilon( 0.0001 ));
                }
                flag = flag && (0.0f == Approx(imag_out[i]).epsilon( 0.0001 ));
            }
        }
        REQUIRE( flag);
}

TEST_CASE( "goldstein_patch_ft_rand", "[cl_kernels]" ) {

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
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // io buffers
        cl::Buffer dev_real_in {context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), real_in.data(), NULL};
        cl::Buffer dev_imag_in {context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), imag_in.data(), NULL};

        cl::Buffer dev_real_out {context, CL_MEM_READ_WRITE, height * width * sizeof(float), NULL, NULL};
        cl::Buffer dev_imag_out {context, CL_MEM_READ_WRITE, height * width * sizeof(float), NULL, NULL};

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
        clfftSetResultLocation(plan_handle, CLFFT_OUTOFPLACE);
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
                           dev_real_in,
                           dev_imag_in,
                           dev_real_out,
                           dev_imag_out,
                           height,
                           width,
                           patch_size,
                           CLFFT_FORWARD);

        goldstein_patch_ft(cmd_queue,
                           plan_handle,
                           dev_real_out,
                           dev_imag_out,
                           dev_real_in,
                           dev_imag_in,
                           height,
                           width,
                           patch_size,
                           CLFFT_BACKWARD);

        /* Release the plan. */
        clfftDestroyPlan( &plan_handle );

        /* Release clFFT library. */
        clfftTeardown( );

        cmd_queue.enqueueReadBuffer(dev_real_in, CL_TRUE, 0, height*width*sizeof(float), real_out.data(), NULL, NULL);
        cmd_queue.enqueueReadBuffer(dev_imag_in, CL_TRUE, 0, height*width*sizeof(float), imag_out.data(), NULL, NULL);

        /*
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                std::cout << std::setw(1) << real_out[y*width + x] << ":" << real_in[y*width + x] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        */

        bool flag = true;
        for(int i = 0; i < height*width; i++) {
            flag = flag && (real_in[i] == Approx(real_out[i]).epsilon( 0.0001 ));
            flag = flag && (imag_in[i] == Approx(imag_out[i]).epsilon( 0.0001 ));
        }
        REQUIRE( flag);
}
