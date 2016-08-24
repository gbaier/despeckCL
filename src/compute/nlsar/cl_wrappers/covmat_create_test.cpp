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
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "unit_test_helper.h"

#include "covmat_create.h"

using namespace nlsar;
using testing::Each;
using testing::FloatEq;
using testing::Gt;
using testing::Pointwise;

TEST(covmat_create, determinant) {

        // data setup
        const int height = 10;
        const int width = 10;
        const int dimension = 2;

        std::vector<float> ampl_master (                      height*width, 1.0);
        std::vector<float> ampl_slave  (                      height*width, 2.0);
        std::vector<float> phase      (                      height*width, 0.5);
        std::vector<float> covmat      (2*dimension*dimension*height*width, 1.0);

        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        covmat_create KUT{block_size, context};

        // allocate memory
        cl::Buffer device_ampl_master {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), ampl_master.data(), NULL};
        cl::Buffer device_ampl_slave  {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), ampl_slave.data(),  NULL};
        cl::Buffer device_phase      {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), phase.data(),      NULL};
        cl::Buffer device_covmat      {context, CL_MEM_READ_WRITE,                       2*dimension*dimension*height*width*sizeof(float), NULL,               NULL};

        KUT.run(cmd_queue, 
                device_ampl_master,
                device_ampl_slave,
                device_phase,
                device_covmat,
                height,
                width);

        cmd_queue.enqueueReadBuffer(device_covmat, CL_TRUE, 0, 2*height*width*dimension*dimension*sizeof(float), covmat.data(), NULL, NULL);

        const size_t offset = height*width;

        std::vector<float> dets;

        // workaround, since Approx does not work with vectors
        for(size_t i = 0; i < offset; i++) {
            //std::complex<float> el00 {covmat[i],            covmat[i +   offset]};
            std::complex<float> el00 {covmat[i],            covmat[i +   offset]};
            std::complex<float> el01 {covmat[i + 2*offset], covmat[i + 3*offset]};
            std::complex<float> el10 {covmat[i + 4*offset], covmat[i + 5*offset]};
            std::complex<float> el11 {covmat[i + 6*offset], covmat[i + 7*offset]};
            dets.push_back(std::abs(el00*el11 - el01*el10));
        }
        ASSERT_THAT(dets, Each(FloatEq(0.0f)));
}

TEST(covmat_create, sanity_check) {

        // data setup
        const int height = 40;
        const int width = 50;
        const int dimension = 2;

        std::vector<float> ampl_master (                      height*width, -1.0);
        std::vector<float> ampl_slave  (                      height*width, -1.0);
        std::vector<float> phase       (                      height*width, -1.0);
        std::vector<float> covmat      (2*dimension*dimension*height*width, -1.0);

        static std::default_random_engine rand_eng{};
        static std::gamma_distribution<float>        dist_ampl(1.0, 5.0);
        static std::uniform_real_distribution<float> dist_phase(1.0, 5.0);

        for(int i = 0; i < height*width; i++) {
            ampl_master[i] = dist_ampl(rand_eng);
            ampl_slave [i] = dist_ampl(rand_eng);
            phase     [i] = dist_phase(rand_eng);
        }

        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        covmat_create KUT{block_size, context};

        // allocate memory
        cl::Buffer device_ampl_master {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), ampl_master.data(), NULL};
        cl::Buffer device_ampl_slave  {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), ampl_slave.data(),  NULL};
        cl::Buffer device_phase      {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), phase.data(),      NULL};
        cl::Buffer device_covmat      {context, CL_MEM_READ_WRITE,                       2*dimension*dimension*height*width*sizeof(float), NULL,               NULL};

        KUT.run(cmd_queue, 
                device_ampl_master,
                device_ampl_slave,
                device_phase,
                device_covmat,
                height,
                width);

        cmd_queue.enqueueReadBuffer(device_covmat, CL_TRUE, 0, 2*height*width*dimension*dimension*sizeof(float), covmat.data(), NULL, NULL);

        std::vector<std::complex<float>> el00; 
        std::vector<std::complex<float>> el01; 
        std::vector<std::complex<float>> el10; 
        std::vector<std::complex<float>> el11; 

        const size_t offset = height*width;

        for(size_t i = 0; i < offset; i++) {
            el00.push_back({covmat[i],            covmat[i +   offset]});
            el01.push_back({covmat[i + 2*offset], covmat[i + 3*offset]});
            el10.push_back({covmat[i + 4*offset], covmat[i + 5*offset]});
            el11.push_back({covmat[i + 6*offset], covmat[i + 7*offset]});
        }

        // test that real part of reflectivity is greater than 0
        std::vector<float> el00_real(el00.size());
        std::vector<float> el11_real(el00.size());
        std::transform(el00.begin(), el00.end(), el00_real.begin(), [] (std::complex<float> z) { return z.real(); });
        std::transform(el11.begin(), el11.end(), el11_real.begin(), [] (std::complex<float> z) { return z.real(); });

        ASSERT_THAT(el00_real, Each(Gt(0.0f)));
        ASSERT_THAT(el11_real, Each(Gt(0.0f)));

        // test that imaginar part of reflectivity is zero
        std::vector<float> el00_imag(el00.size());
        std::vector<float> el11_imag(el00.size());
        std::transform(el00.begin(), el00.end(), el00_imag.begin(), [] (std::complex<float> z) { return z.imag(); });
        std::transform(el11.begin(), el11.end(), el11_imag.begin(), [] (std::complex<float> z) { return z.imag(); });

        ASSERT_THAT(el00_imag, Each(FloatEq(0.0f)));
        ASSERT_THAT(el11_imag, Each(FloatEq(0.0f)));

        // test that the off diagonal elements are the complex conjugate
        ASSERT_THAT(el10, Pointwise(ComplexConjugate(), el01));
}
