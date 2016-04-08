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

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include "patches_pack.h"
#include "patches_unpack.h"

using namespace goldstein;

TEST_CASE( "patches_packing", "[cl_kernels]" ) {

        // data setup
        const int height_packed = 48;
        const int width_packed  = 72;
        const int patch_size    = 32;
        const int overlap       = 4;

        const int n_patches_h = height_packed / (patch_size - 2*overlap);
        const int n_patches_w = width_packed  / (patch_size - 2*overlap);

        const int height_unpacked = n_patches_h * patch_size;
        const int width_unpacked  = n_patches_w * patch_size;

        std::vector<float> interf_real          (height_packed*width_packed, -1.0);
        std::vector<float> interf_imag          (height_packed*width_packed, -1.0);

        std::vector<float> interf_real_out      (height_packed*width_packed, -1.0);
        std::vector<float> interf_imag_out      (height_packed*width_packed, -1.0);

        static std::default_random_engine rand_eng{};
        static std::uniform_int_distribution<int> dist_params(1, 9.0f);

        for(int i = 0; i < height_packed * width_packed; i++) {
            interf_real[i] = dist_params(rand_eng);
            interf_imag[i] = dist_params(rand_eng);
        }

        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        patches_unpack kut_unpack {block_size, context};
        patches_pack   kut_pack   {block_size, context};

        // allocate memory
        cl::Buffer device_interf_real_packed   {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height_packed*width_packed*sizeof(float), interf_real.data(), NULL};
        cl::Buffer device_interf_imag_packed   {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height_packed*width_packed*sizeof(float), interf_imag.data(), NULL};

        cl::Buffer device_interf_real_unpacked {context, CL_MEM_READ_ONLY, height_unpacked*width_unpacked*sizeof(float), NULL, NULL};
        cl::Buffer device_interf_imag_unpacked {context, CL_MEM_READ_ONLY, height_unpacked*width_unpacked*sizeof(float), NULL, NULL};

        cl::Buffer device_interf_real_out      {context, CL_MEM_READ_ONLY, height_packed*width_packed*sizeof(float), NULL, NULL};
        cl::Buffer device_interf_imag_out      {context, CL_MEM_READ_ONLY, height_packed*width_packed*sizeof(float), NULL, NULL};

        kut_unpack.run(cmd_queue,
                       device_interf_real_packed, 
                       device_interf_imag_packed,
                       device_interf_real_unpacked, 
                       device_interf_imag_unpacked,
                       height_unpacked,
                       width_unpacked,
                       patch_size,
                       overlap);

        kut_pack.run(cmd_queue,
                     device_interf_real_unpacked, 
                     device_interf_imag_unpacked,
                     device_interf_real_out, 
                     device_interf_imag_out,
                     height_unpacked,
                     width_unpacked,
                     patch_size,
                     overlap);

        cmd_queue.enqueueReadBuffer(device_interf_real_out, CL_TRUE, 0, height_packed*width_packed*sizeof(float), interf_real_out.data(), NULL, NULL);
        cmd_queue.enqueueReadBuffer(device_interf_imag_out, CL_TRUE, 0, height_packed*width_packed*sizeof(float), interf_imag_out.data(), NULL, NULL);

        bool flag = true;
        for(int y = 0; y < height_packed; y++) {
            for(int x = 0; x < width_packed; x++) {
                std::cout << std::setprecision(5) << (float) interf_real    [y*width_packed + x] << ":";
                std::cout << std::setprecision(5) << (float) interf_real_out[y*width_packed + x] << ", ";
            }
            std::cout << std::endl;
        }

        for(int i = 0; i < height_packed*width_packed; i++) {
            flag = flag && (interf_real_out[i] == Approx(interf_real[i]).epsilon( 0.0001 ));
            flag = flag && (interf_imag_out[i] == Approx(interf_imag[i]).epsilon( 0.0001 ));
        }

        REQUIRE( flag);
}
