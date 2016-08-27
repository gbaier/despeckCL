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

#include "patches_pack.h"

using namespace goldstein;
using testing::Each;
using testing::FloatEq;
using testing::Pointwise;

TEST(patches_pack, completeness) {

        // data setup
        const int height_packed = 48;
        const int width_packed  = 72;
        const int patch_size    = 32;
        const int overlap       = 4;

        const int n_patches_h = height_packed / (patch_size - 2*overlap);
        const int n_patches_w = width_packed  / (patch_size - 2*overlap);

        const int height_unpacked = n_patches_h * patch_size;
        const int width_unpacked  = n_patches_w * patch_size;

        std::vector<float> interf_real_unpacked (height_unpacked*width_unpacked, 1.0);
        std::vector<float> interf_imag_unpacked (height_unpacked*width_unpacked, 1.0);

        std::vector<float> interf_real_packed   (height_packed*width_packed, -1.0);
        std::vector<float> interf_imag_packed   (height_packed*width_packed, -1.0);

        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        patches_pack   kut_pack   {block_size, context};

        // allocate memory
        cl::Buffer device_interf_real_unpacked   {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height_unpacked*width_unpacked*sizeof(float), interf_real_unpacked.data(), NULL};
        cl::Buffer device_interf_imag_unpacked   {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height_unpacked*width_unpacked*sizeof(float), interf_imag_unpacked.data(), NULL};

        cl::Buffer device_interf_real_packed {context, CL_MEM_READ_ONLY, height_packed*width_packed*sizeof(float), NULL, NULL};
        cl::Buffer device_interf_imag_packed {context, CL_MEM_READ_ONLY, height_packed*width_packed*sizeof(float), NULL, NULL};

        kut_pack.run(cmd_queue,
                     device_interf_real_unpacked, 
                     device_interf_imag_unpacked,
                     device_interf_real_packed, 
                     device_interf_imag_packed,
                     height_unpacked,
                     width_unpacked,
                     patch_size,
                     overlap);

        cmd_queue.enqueueReadBuffer(device_interf_real_packed, CL_TRUE, 0, height_packed*width_packed*sizeof(float), interf_real_packed.data(), NULL, NULL);
        cmd_queue.enqueueReadBuffer(device_interf_imag_packed, CL_TRUE, 0, height_packed*width_packed*sizeof(float), interf_imag_packed.data(), NULL, NULL);

        ASSERT_THAT(interf_real_packed, Each(FloatEq(1.0f)));
        ASSERT_THAT(interf_imag_packed, Each(FloatEq(1.0f)));
}

TEST(patches_pack, random) {

        // data setup
        const int height_packed = 72;
        const int width_packed  = 72;
        const int patch_size    = 32;
        const int overlap       = 4;

        const int n_patches_h = height_packed / (patch_size - 2*overlap);
        const int n_patches_w = width_packed  / (patch_size - 2*overlap);

        const int height_unpacked = n_patches_h * patch_size;
        const int width_unpacked  = n_patches_w * patch_size;

        std::vector<float> interf_real_packed_desired (height_packed*width_packed, 1.0);
        std::vector<float> interf_imag_packed_desired (height_packed*width_packed, 1.0);

        std::vector<float> interf_real_packed         (height_packed*width_packed, -1.0);
        std::vector<float> interf_imag_packed         (height_packed*width_packed, -1.0);

        std::vector<float> interf_real_unpacked       (height_unpacked*width_unpacked, -1.0);
        std::vector<float> interf_imag_unpacked       (height_unpacked*width_unpacked, -1.0);

        static std::default_random_engine rand_eng{};
        static std::uniform_int_distribution<int> dist_params(1, 2.0f);

        for(int y = 0; y < height_packed; y++) {
            for(int x = 0; x < width_packed; x++) {
                interf_real_packed_desired[y*width_packed + x] = dist_params(rand_eng);
                interf_imag_packed_desired[y*width_packed + x] = dist_params(rand_eng);
            }
        }
        for(int y = 0; y < height_unpacked; y++) {
            for(int x = 0; x < width_unpacked; x++) {
                const int patch_idx = x/patch_size;
                const int patch_idy = y/patch_size;

                const int rel_tx = x % patch_size;
                const int rel_ty = y % patch_size;

                const int xx = std::min(width_packed-1, std::max(0, patch_idx*(patch_size-2*overlap) + (rel_tx - overlap)));
                const int yy = std::min(width_packed-1, std::max(0, patch_idy*(patch_size-2*overlap) + (rel_ty - overlap)));

                interf_real_unpacked[y*width_unpacked + x] = interf_real_packed_desired[yy*width_packed + xx];
                interf_imag_unpacked[y*width_unpacked + x] = interf_imag_packed_desired[yy*width_packed + xx];
            }
        }

        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        patches_pack   kut_pack   {block_size, context};

        // allocate memory
        cl::Buffer device_interf_real_unpacked   {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height_unpacked*width_unpacked*sizeof(float), interf_real_unpacked.data(), NULL};
        cl::Buffer device_interf_imag_unpacked   {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height_unpacked*width_unpacked*sizeof(float), interf_imag_unpacked.data(), NULL};

        cl::Buffer device_interf_real_packed {context, CL_MEM_READ_ONLY, height_packed*width_packed*sizeof(float), NULL, NULL};
        cl::Buffer device_interf_imag_packed {context, CL_MEM_READ_ONLY, height_packed*width_packed*sizeof(float), NULL, NULL};

        kut_pack.run(cmd_queue,
                     device_interf_real_unpacked, 
                     device_interf_imag_unpacked,
                     device_interf_real_packed, 
                     device_interf_imag_packed,
                     height_unpacked,
                     width_unpacked,
                     patch_size,
                     overlap);

        cmd_queue.enqueueReadBuffer(device_interf_real_packed, CL_TRUE, 0, height_packed*width_packed*sizeof(float), interf_real_packed.data(), NULL, NULL);
        cmd_queue.enqueueReadBuffer(device_interf_imag_packed, CL_TRUE, 0, height_packed*width_packed*sizeof(float), interf_imag_packed.data(), NULL, NULL);

        ASSERT_THAT(interf_real_packed, Pointwise(FloatNearPointwise(1e-4), interf_real_packed_desired));
        ASSERT_THAT(interf_imag_packed, Pointwise(FloatNearPointwise(1e-4), interf_imag_packed_desired));
}
