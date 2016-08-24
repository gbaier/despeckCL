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

#include "compute_patch_similarities.h"

using namespace nlsar;
using testing::Pointwise;

TEST(compute_patch_similarities, simple_convolution) {

        // data setup
        const int height_sim = 20;
        const int width_sim  = 30;
        const int search_window_size = 5;
        const int wsh = (search_window_size-1)/2;
        const int patch_size         = 3;
        const int patch_size_max     = patch_size;

        const int height_ori = height_sim - patch_size + 1;
        const int width_ori  = width_sim  - patch_size + 1;

        std::vector<float> pixel_similarities         ((search_window_size*wsh+wsh) * height_sim * width_sim, 1.0);
        std::vector<float> patch_similarities         ((search_window_size*wsh+wsh) * height_ori * width_ori, 0.0);
        std::vector<float> desired_patch_similarities ((search_window_size*wsh+wsh) * height_ori * width_ori, 0.0);

        for(int d = 0; d<search_window_size*wsh+wsh; d++) {
            for(int i = 0; i < height_sim * width_sim; i++) {
                pixel_similarities[d*height_sim*width_sim + i] = d;
            }
            for(int i = 0; i < height_ori * width_ori; i++) {
                desired_patch_similarities[d*height_ori*width_ori + i] = d*patch_size*patch_size;
            }
        }

        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        compute_patch_similarities KUT{context, block_size, 4, 4, 4};

        // allocate memory
        cl::Buffer device_pixel_similarities{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pixel_similarities.size() * sizeof(float), pixel_similarities.data(), NULL};
        cl::Buffer device_patch_similarities{context, CL_MEM_READ_WRITE,                       patch_similarities.size() * sizeof(float), NULL, NULL};

        KUT.run(cmd_queue, 
                device_pixel_similarities,
                device_patch_similarities,
                height_sim,
                width_sim,
                search_window_size,
                patch_size,
                patch_size_max);

        cmd_queue.enqueueReadBuffer(device_patch_similarities, CL_TRUE, 0, patch_similarities.size() * sizeof(float), patch_similarities.data(), NULL, NULL);
        ASSERT_THAT(patch_similarities, Pointwise(FloatNearPointwise(1e-4), desired_patch_similarities));
}
