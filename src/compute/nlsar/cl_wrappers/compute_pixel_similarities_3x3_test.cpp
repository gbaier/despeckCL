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

#include "compute_pixel_similarities_3x3.h"

using namespace nlsar;

TEST(compute_pixel_similarities_3x3, nonzero) {

        // data setup
        const int height_overlap = 5;
        const int width_overlap  = 5;
        const int dimension = 3;
        const int nlooks = 1;
        const int search_window_size = 3;
        const int wsh = (search_window_size-1)/2;
        const int width_symm  = width_overlap;
        const int height_symm = height_overlap - wsh;

        const int nelems              = height_overlap * width_overlap;
        const int covmat_nelems       = 2 * dimension * dimension * height_overlap * width_overlap;
        const int similarities_nelems = (search_window_size*wsh+wsh) * height_symm * width_symm;

        std::vector<float> covmat                 (covmat_nelems, 1.0);
        std::vector<float> similarities           (similarities_nelems, 0.0);

        // use matrix
        // [[ 5.+0.j,  0.-4.j, -4.+0.j],
        //  [ 0.+4.j,  5.+0.j,  0.-4.j],
        //  [-4.+0.j,  0.+4.j,  5.+0.j]]
        //  which has determinant 13


        for(int i = 0; i<nelems; i++) {
            covmat[i            ] =  5;
            covmat[i +    nelems] =  0;            // imaginary part must be zero for diaognal elements
            covmat[i +  2*nelems] =  0;
            covmat[i +  3*nelems] = -4;
            covmat[i +  4*nelems] = -4;
            covmat[i +  5*nelems] =  0;
            covmat[i +  6*nelems] =  0;
            covmat[i +  7*nelems] =  4;
            covmat[i +  8*nelems] =  5;
            covmat[i +  9*nelems] =  0;
            covmat[i + 10*nelems] =  0;
            covmat[i + 11*nelems] = -4;
            covmat[i + 12*nelems] = -4;
            covmat[i + 13*nelems] =  0;
            covmat[i + 14*nelems] =  0; 
            covmat[i + 15*nelems] =  4;
            covmat[i + 16*nelems] =  5;
            covmat[i + 17*nelems] =  0;
        }
         
        // opencl setup
        auto cl_devs = get_platform_devs(0);
        cl::Context context{cl_devs};

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        compute_pixel_similarities_3x3 KUT{block_size, context};

        // allocate memory
        cl::Buffer device_covmat       {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,       covmat_nelems * sizeof(float), covmat.data(), NULL};
        cl::Buffer device_similarities {context, CL_MEM_READ_WRITE,                       similarities_nelems * sizeof(float), NULL,          NULL};

        KUT.run(cmd_queue, 
                device_covmat,
                device_similarities,
                height_overlap,
                width_overlap,
                dimension,
                nlooks,
                search_window_size);

        cmd_queue.enqueueReadBuffer(device_similarities, CL_TRUE, 0, similarities_nelems * sizeof(float), similarities.data(), NULL, NULL);

        bool flag = true;
        // check 2nd quadrant
        for(int hh=0; hh<wsh+1; hh++) {
            for(int ww=0; ww<wsh; ww++) {
                for(int h=0; h<height_symm; h++) {
                   for(int w=wsh; w<width_symm; w++) {
                       flag = flag && similarities[(hh*search_window_size + ww)*height_symm*width_symm + h*width_symm + w] != 0;
                   }
                }
            }
        }
        
        // check 1st quadrant
        for(int hh=0; hh<wsh; hh++) {
            for(int ww=wsh; ww<search_window_size; ww++) {
                for(int h=0; h<height_symm; h++) {
                   for(int w=0; w<width_symm-wsh; w++) {
                       flag = flag && similarities[(hh*search_window_size + ww)*height_symm*width_symm + h*width_symm + w] != 0;
                   }
                }
            }
        }

        ASSERT_TRUE( flag );
}
