#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <complex>
#include <random>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include "copy_symm_weights.h"

using namespace nlsar;

bool check_arr(std::vector<float> weights_symm,
               std::vector<float> weights_full,
               const int height_ori,
               const int width_ori,
               const int search_window_size)
{
    const int wsh = (search_window_size - 1)/2;
    const int width_symm  = width_ori  + 2*wsh;
    const int height_symm = height_ori +   wsh;

    bool flag = true;

    //compare copied values
    for(int i=0; i < search_window_size*wsh+wsh; i++) {
        for(int h=0; h<height_ori; h++) {
            for(int w=0; w<width_ori; w++) {
                flag = flag && (weights_full[i*height_ori*width_ori + h*width_ori + w] == \
                                weights_symm[i*height_symm*width_symm + h*width_symm + w + wsh]);
            }
        }
    }

    std::cout << std::endl;
    std::cout << std::endl;
    //compare symmetric copied values
    for(int hh=wsh; hh < search_window_size; hh++) {
        int ww_start = 0;
        if (hh == wsh) {
            ww_start = wsh+1;
        }
        for(int ww=ww_start; ww < search_window_size; ww++) {
            const int hhs = search_window_size - hh - 1;
            const int wws = search_window_size - ww - 1;
            for(int h=0; h<height_ori; h++) {
                for(int w=0; w<width_ori; w++) {
                    const int hs = h + (hh - wsh);
                    const int ws = wsh + w + (ww - wsh);
                    flag = flag && (weights_full[ (hh*search_window_size+ww) *height_ori*width_ori + h*width_ori + w] == \
                                    weights_symm[(hhs*search_window_size+wws)*height_symm*width_symm + hs*width_symm + ws]);
                }
            }
        }
    }
    return flag;
}

TEST_CASE( "copy_symm_weights", "[cl_kernels]" ) {

        // data setup
        const int height_ori = 70;
        const int width_ori  = 30;

        const int search_window_size = 7;
        const int wsh = (search_window_size - 1)/2;

        const int width_symm  = width_ori  + 2*wsh;
        const int height_symm = height_ori +   wsh;

        std::vector<float> weights_symm (height_symm*width_symm*(wsh+search_window_size*wsh), 1.0);
        std::vector<float> weights_full (height_ori*width_ori*search_window_size*search_window_size, 0.0);

        // simulate coherence value
        static std::default_random_engine rand_eng{};
        static std::uniform_int_distribution<> dist {0, 9};

        for(unsigned int i = 0; i<weights_symm.size(); i++) {
            weights_symm[i] = dist(rand_eng);
        }
         
        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        copy_symm_weights KUT{block_size, context};

        // allocate memory
        cl::Buffer device_weights_symm {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_symm.size() * sizeof(float), weights_symm.data(), NULL};
        cl::Buffer device_weights_full {context, CL_MEM_READ_WRITE,                       weights_full.size() * sizeof(float), NULL,                NULL};

        KUT.run(cmd_queue, 
                device_weights_symm,
                device_weights_full,
                height_ori,
                width_ori,
                search_window_size);

        cmd_queue.enqueueReadBuffer(device_weights_full, CL_TRUE, 0, weights_full.size() * sizeof(float), weights_full.data(), NULL, NULL);

        /*
        for(int s=0; s<search_window_size*search_window_size; s++) {
            for(int h=0; h<height_ori; h++) {
                for(int w=0; w<width_ori; w++) {
                    std::cout << weights_full[s*height_ori*width_ori + h*width_ori + w] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        std::cout << "xxxxxxxx" << std::endl;

        for(int s=0; s<wsh+search_window_size*wsh; s++) {
            for(int h=0; h<height_symm; h++) {
                for(int w=0; w<width_symm; w++) {
                    std::cout << weights_symm[s*height_symm*width_symm + h*width_symm + w] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << "xxxxxxxx" << std::endl;
        */
        bool flag = check_arr(weights_symm,
                              weights_full,
                              height_ori,
                              width_ori,
                              search_window_size);
        REQUIRE( ( flag ) );
}
