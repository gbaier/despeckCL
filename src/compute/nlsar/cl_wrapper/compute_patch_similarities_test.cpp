#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include "compute_patch_similarities.h"

TEST_CASE( "compute_patch_similarities", "[cl_kernels]" ) {

        // data setup
        const int height_sim = 10;
        const int width_sim  = 10;
        const int search_window_size = 5;
        const int patch_size         = 3;

        const int height_ori = height_sim - patch_size + 1;
        const int width_ori  = width_sim  - patch_size + 1;

        std::vector<float> pixel_similarities         (search_window_size * search_window_size * height_sim * width_sim, 1.0);
        std::vector<float> patch_similarities         (search_window_size * search_window_size * height_ori * width_ori, 0.0);
        std::vector<float> desired_patch_similarities (search_window_size * search_window_size * height_ori * width_ori, 0.0);

        // simulate coherence value
        static std::default_random_engine rand_eng{};
        static std::uniform_real_distribution<float> dist {0, 1};

        for(int d = 0; d<search_window_size*search_window_size; d++) {
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
        compute_patch_similarities KUT{block_size, context, patch_size};

        // allocate memory
        cl::Buffer device_pixel_similarities{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pixel_similarities.size() * sizeof(float), pixel_similarities.data(), NULL};
        cl::Buffer device_patch_similarities{context, CL_MEM_READ_WRITE,                       patch_similarities.size() * sizeof(float), NULL, NULL};

        KUT.run(cmd_queue, 
                device_pixel_similarities,
                device_patch_similarities,
                height_sim,
                width_sim,
                search_window_size,
                patch_size);

        cmd_queue.enqueueReadBuffer(device_patch_similarities, CL_TRUE, 0, patch_similarities.size() * sizeof(float), patch_similarities.data(), NULL, NULL);

        REQUIRE( ( patch_similarities == desired_patch_similarities ) );
}
