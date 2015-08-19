#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <complex>
#include <random>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include "compute_pixel_similarities_2x2.h"

TEST_CASE( "compute_pixel_similarities_2x2", "[cl_kernels]" ) {

        // data setup
        const int height_overlap = 10;
        const int width_overlap  = 10;
        const int dimension = 2;
        const int search_window_size = 5;
        const int height_sim = height_overlap - search_window_size + 1;
        const int width_sim  = width_overlap  - search_window_size + 1;

        const int nelems              = height_overlap * width_overlap;
        const int covmat_nelems       = 2 * dimension * dimension * height_overlap * width_overlap;
        const int similarities_nelems = search_window_size * search_window_size * height_sim * width_sim;

        std::vector<float> covmat                 (covmat_nelems, 1.0);
        std::vector<float> similarities           (similarities_nelems, 0.0);
        std::vector<float> undesired_similarities (similarities_nelems, 0.0);

        // simulate coherence value
        static std::default_random_engine rand_eng{};
        static std::uniform_real_distribution<float> dist {0, 1};

        for(int i = 0; i<nelems; i++) {
            const float gamma = dist(rand_eng);
            covmat[i           ] *=  std::sqrt(2); // diagonal elements must have the same power as off diagonal elements
            covmat[i +   nelems]  =  0;            // and their imaginary part must be zero
            covmat[i + 2*nelems] *=  gamma;
            covmat[i + 3*nelems] *=  gamma;
            covmat[i + 4*nelems] *=  gamma;
            covmat[i + 5*nelems] *= -gamma;
            covmat[i + 6*nelems] *=  std::sqrt(2);
            covmat[i + 7*nelems]  =  0;
        }
         
        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        compute_pixel_similarities_2x2 KUT{block_size, context};

        // allocate memory
        cl::Buffer device_covmat       {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,       covmat_nelems * sizeof(float), covmat.data(), NULL};
        cl::Buffer device_similarities {context, CL_MEM_READ_WRITE,                       similarities_nelems * sizeof(float), NULL,          NULL};

        KUT.run(cmd_queue, 
                device_covmat,
                device_similarities,
                height_overlap,
                width_overlap,
                search_window_size);

        cmd_queue.enqueueReadBuffer(device_similarities, CL_TRUE, 0, similarities_nelems * sizeof(float), similarities.data(), NULL, NULL);

        REQUIRE( ( similarities != undesired_similarities ) );
}
