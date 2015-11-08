#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <complex>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include "patches_unpack.h"

using namespace goldstein;

TEST_CASE( "patches_unpack", "[cl_kernels]" ) {

        // data setup
        const int height_packed = 72;
        const int width_packed  = 72;
        const int patch_size    = 32;
        const int overlap       = 4;

        const int n_patches_h = height_packed / (patch_size - 2*overlap);
        const int n_patches_w = width_packed  / (patch_size - 2*overlap);

        const int height_unpacked = n_patches_h * patch_size;
        const int width_unpacked  = n_patches_w * patch_size;

        std::vector<float> interf_real_packed   (height_packed*width_packed, 1.0);
        std::vector<float> interf_imag_packed   (height_packed*width_packed, 1.0);

        std::vector<float> interf_real_unpacked (height_unpacked*width_unpacked, -1.0);
        std::vector<float> interf_imag_unpacked (height_unpacked*width_unpacked, -1.0);


        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        patches_unpack kut_unpack {block_size, context};

        // allocate memory
        cl::Buffer device_interf_real_packed   {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height_packed*width_packed*sizeof(float), interf_real_packed.data(), NULL};
        cl::Buffer device_interf_imag_packed   {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height_packed*width_packed*sizeof(float), interf_imag_packed.data(), NULL};

        cl::Buffer device_interf_real_unpacked {context, CL_MEM_READ_ONLY, height_unpacked*width_unpacked*sizeof(float), NULL, NULL};
        cl::Buffer device_interf_imag_unpacked {context, CL_MEM_READ_ONLY, height_unpacked*width_unpacked*sizeof(float), NULL, NULL};

        kut_unpack.run(cmd_queue,
                       device_interf_real_packed, 
                       device_interf_imag_packed,
                       device_interf_real_unpacked, 
                       device_interf_imag_unpacked,
                       height_unpacked,
                       width_unpacked,
                       patch_size,
                       overlap);

        cmd_queue.enqueueReadBuffer(device_interf_real_unpacked, CL_TRUE, 0, height_unpacked*width_unpacked*sizeof(float), interf_real_unpacked.data(), NULL, NULL);
        cmd_queue.enqueueReadBuffer(device_interf_imag_unpacked, CL_TRUE, 0, height_unpacked*width_unpacked*sizeof(float), interf_imag_unpacked.data(), NULL, NULL);

        bool flag = true;
        for(int i = 0; i < height_packed*width_packed; i++) {
            flag = flag && ( 1.0f == Approx(interf_real_unpacked[i]).epsilon( 0.0001 ));
            flag = flag && ( 1.0f == Approx(interf_imag_unpacked[i]).epsilon( 0.0001 ));
        }
        REQUIRE( flag);
}
