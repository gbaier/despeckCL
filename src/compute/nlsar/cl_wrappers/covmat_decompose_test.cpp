#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <complex>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include "covmat_decompose.h"
#include "covmat_create.h"

using namespace nlsar;

TEST_CASE( "covmat_decompose", "[cl_kernels]" ) {

        // data setup
        const int height = 10;
        const int width  = 10;
        const int dimension = 2;

        const int nelems = height*width;

        std::vector<float> ampl_master (                      height*width, 1.0);
        std::vector<float> ampl_slave  (                      height*width, 1.0);
        std::vector<float> dphase      (                      height*width, 1.0);
        std::vector<float> covmat      (2*dimension*dimension*height*width, 0.0);
        std::vector<float> ampl_filt   (                      height*width, 0.0);
        std::vector<float> dphase_filt (                      height*width, 0.0);
        
        // simulate coherence value
        static std::default_random_engine rand_eng{};
        static std::uniform_real_distribution<float> dist_dphase {0, 1};
        // should be rayleigh
        static std::gamma_distribution<float> dist_ampl(2.0, 2.0);

        for(int i = 0; i<nelems; i++) {
            ampl_master[i] = dist_ampl(rand_eng);
            ampl_slave [i] = dist_ampl(rand_eng);
            dphase[i]      = dist_dphase(rand_eng);
        }

        // opencl setup
        cl::Context context = opencl_setup();

        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        cl::CommandQueue cmd_queue{context, devices[0]};

        // kernel setup
        const int block_size = 16;
        covmat_create    KUT_create    {block_size, context};
        covmat_decompose KUT_decompose {block_size, context};

        // allocate memory
        cl::Buffer device_ampl_master {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), ampl_master.data(), NULL};
        cl::Buffer device_ampl_slave  {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), ampl_slave.data(),  NULL};
        cl::Buffer device_dphase      {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), dphase.data(),      NULL};
        cl::Buffer device_covmat      {context, CL_MEM_READ_WRITE,                       2*dimension*dimension*height*width*sizeof(float), NULL,               NULL};
        cl::Buffer device_ampl_filt   {context, CL_MEM_READ_WRITE,                                             height*width*sizeof(float), NULL,               NULL};
        cl::Buffer device_dphase_filt {context, CL_MEM_READ_WRITE,                                             height*width*sizeof(float), NULL,               NULL};
        cl::Buffer device_coh_filt    {context, CL_MEM_READ_WRITE,                                             height*width*sizeof(float), NULL,               NULL};

        KUT_create.run(cmd_queue, 
                       device_ampl_master,
                       device_ampl_slave,
                       device_dphase,
                       device_covmat,
                       height,
                       width);

        KUT_decompose.run(cmd_queue, 
                          device_covmat,
                          device_ampl_filt,
                          device_dphase_filt,
                          device_coh_filt,
                          height,
                          width);

        cmd_queue.enqueueReadBuffer(device_ampl_filt,   CL_TRUE, 0, height*width*sizeof(float), ampl_filt.data(),   NULL, NULL);
        cmd_queue.enqueueReadBuffer(device_dphase_filt, CL_TRUE, 0, height*width*sizeof(float), dphase_filt.data(), NULL, NULL);
        
        // workaround, since Approx does not work with vectors
        bool flag = true;
        for(unsigned int i = 0; i < ampl_filt.size(); i++) {
            flag = flag && (ampl_master[i] == Approx(ampl_filt  [i]).epsilon( 0.0001 ));
            flag = flag && (dphase     [i] == Approx(dphase_filt[i]).epsilon( 0.0001 ));
        }
        REQUIRE( (flag) );
}
