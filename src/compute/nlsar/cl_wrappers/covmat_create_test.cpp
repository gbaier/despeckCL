#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <complex>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include "covmat_create.h"

using namespace nlsar;

TEST_CASE( "covmat_create", "[cl_kernels]" ) {

        // data setup
        const int height = 10;
        const int width = 10;
        const int dimension = 2;

        std::vector<float> ampl_master (                      height*width, 1.0);
        std::vector<float> ampl_slave  (                      height*width, 2.0);
        std::vector<float> dphase      (                      height*width, 0.5);
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
        cl::Buffer device_dphase      {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), dphase.data(),      NULL};
        cl::Buffer device_covmat      {context, CL_MEM_READ_WRITE,                       2*dimension*dimension*height*width*sizeof(float), NULL,               NULL};

        KUT.run(cmd_queue, 
                device_ampl_master,
                device_ampl_slave,
                device_dphase,
                device_covmat,
                height,
                width);

        cmd_queue.enqueueReadBuffer(device_covmat, CL_TRUE, 0, 2*height*width*dimension*dimension*sizeof(float), covmat.data(), NULL, NULL);

        // workaround, since Approx does not work with vectors
        bool flag = true;
        for(int i = 0; i < height*width; i++) {
            const int offset = height*width;
            //std::complex<float> el00 {covmat[i],            covmat[i +   offset]};
            std::complex<float> el00 {covmat[i],            covmat[i +   offset]};
            std::complex<float> el01 {covmat[i + 2*offset], covmat[i + 3*offset]};
            std::complex<float> el10 {covmat[i + 4*offset], covmat[i + 5*offset]};
            std::complex<float> el11 {covmat[i + 6*offset], covmat[i + 7*offset]};
            flag = flag && (0 == Approx(std::abs(el00*el11 - el01*el10)).epsilon( 0.0001 ));
        }
        REQUIRE( ( flag ) );
}

TEST_CASE( "covmat_create sanity check", "[cl_kernels]" ) {

        // data setup
        const int height = 40;
        const int width = 50;
        const int dimension = 2;

        std::vector<float> ampl_master (                      height*width, -1.0);
        std::vector<float> ampl_slave  (                      height*width, -1.0);
        std::vector<float> dphase      (                      height*width, -1.0);
        std::vector<float> covmat      (2*dimension*dimension*height*width, -1.0);

        static std::default_random_engine rand_eng{};
        static std::gamma_distribution<float>        dist_ampl(1.0, 5.0);
        static std::uniform_real_distribution<float> dist_dphase(1.0, 5.0);

        for(int i = 0; i < height*width; i++) {
            ampl_master[i] = dist_ampl(rand_eng);
            ampl_slave [i] = dist_ampl(rand_eng);
            dphase     [i] = dist_dphase(rand_eng);
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
        cl::Buffer device_dphase      {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                       height*width*sizeof(float), dphase.data(),      NULL};
        cl::Buffer device_covmat      {context, CL_MEM_READ_WRITE,                       2*dimension*dimension*height*width*sizeof(float), NULL,               NULL};

        KUT.run(cmd_queue, 
                device_ampl_master,
                device_ampl_slave,
                device_dphase,
                device_covmat,
                height,
                width);

        cmd_queue.enqueueReadBuffer(device_covmat, CL_TRUE, 0, 2*height*width*dimension*dimension*sizeof(float), covmat.data(), NULL, NULL);

        // workaround, since Approx does not work with vectors
        bool flag = true;
        for(int i = 0; i < height*width; i++) {
            const int offset = height*width;
            //std::complex<float> el00 {covmat[i],            covmat[i +   offset]};
            std::complex<float> el00 {covmat[i],            covmat[i +   offset]};
            std::complex<float> el01 {covmat[i + 2*offset], covmat[i + 3*offset]};
            std::complex<float> el10 {covmat[i + 4*offset], covmat[i + 5*offset]};
            std::complex<float> el11 {covmat[i + 6*offset], covmat[i + 7*offset]};
            flag = flag && (0 <= el00.real());
            flag = flag && (0 == Approx(el00.imag()).epsilon( 0.0001 ));
            flag = flag && (std::abs(el01) == Approx(std::abs(el10)).epsilon( 0.0001 ));
            flag = flag && (-el01.imag() == Approx(el10.imag()).epsilon( 0.0001 ));
            flag = flag && (0 < el11.real());
        }
        REQUIRE( ( flag ) );
}
