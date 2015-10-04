#include "despeckcl.h"

#include <CL/cl.h>
#include <chrono>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h> // for memcpy
#include "logging.h"

#include "boxcar_sub_image.h"
#include "insar_data.h"


void despeckcl::boxcar(float* ampl_master,
                       float* ampl_slave,
                       float* dphase,
                       float* ampl_filt,
                       float* dphase_filt,
                       float* coh_filt,
                       const int height,
                       const int width,
                       const int window_width,
                       std::vector<std::string> enabled_log_levels)
{
    logging_setup(enabled_log_levels);

    insar_data total_image{ampl_master, ampl_slave, dphase,
                           ampl_filt, dphase_filt, coh_filt,
                           height, width};

    LOG(INFO) << "filter parameters";
    LOG(INFO) << "window width: " << window_width;

    LOG(INFO) << "data dimensions";
    LOG(INFO) << "height: " << height;
    LOG(INFO) << "width: " << width;

    const int overlap = (window_width - 1) / 2;

    // legacy opencl setup
    cl::Context context = opencl_setup();

    // filtering
    LOG(INFO) << "starting filtering";
    // the sub image size needs to be picked so that all buffers fit in the GPUs memory
    const int sub_image_size = 150;

    // new build kernel interface
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds = end-start;
    start = std::chrono::system_clock::now();
    VLOG(0) << "Building kernel";
    boxcar_wrapper boxcar_routine{16, context, window_width};
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    VLOG(0) << "Time it took to build the kernels: " << elapsed_seconds.count() << "secs";


    total_image.pad(overlap);
    // timing
    double boxcar_timing = 0.0;
    

    LOG(INFO) << "starting filtering";
#pragma omp parallel shared(total_image)
{
#pragma omp master
    for( auto boundaries : gen_sub_images(total_image.height, total_image.width, sub_image_size, overlap) ) {
#pragma omp task firstprivate(boundaries)
        {
        insar_data sub_image = total_image.get_sub_insar_data(boundaries);
        boxcar_sub_image(context, boxcar_routine, // opencl stuff
                         sub_image); // data
        total_image.write_sub_insar_data(sub_image, overlap, boundaries);
        }
    }
#pragma omp taskwait
}
    total_image.unpad(overlap);
    LOG(INFO) << "filtering done";

    memcpy(ampl_filt,   total_image.amp_filt, sizeof(float)*height*width);
    memcpy(dphase_filt, total_image.phi_filt, sizeof(float)*height*width);
    memcpy(coh_filt,    total_image.coh_filt, sizeof(float)*height*width);

    VLOG(0) << "elapsed time for boxcar: " << boxcar_timing << " secs";

    return;
}
