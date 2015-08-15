#include <CL/cl.h>
#include <chrono>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h> // for memcpy

#include "boxcar.h"
#include "boxcar_sub_image.h"
#include "insar_data.h"

INITIALIZE_EASYLOGGINGPP

extern boxcar_wrapper boxcar_routine;

void boxcar(float* master_amplitude,
            float* slave_amplitude,
            float* dphase,
            float* ampl_filt,
            float* dphase_filt,
            float* coh_filt,
            const int height,
            const int width,
            const int window_width,
            std::vector<el::Level> enabled_log_levels)
{
    el::Configurations log_config;
    log_config.setToDefault();
    log_config.setGlobally(el::ConfigurationType::Enabled, "false");

    log_config.set(el::Level::Info,    el::ConfigurationType::Format, "[%level] %msg");
    log_config.set(el::Level::Verbose, el::ConfigurationType::Format, "[%level] %msg");
    log_config.set(el::Level::Debug,   el::ConfigurationType::Format, "[%level] %fbase:%line %msg");
    log_config.set(el::Level::Warning, el::ConfigurationType::Format, "[%level] %fbase:%line %msg");
    log_config.set(el::Level::Fatal,   el::ConfigurationType::Format, "[%level] %fbase:%line %msg");
    for(auto level : enabled_log_levels) {
        log_config.set(level, el::ConfigurationType::Enabled, "true");
    }
    el::Loggers::reconfigureLogger("default", log_config);

    insar_data total_image{master_amplitude,
                           slave_amplitude,
                           dphase,
                           ampl_filt,
                           dphase_filt,
                           coh_filt,
                           height,
                           width};

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
    boxcar_wrapper boxcar_routine_base{16, context, window_width};
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    VLOG(0) << "Time it took to build the kernels: " << elapsed_seconds.count() << "secs";


    total_image.pad(overlap);
    // timing
    double boxcar_timing = 0.0;
    

    LOG(INFO) << "starting filtering";
#pragma omp threadprivate(boxcar_routine)

#pragma omp parallel shared(total_image)
{
    // every thread needs its own kernel, in order not to recompile the program again
    // a new kernel is created via the copy constructor
    boxcar_wrapper boxcar_routine{boxcar_routine_base};
#pragma omp master
    for( auto boundaries : gen_sub_images(total_image.height, total_image.width, sub_image_size, overlap) ) {
#pragma omp task firstprivate(boundaries)
        {
        insar_data sub_image = total_image.get_sub_insar_data(boundaries);
        boxcar_sub_image(context, boxcar_routine_base, // opencl stuff
                         sub_image, // data
                         window_width); // filter parameters
        total_image.write_sub_insar_data(sub_image, overlap, boundaries);
        }
    }
#pragma omp taskwait
#pragma omp critical
    {
        boxcar_timing += boxcar_routine_base.elapsed_seconds.count();
    }
}
    total_image.unpad(overlap);
    LOG(INFO) << "filtering done";

    memcpy(ampl_filt,   total_image.amp_filt, sizeof(float)*height*width);
    memcpy(dphase_filt, total_image.phi_filt, sizeof(float)*height*width);
    memcpy(coh_filt,    total_image.coh_filt, sizeof(float)*height*width);

    VLOG(0) << "elapsed time for boxcar: " << boxcar_timing << " secs";

    return;
}
