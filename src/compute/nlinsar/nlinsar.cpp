#include <CL/cl.h>
#include <chrono>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h> // for memcpy

#include "nlinsar.h"
#include "insar_data.h"
#include "nlinsar_sub_image.h"
#include "sub_images.h"
#include "insarsim_simu.h"

#include "clcfg.h"
// opencl kernels
#include "precompute_similarities_1st_pass.h"
#include "precompute_similarities_2nd_pass.h"
#include "precompute_patch_similarities.h"
#include "compute_weights.h"
#include "compute_number_of_looks.h"
#include "transpose.h"
#include "precompute_filter_values.h"
#include "compute_insar.h"
// cpu routine
#include "smoothing.h"

nlinsar::routines nl_routines;

int nlinsar::nlinsar(float* master_amplitude, float* slave_amplitude, float* dphase,
                     float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered,
                     const int height, const int width,
                     const int search_window_size,
                     const int patch_size,
                     const int niter,
                     const int lmin,
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

    insar_data total_image{master_amplitude, slave_amplitude, dphase,
                           amplitude_filtered, dphase_filtered, coherence_filtered,
                           height, width};

    const int overlap = (patch_size - 1)/2 + (search_window_size - 1)/2;

    const int patch_area = std::pow(patch_size, 2);
    float h_theo = 0;
    switch(patch_size) {
        case 3:
            h_theo = 0.58; // alpha = 0.92
            break;
        case 7:
            h_theo = 0.244; // alpha = 0.92
            break;
        default:
            h_theo = quantile_insar(patch_size, 0.92);
    }
    const float h_para = h_theo * patch_area;
    const float T_para = 2.0 / h_para * M_PI / 4 * patch_area;

    LOG(INFO) << "filter parameters";
    LOG(INFO) << "search window size: " << search_window_size;
    LOG(INFO) << "patch_size: " << patch_size;
    LOG(INFO) << "overlap: " << overlap;
    LOG(INFO) << "h_para: " << h_para;
    LOG(INFO) << "T_para: " << T_para;
    LOG(INFO) << "niter: " << niter;
    LOG(INFO) << "lmin: " << lmin;

    LOG(INFO) << "data dimensions";
    LOG(INFO) << "height: " << height;
    LOG(INFO) << "width: " << width;

    // legacy opencl setup
    cl::Context context = opencl_setup();

    // filtering
    LOG(INFO) << "starting filtering";
    // the sub image size needs to be picked so that all buffers fit in the GPUs memory
    // Use the following formula to get a rough estimate of the memory consumption
    // sws: search window size
    // sis: sub image size
    // the factor of 5 is due to the number of large buffers
    // similarty, patch_similarity, kullback_leibler, patch_kullback_leibler, weights
    // memory consumption in bytes:
    // sws^2 * sis^2 * n_threads * 4 (float) * 5
    const int sub_image_size = 80;

    total_image.pad(overlap);

    insar_data total_image_temp = total_image;
    
    // new build kernel interface
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds = end-start;
    start = std::chrono::system_clock::now();
    nlinsar::routines nl_routines_base;
    VLOG(0) << "Building kernels";
    nl_routines_base.precompute_patch_similarities_routine    = new precompute_patch_similarities   (14, context, patch_size);
    nl_routines_base.precompute_similarities_1st_pass_routine = new precompute_similarities_1st_pass(16, context);
    nl_routines_base.precompute_similarities_2nd_pass_routine = new precompute_similarities_2nd_pass(16, context);
    nl_routines_base.compute_weights_routine                  = new compute_weights                 (64, context);
    nl_routines_base.compute_number_of_looks_routine          = new compute_number_of_looks         (16, context);
    nl_routines_base.transpose_routine                        = new transpose                       (32, context, 8, 32);
    nl_routines_base.precompute_filter_values_routine         = new precompute_filter_values        (16, context);
    nl_routines_base.compute_insar_routine                    = new compute_insar                   (14, context, search_window_size);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    VLOG(0) << "Time it took to build all kernels: " << elapsed_seconds.count() << "secs";

    double precompute_similarities_1st_pass_timing = 0.0;
    double precompute_similarities_2nd_pass_timing = 0.0;
    double precompute_patch_similarities_timing = 0.0;
    double compute_weights_timing = 0.0;
    double compute_number_of_looks_timing = 0.0;
    double transpose_timing = 0.0;
    double precompute_filter_values_timing = 0.0;
    double compute_insar_timing = 0.0;
    double smoothing_timing = 0.0;
#pragma omp threadprivate(nl_routines)

#pragma omp parallel shared(total_image, total_image_temp)
{
// every thread needs its own kernel, in order not to recompile the program again
// a new kernel is created via the copy constructor
    nl_routines.precompute_patch_similarities_routine    = new precompute_patch_similarities   (*(nl_routines_base.precompute_patch_similarities_routine));
    nl_routines.precompute_similarities_1st_pass_routine = new precompute_similarities_1st_pass(*(nl_routines_base.precompute_similarities_1st_pass_routine));
    nl_routines.precompute_similarities_2nd_pass_routine = new precompute_similarities_2nd_pass(*(nl_routines_base.precompute_similarities_2nd_pass_routine));
    nl_routines.compute_weights_routine                  = new compute_weights                 (*(nl_routines_base.compute_weights_routine));
    nl_routines.compute_number_of_looks_routine          = new compute_number_of_looks         (*(nl_routines_base.compute_number_of_looks_routine));
    nl_routines.transpose_routine                        = new transpose                       (*(nl_routines_base.transpose_routine));
    nl_routines.precompute_filter_values_routine         = new precompute_filter_values        (*(nl_routines_base.precompute_filter_values_routine));
    nl_routines.compute_insar_routine                    = new compute_insar                   (*(nl_routines_base.compute_insar_routine));
    nl_routines.smoothing_routine                        = new smoothing;

#pragma omp master
    for(int n = 0; n<niter; n++) {
        LOG(INFO) << "Iteration " << n + 1 << " of " << niter;
        total_image_temp = total_image;
        for( auto boundaries : gen_sub_images(total_image.height, total_image.width, sub_image_size, overlap) ) {
#pragma omp task firstprivate(boundaries)
            {
            insar_data sub_image = total_image.get_sub_insar_data(boundaries);
            nlinsar_sub_image(context, nl_routines, // opencl stuff
                              sub_image, // data
                              search_window_size, patch_size, lmin, h_para, T_para); // filter parameters
            total_image_temp.write_sub_insar_data(sub_image, overlap, boundaries);
            }
        }
#pragma omp taskwait
        total_image = total_image_temp;
        // the next two lines pad the overlap borders with the filtered pixel values
        total_image.unpad(overlap);
        total_image.pad(overlap);
    }
#pragma omp critical
    {
        precompute_similarities_1st_pass_timing += nl_routines.precompute_similarities_1st_pass_routine->elapsed_seconds.count();
        precompute_similarities_2nd_pass_timing += nl_routines.precompute_similarities_2nd_pass_routine->elapsed_seconds.count();
        precompute_patch_similarities_timing    += nl_routines.precompute_patch_similarities_routine->elapsed_seconds.count();
        compute_weights_timing                  += nl_routines.compute_weights_routine->elapsed_seconds.count();
        compute_number_of_looks_timing          += nl_routines.compute_number_of_looks_routine->elapsed_seconds.count();
        transpose_timing                        += nl_routines.transpose_routine->elapsed_seconds.count();
        precompute_filter_values_timing         += nl_routines.precompute_filter_values_routine->elapsed_seconds.count();
        compute_insar_timing                    += nl_routines.compute_insar_routine->elapsed_seconds.count();
        smoothing_timing                        += nl_routines.smoothing_routine->elapsed_seconds.count();
    }
}
    total_image.unpad(overlap);
    LOG(INFO) << "filtering done";

    memcpy(amplitude_filtered, total_image.amp_filt, sizeof(float)*height*width);
    memcpy(dphase_filtered, total_image.phi_filt, sizeof(float)*height*width);
    memcpy(coherence_filtered, total_image.coh_filt, sizeof(float)*height*width);

    VLOG(0) << "elapsed time for precompute_similarities_1st_pass: " << precompute_similarities_1st_pass_timing << " secs";
    VLOG(0) << "elapsed time for precompute_similarities_2nd_pass: " << precompute_similarities_2nd_pass_timing << " secs";
    VLOG(0) << "elapsed time for precompute_patch_similarities: "    << precompute_patch_similarities_timing    << " secs";
    VLOG(0) << "elapsed time for compute_weights: "                  << compute_weights_timing                  << " secs";
    VLOG(0) << "elapsed time for compute_number_of_looks: "          << compute_number_of_looks_timing          << " secs";
    VLOG(0) << "elapsed time for transpose: "                        << transpose_timing                        << " secs";
    VLOG(0) << "elapsed time for precompute_filter_values: "         << precompute_filter_values_timing         << " secs";
    VLOG(0) << "elapsed time for compute_insar: "                    << compute_insar_timing                    << " secs";
    VLOG(0) << "elapsed time for smoothing: "                        << smoothing_timing                        << " secs";

    return 0;
}
