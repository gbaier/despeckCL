#include <CL/cl.h>
#include <chrono>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h> // for memcpy

#include "nlsar.h"
#include "insar_data.h"
#include "nlsar_sub_image.h"
#include "sub_images.h"
#include "stats.h"
#include "get_dissims.h"
#include "clcfg.h"

// opencl kernels

#include "covmat_create.h"
#include "covmat_rescale.h"
#include "covmat_spatial_avg.h"
#include "compute_pixel_similarities_2x2.h"
#include "compute_patch_similarities.h"
#include "weighted_means.h"

nlsar_routines nl_routines;

int nlsar(float* master_amplitude, float* slave_amplitude, float* dphase,
          float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered,
          const int height, const int width,
          const int search_window_size,
          const int patch_size,
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

    // overlap consists of:
    // - (patch_size - 1)/2 + (search_window_size - 1)/2 for similarities
    // - (patch_size - 1)/2 for spatial averaging of covariance matrices
    const int overlap = (patch_size - 1) + (search_window_size - 1)/2;

    LOG(INFO) << "filter parameters";
    LOG(INFO) << "search window size: " << search_window_size;
    LOG(INFO) << "patch_size: " << patch_size;
    LOG(INFO) << "overlap: " << overlap;

    LOG(INFO) << "data dimensions";
    LOG(INFO) << "height: " << height;
    LOG(INFO) << "width: " << width;

    // FIXME
    const int window_width = 3;
    const int dimension = 2;

    stats nlsar_stats(get_dissims(total_image.get_sub_insar_data(bbox{0,15,0,15}), patch_size, window_width), patch_size);

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
    nlsar_routines nl_routines_base;
    VLOG(0) << "Building kernels";
    nl_routines_base.covmat_create_routine                  = new covmat_create                  (16, context);
    nl_routines_base.covmat_rescale_routine                 = new covmat_rescale                 (16, context);
    nl_routines_base.covmat_spatial_avg_routine             = new covmat_spatial_avg             (16, context, window_width);
    nl_routines_base.compute_pixel_similarities_2x2_routine = new compute_pixel_similarities_2x2 (16, context);
    nl_routines_base.weighted_means_routine                 = new weighted_means                 (16, context, search_window_size, dimension);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    VLOG(0) << "Time it took to build all kernels: " << elapsed_seconds.count() << "secs";

#pragma omp threadprivate(nl_routines)

#pragma omp parallel shared(total_image, total_image_temp)
{
// every thread needs its own kernel, in order not to recompile the program again
// a new kernel is created via the copy constructor
    nl_routines.covmat_create_routine                  = new covmat_create                  (*(nl_routines_base.covmat_create_routine));
    nl_routines.covmat_rescale_routine                 = new covmat_rescale                 (*(nl_routines_base.covmat_rescale_routine));
    nl_routines.covmat_spatial_avg_routine             = new covmat_spatial_avg             (*(nl_routines_base.covmat_spatial_avg_routine));
    nl_routines.compute_pixel_similarities_2x2_routine = new compute_pixel_similarities_2x2 (*(nl_routines_base.compute_pixel_similarities_2x2_routine));
    nl_routines.weighted_means_routine                 = new weighted_means                 (*(nl_routines_base.weighted_means_routine));
#pragma omp master
    total_image_temp = total_image;
    for( auto boundaries : gen_sub_images(total_image.height, total_image.width, sub_image_size, overlap) ) {
#pragma omp task firstprivate(boundaries)
        {
        insar_data sub_image = total_image.get_sub_insar_data(boundaries);
        nlsar_sub_image(context, nl_routines, // opencl stuff
                        sub_image, // data
                        search_window_size,
                        patch_size,
                        dimension,
                        nlsar_stats);
        total_image_temp.write_sub_insar_data(sub_image, overlap, boundaries);
        }
#pragma omp taskwait
        total_image = total_image_temp;
        // the next two lines pad the overlap borders with the filtered pixel values
        total_image.unpad(overlap);
        total_image.pad(overlap);
    }
}
    total_image.unpad(overlap);
    LOG(INFO) << "filtering done";

    memcpy(amplitude_filtered, total_image.amp_filt, sizeof(float)*height*width);
    memcpy(dphase_filtered, total_image.phi_filt, sizeof(float)*height*width);
    memcpy(coherence_filtered, total_image.coh_filt, sizeof(float)*height*width);

    return 0;
}
