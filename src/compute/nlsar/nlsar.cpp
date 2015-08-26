#include "nlsar.h"

#include <CL/cl.h>
#include <chrono>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h> // for memcpy

#include "nlsar_routines.h"
#include "insar_data.h"
#include "nlsar_sub_image.h"
#include "sub_images.h"
#include "stats.h"
#include "get_dissims.h"
#include "clcfg.h"
#include "logging.h"

int nlsar(float* master_amplitude, float* slave_amplitude, float* dphase,
          float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered,
          const int height, const int width,
          const int search_window_size,
          const int patch_size,
          std::vector<el::Level> enabled_log_levels)
{
    // FIXME
    const int window_width = 3;
    const int dimension = 2;

    logging_setup(enabled_log_levels);

    insar_data total_image{master_amplitude, slave_amplitude, dphase,
                           amplitude_filtered, dphase_filtered, coherence_filtered,
                           height, width};

    // overlap consists of:
    // - (patch_size - 1)/2 + (search_window_size - 1)/2 for similarities
    // - (window_width - 1)/2 for spatial averaging of covariance matrices
    const int overlap = (patch_size - 1)/2 + (search_window_size - 1)/2 + (window_width - 1)/2;

    LOG(INFO) << "filter parameters";
    LOG(INFO) << "search window size: " << search_window_size;
    LOG(INFO) << "patch_size: " << patch_size;
    LOG(INFO) << "overlap: " << overlap;

    LOG(INFO) << "data dimensions";
    LOG(INFO) << "height: " << height;
    LOG(INFO) << "width: " << width;

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
    const int sub_image_size = 200;

    total_image.pad(overlap);

    insar_data total_image_temp = total_image;
    
    // new build kernel interface
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds = end-start;
    start = std::chrono::system_clock::now();
    VLOG(0) << "Building kernels";
    nlsar_routines nl_routines (context, search_window_size, patch_size, window_width, dimension);

    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    VLOG(0) << "Time it took to build all kernels: " << elapsed_seconds.count() << "secs";

#pragma omp parallel shared(total_image, total_image_temp)
{
#pragma omp master
    {
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
    }
#pragma omp taskwait
    total_image = total_image_temp;
    }
}
    total_image.unpad(overlap);
    LOG(INFO) << "filtering done";

    memcpy(amplitude_filtered, total_image.amp_filt, sizeof(float)*height*width);
    memcpy(dphase_filtered, total_image.phi_filt, sizeof(float)*height*width);
    memcpy(coherence_filtered, total_image.coh_filt, sizeof(float)*height*width);

    return 0;
}
