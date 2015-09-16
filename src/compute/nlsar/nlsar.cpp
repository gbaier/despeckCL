#include "nlsar.h"

#include <CL/cl.h>
#include <chrono>
#include <string.h> // for memcpy

#include "cl_wrappers.h"
#include "insar_data.h"
#include "nlsar_filter_sub_image.h"
#include "sub_images.h"
#include "stats.h"
#include "get_dissims.h"
#include "clcfg.h"
#include "logging.h"
#include "best_params.h"

int nlsar::nlsar(float* master_amplitude, float* slave_amplitude, float* dphase,
                 float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered,
                 const int height, const int width,
                 const int search_window_size,
                 const std::vector<int> patch_sizes,
                 const std::vector<int> scale_sizes,
                 std::vector<std::string> enabled_log_levels)
{
    const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
    const int scale_size_max = *std::max_element(scale_sizes.begin(), scale_sizes.end());

    // FIXME
    const int dimension = 2;
    const int lut_size = 256;
    // overlap consists of:
    // - (patch_size_max - 1)/2 + (search_window_size - 1)/2 for similarities
    // - (window_width - 1)/2 for spatial averaging of covariance matrices
    const int overlap = (patch_size_max - 1)/2 + (search_window_size - 1)/2 + (scale_size_max - 1)/2;

    // the sub image size needs to be picked so that all buffers fit in the GPUs memory
    // Use the following formula to get a rough estimate of the memory consumption
    // sws: search window size
    // sis: sub image size
    // the factor of 5 is due to the number of large buffers
    // similarty, patch_similarity, kullback_leibler, patch_kullback_leibler, weights
    // memory consumption in bytes:
    // sws^2 * sis^2 * n_threads * 4 (float) * 5
    const int sub_image_size = 200;

    logging_setup(enabled_log_levels);


    LOG(INFO) << "filter parameters";
    LOG(INFO) << "search window size: " << search_window_size;
    LOG(INFO) << "patch_size_max: " << patch_size_max;
    LOG(INFO) << "scale_size_max: " << scale_size_max;
    LOG(INFO) << "overlap: " << overlap;

    LOG(INFO) << "data dimensions";
    LOG(INFO) << "height: " << height;
    LOG(INFO) << "width: " << width;

    // legacy opencl setup
    cl::Context context = opencl_setup();

    // new build kernel interface
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds = end-start;
    start = std::chrono::system_clock::now();
    VLOG(0) << "Building kernels";
    cl_wrappers nlsar_cl_wrappers (context, search_window_size, dimension);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    VLOG(0) << "Time it took to build all kernels: " << elapsed_seconds.count() << "secs";

    // prepare data
    insar_data total_image{master_amplitude, slave_amplitude, dphase,
                           amplitude_filtered, dphase_filtered, coherence_filtered,
                           height, width};
    std::map<params, stats> nlsar_stats;
    for(int patch_size : patch_sizes) {
        for(int scale_size : scale_sizes) {
            nlsar_stats.emplace(params{patch_size, scale_size},
                                stats(get_dissims(total_image.get_sub_insar_data(bbox{0,15,0,15}), patch_size, scale_size), patch_size, lut_size));
        }
    }
    total_image.pad(overlap);
    insar_data total_image_temp = total_image;

    // filtering
    LOG(INFO) << "starting filtering";
#pragma omp parallel shared(total_image, total_image_temp)
{
#pragma omp master
    {
    for( auto boundaries : gen_sub_images(total_image.height, total_image.width, sub_image_size, overlap) ) {
#pragma omp task firstprivate(boundaries)
        {
        insar_data sub_image = total_image.get_sub_insar_data(boundaries);
        filter_sub_image(context, nlsar_cl_wrappers, // opencl stuff
                         sub_image, // data
                         search_window_size,
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
