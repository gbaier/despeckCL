#include "despeckcl.h"

#include <CL/cl.h>
#include <chrono>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h> // for memcpy
#include <vector>
#include <string>

#include "insar_data.h"
#include "tile_iterator.h"
#include "nlinsar_sub_image.h"
#include "sub_images.h"
#include "insarsim_simu.h"
#include "logging.h"

#include "clcfg.h"
#include "cl_wrappers.h"

int despeckcl::nlinsar(float* ampl_master,
                       float* ampl_slave,
                       float* dphase,
                       float* ampl_filt,
                       float* dphase_filt,
                       float* coh_filt,
                       const int height,
                       const int width,
                       const int search_window_size,
                       const int patch_size,
                       const int niter,
                       const int lmin,
                       std::vector<std::string> enabled_log_levels)
{
    logging_setup(enabled_log_levels);

    insar_data total_image{ampl_master, ampl_slave, dphase,
                           ampl_filt, dphase_filt, coh_filt,
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
            h_theo = nlinsar::simu::quantile_insar(patch_size, 0.92);
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
    LOG(INFO);
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

    
    // new build kernel interface
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds = end-start;
    start = std::chrono::system_clock::now();
    VLOG(0) << "Building kernels";
    nlinsar::cl_wrappers nl_routines_base(context, search_window_size, patch_size, 16);
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

    insar_data total_image_temp = total_image;
#pragma omp parallel shared(total_image, total_image_temp)
{
// every thread needs its own kernel, in order not to recompile the program again
// a new kernel is created via the copy constructor
    nlinsar::cl_wrappers nl_routines(nl_routines_base);
#pragma omp master
    for(int n = 0; n<niter; n++) {
        LOG(INFO) << "Iteration " << n + 1 << " of " << niter;
        total_image_temp = total_image;
        for( auto imgtile : tile_iterator(total_image, sub_image_size, overlap, overlap) ) {
#pragma omp task firstprivate(imgtile)
            {
            nlinsar_sub_image(context, nl_routines, // opencl stuff
                             imgtile.get(),
                             search_window_size, patch_size, lmin, h_para, T_para); // filter parameters
            imgtile.write(total_image_temp);
            }
        }
#pragma omp taskwait
        total_image = total_image_temp;
    }
#pragma omp critical
    {
        precompute_similarities_1st_pass_timing += nl_routines.precompute_similarities_1st_pass_routine.elapsed_seconds.count();
        precompute_similarities_2nd_pass_timing += nl_routines.precompute_similarities_2nd_pass_routine.elapsed_seconds.count();
        precompute_patch_similarities_timing    += nl_routines.precompute_patch_similarities_routine.elapsed_seconds.count();
        compute_weights_timing                  += nl_routines.compute_weights_routine.elapsed_seconds.count();
        compute_number_of_looks_timing          += nl_routines.compute_number_of_looks_routine.elapsed_seconds.count();
        transpose_timing                        += nl_routines.transpose_routine.elapsed_seconds.count();
        precompute_filter_values_timing         += nl_routines.precompute_filter_values_routine.elapsed_seconds.count();
        compute_insar_timing                    += nl_routines.compute_insar_routine.elapsed_seconds.count();
        smoothing_timing                        += nl_routines.smoothing_routine.elapsed_seconds.count();
    }
}
    LOG(INFO) << "filtering done";

    memcpy(ampl_filt,   total_image.amp_filt, sizeof(float)*height*width);
    memcpy(dphase_filt, total_image.phi_filt, sizeof(float)*height*width);
    memcpy(coh_filt,    total_image.coh_filt, sizeof(float)*height*width);

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
