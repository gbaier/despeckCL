#include "despeckcl.h"

#include <CL/cl.h>
#include <chrono>
#include <string.h> // for memcpy
#include <vector>
#include <string>
#include <omp.h>

#include "cl_wrappers.h"
#include "insar_data.h"
#include "tile_iterator.h"
#include "nlsar_filter_sub_image.h"
#include "sub_images.h"
#include "stats.h"
#include "get_dissims.h"
#include "clcfg.h"
#include "logging.h"
#include "best_params.h"
#include "timings.h"


int round_down(const int num, const int multiple)
{
     int remainder = num % multiple;
     return num - remainder;
}

int return_sub_image_size(cl::Context context,
                          const int search_window_size,
                          const std::vector<int>& patch_sizes,
                          const std::vector<int>& scale_sizes)
{
    const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
    const int scale_size_max = *std::max_element(scale_sizes.begin(), scale_sizes.end());

    const int overlap = search_window_size + patch_size_max + scale_size_max - 3;

    const int n_params = patch_sizes.size() * scale_sizes.size();
    VLOG(0) << "number of parameters = " << n_params;

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::Device dev = devices[0];

    int long global_mem_size;
    dev.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem_size);
    VLOG(0) << "global memory size = " << global_mem_size;

    int long max_mem_alloc_size;
    dev.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_mem_alloc_size);
    VLOG(0) << "maximum memory allocation size = " << max_mem_alloc_size;

    // Most the most memory is required for storing weights, so only this is taken into account.
    // required bytes per pixel = reg_bpp
    const int req_bpp = 4 * search_window_size * search_window_size * n_params;
    const int n_pixels_global = global_mem_size    / (req_bpp * omp_get_num_threads());
    const int n_pixels_alloc  = max_mem_alloc_size /  req_bpp;
    const int n_pixels = std::min(n_pixels_global, n_pixels_alloc);

    const int sub_img_size_fit        = std::sqrt(n_pixels);
    const int sub_img_size_fit_rounded = round_down(sub_img_size_fit, 64);

    VLOG(0) << "sub_image_size_fit = "         << sub_img_size_fit;
    VLOG(0) << "sub_image_size_fit_rounded = " << sub_img_size_fit_rounded;

    const float safety_factor = 0.75;
    int safe_sub_img_size = 0;
    if (float(sub_img_size_fit_rounded)/sub_img_size_fit < safety_factor) {
        safe_sub_img_size = sub_img_size_fit_rounded;
    } else {
        safe_sub_img_size = sub_img_size_fit_rounded - 64;
    }
    VLOG(0) << "safe_sub_image_size = " << safe_sub_img_size;
    safe_sub_img_size += overlap;
    VLOG(0) << "safe_sub_image_size with overlap = " << safe_sub_img_size;


    return safe_sub_img_size;
}

int despeckcl::nlsar(float* ampl_master,
                     float* ampl_slave,
                     float* dphase,
                     float* ampl_filt,
                     float* dphase_filt,
                     float* coh_filt,
                     const int height,
                     const int width,
                     const int search_window_size,
                     const std::vector<int> patch_sizes,
                     const std::vector<int> scale_sizes,
                     const bbox training_dims,
                     std::vector<std::string> enabled_log_levels)
{
    timings::map tm;

    const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
    const int scale_size_max = *std::max_element(scale_sizes.begin(), scale_sizes.end());

    // FIXME
    const int dimension = 2;
    const int lut_size = 1024;
    // overlap consists of:
    // - (patch_size_max - 1)/2 + (search_window_size - 1)/2 for similarities
    // - (window_width - 1)/2 for spatial averaging of covariance matrices
    const int overlap = (patch_size_max - 1)/2 + (search_window_size - 1)/2 + (scale_size_max - 1)/2;

    logging_setup(enabled_log_levels);

    LOG(INFO) << "filter parameters";
    LOG(INFO) << "search window size: " << search_window_size;
    auto intvec2string = [] (std::vector<int> ints) { return std::accumulate(ints.begin(), ints.end(), (std::string)"",
                                                                             [] (std::string acc, int i) {return acc + std::to_string(i) + ", ";});
                                                    };

    LOG(INFO) << "patch_sizes: " << intvec2string(patch_sizes);
    LOG(INFO) << "scale_sizes: " << intvec2string(scale_sizes);
    LOG(INFO) << "overlap: " << overlap;

    LOG(INFO) << "data dimensions";
    LOG(INFO) << "height: " << height;
    LOG(INFO) << "width: " << width;

    // legacy opencl setup
    cl::Context context = opencl_setup();

    const int sub_image_size = return_sub_image_size(context, search_window_size, patch_sizes, scale_sizes);

    // new build kernel interface
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds = end-start;
    start = std::chrono::system_clock::now();
    VLOG(0) << "Building kernels";
    nlsar::cl_wrappers nlsar_cl_wrappers (context, search_window_size, dimension);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    VLOG(0) << "Time it took to build all kernels: " << elapsed_seconds.count() << "secs";

    // prepare data
    insar_data total_image{ampl_master, ampl_slave, dphase,
                           ampl_filt, dphase_filt, coh_filt,
                           height, width};
    std::map<nlsar::params, nlsar::stats> nlsar_stats;
    for(int patch_size : patch_sizes) {
        for(int scale_size : scale_sizes) {
            std::vector<float> dissims  = nlsar::get_dissims(context,
                                                             tile{total_image,
                                                                  training_dims.h_low,
                                                                  training_dims.w_low,
                                                                  training_dims.h_up - training_dims.h_low,
                                                                  0}.get(),
                                                             patch_size, scale_size);
            nlsar_stats.emplace(nlsar::params{patch_size, scale_size},
                                nlsar::stats(dissims, lut_size));
        }
    }
    // filtering
    start = std::chrono::system_clock::now();
    LOG(INFO) << "starting filtering";
#pragma omp parallel shared(total_image)
{
#pragma omp master
    {
    for( auto imgtile : tile_iterator(total_image, sub_image_size, overlap, overlap) ) {
#pragma omp task firstprivate(imgtile)
        {
        try {
            timings::map tm_sub = filter_sub_image(context, nlsar_cl_wrappers, // opencl stuff
                                                   imgtile.get(), // data
                                                   search_window_size,
                                                   patch_sizes,
                                                   scale_sizes,
                                                   dimension,
                                                   nlsar_stats);
#pragma omp critical
            tm = timings::join(tm, tm_sub);
        } catch (cl::Error error) {
            LOG(ERROR) << error.what() << "(" << error.err() << ")";
            LOG(ERROR) << "ERR while filtering sub image";
            std::terminate();
        }
        imgtile.write(total_image);
        }
    }
#pragma omp taskwait
    }
}
    LOG(INFO) << "filtering done";
    timings::print(tm);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end-start;
    std::cout << "filtering ran for " << duration.count() << " secs" << std::endl;

    memcpy(ampl_filt,   total_image.amp_filt, sizeof(float)*height*width);
    memcpy(dphase_filt, total_image.phi_filt, sizeof(float)*height*width);
    memcpy(coh_filt,    total_image.coh_filt, sizeof(float)*height*width);

    return 0;
}
