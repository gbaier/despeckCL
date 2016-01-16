#include "despeckcl.h"

#include <chrono>
#include <vector>
#include <string>

#include "cl_wrappers.h"
#include "insar_data.h"
#include "tile_iterator.h"
#include "tile_size.h"
#include "nlsar_filter_sub_image.h"
#include "sub_images.h"
#include "stats.h"
#include "get_stats.h"
#include "clcfg.h"
#include "logging.h"
#include "best_params.h"
#include "timings.h"

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
                     const std::tuple<int, int, int> training_dims,
                     std::vector<std::string> enabled_log_levels)
{
    timings::map tm;

    const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
    const int scale_size_max = *std::max_element(scale_sizes.begin(), scale_sizes.end());

    // FIXME
    const int dimension = 2;
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

    const int tile_size = nlsar::tile_size(context, search_window_size, patch_sizes, scale_sizes);

    // new build kernel interface
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> duration = end-start;
    start = std::chrono::system_clock::now();
    VLOG(0) << "Building kernels";
    nlsar::cl_wrappers nlsar_cl_wrappers (context, search_window_size, dimension);
    end = std::chrono::system_clock::now();
    duration = end-start;
    VLOG(0) << "Time it took to build all kernels: " << duration.count() << "secs";

    // prepare data
    insar_data_shared total_image{ampl_master, ampl_slave, dphase,
                                  ampl_filt, dphase_filt, coh_filt,
                                  height, width};

    VLOG(0) << "Training weighting kernels";
    auto training_data = tile(total_image, std::get<0>(training_dims), std::get<1>(training_dims), std::get<2>(training_dims), 0).get();
    start = std::chrono::system_clock::now();
    std::map<nlsar::params, nlsar::stats> nlsar_stats = nlsar::training::get_stats(patch_sizes, scale_sizes, training_data, context);
    end = std::chrono::system_clock::now();
    duration = end-start;
    VLOG(0) << "training ran for: " << duration.count() << " secs";

    // filtering
    start = std::chrono::system_clock::now();
    LOG(INFO) << "starting filtering";
#pragma omp parallel shared(total_image)
{
#pragma omp master
    {
    for( auto imgtile : tile_iterator(total_image, tile_size, overlap, overlap) ) {
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
    duration = end-start;
    std::cout << "filtering ran for " << duration.count() << " secs" << std::endl;

    return 0;
}
