/* Copyright 2015, 2016 Gerald Baier
 *
 * This file is part of despeckCL.
 *
 * despeckCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * despeckCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with despeckCL. If not, see <http://www.gnu.org/licenses/>.
 */

#include "despeckcl.h"

#include <chrono>
#include <vector>
#include <string>
#include <numeric>

#include "cl_wrappers.h"
#include "data.h"
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
#include "map_filter_tiles.h"


int get_overlap(const int search_window_size,
                const std::vector<int>& patch_sizes,
                const std::vector<int>& scale_sizes) {
    const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
    const int scale_size_max = *std::max_element(scale_sizes.begin(), scale_sizes.end());

    // overlap consists of:
    // - (patch_size_max - 1)/2 + (search_window_size - 1)/2 for similarities
    // - (window_width - 1)/2 for spatial averaging of covariance matrices
    return (patch_size_max - 1)/2 + (search_window_size - 1)/2 + (scale_size_max - 1)/2;
}

void print_params(const int height,
                  const int width,
                  const int search_window_size,
                  const std::vector<int>& patch_sizes,
                  const std::vector<int>& scale_sizes,
                  const int overlap,
                  const std::pair<int, int> tile_dims) {

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

    LOG(INFO) << "tile height: " << tile_dims.first;
    LOG(INFO) << "tile width: " << tile_dims.second;
}


// generic data-agnostic implementation
template<typename Data>
int nlsar_gen(Data& data,
              const int search_window_size,
              const std::vector<int> patch_sizes,
              const std::vector<int> scale_sizes,
              std::map<nlsar::params, nlsar::stats> nlsar_stats,
              const float h_param,
              const float c_param,
              std::vector<std::string> enabled_log_levels)
{
    logging_setup(enabled_log_levels);

    auto cl_devs = get_platform_devs(0);

    const int overlap = get_overlap(search_window_size, patch_sizes, scale_sizes);

    // compute maximum tile size that fits into the GPU's VRAM
    const std::pair<int, int> tile_dims = nlsar::tile_size(cl_devs,
                                                           data.height(),
                                                           data.width(),
                                                           data.dim(),
                                                           search_window_size,
                                                           patch_sizes,
                                                           scale_sizes);

    print_params(data.height(),
                 data.width(),
                 search_window_size,
                 patch_sizes,
                 scale_sizes,
                 overlap,
                 tile_dims);

    // new build kernel interface
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> duration = end-start;
    start = std::chrono::system_clock::now();
    VLOG(0) << "Building kernels";
    std::vector<cl::Context> cl_contexts;
    std::vector<nlsar::cl_wrappers> nlsar_cl_wrappers;
    for(auto & cl_dev : cl_devs) {
        cl::Context cl_context (cl_dev);
        nlsar::cl_wrappers nclw{cl_context, search_window_size, data.dim(), h_param, c_param};

        cl_contexts.push_back(cl_context);
        nlsar_cl_wrappers.push_back(nclw);
    }
    end = std::chrono::system_clock::now();
    duration = end-start;
    VLOG(0) << "Time it took to build all kernels: " << duration.count() << "secs";


    // filtering
    start = std::chrono::system_clock::now();
    auto tm = map_filter_tiles(nlsar::filter_sub_image_overload_set{},
                               data, // same image can be used as input and output
                               data,
                               cl_contexts,
                               nlsar_cl_wrappers,
                               tile_dims,
                               overlap,
                               search_window_size,
                               patch_sizes,
                               scale_sizes,
                               nlsar_stats);

    timings::print(tm);
    end = std::chrono::system_clock::now();
    duration = end-start;
    VLOG(0) << "filtering ran for " << duration.count() << " secs" << std::endl;

    return 0;
}

// specialized wrapper using insar_data
int despeckcl::nlsar(float* ampl_master,
                     float* ampl_slave,
                     float* phase,
                     float* ref_filt,
                     float* phase_filt,
                     float* coh_filt,
                     const int height,
                     const int width,
                     const int search_window_size,
                     const std::vector<int> patch_sizes,
                     const std::vector<int> scale_sizes,
                     std::map<nlsar::params, nlsar::stats> nlsar_stats,
                     const float h_param,
                     const float c_param,
                     std::vector<std::string> enabled_log_levels) {

    // prepare data
    insar_data total_image{ampl_master, ampl_slave, phase,
                           ref_filt, phase_filt, coh_filt,
                           height, width};

    int retval = nlsar_gen(total_image, search_window_size, patch_sizes, scale_sizes, nlsar_stats, h_param, c_param, enabled_log_levels);

    memcpy(ref_filt,   total_image.ref_filt(), total_image.height()*total_image.width()*sizeof(float));
    memcpy(phase_filt, total_image.phase_filt(), total_image.height()*total_image.width()*sizeof(float));
    memcpy(coh_filt,   total_image.coh_filt(), total_image.height()*total_image.width()*sizeof(float));

    return retval;
}


// specialized wrapper using covmat_data
int despeckcl::nlsar(float* covmat_raw,
                     float* covmat_filt,
                     const int height,
                     const int width,
                     const int dim,
                     const int search_window_size,
                     const std::vector<int> patch_sizes,
                     const std::vector<int> scale_sizes,
                     std::map<nlsar::params, nlsar::stats> nlsar_stats,
                     const float h_param,
                     const float c_param,
                     std::vector<std::string> enabled_log_levels) {
    // prepare data
    covmat_data total_image{covmat_raw, covmat_filt, height, width, dim};

    int retval = nlsar_gen(total_image, search_window_size, patch_sizes, scale_sizes, nlsar_stats, h_param, c_param, enabled_log_levels);

    memcpy(covmat_filt,
           total_image.covmat_filt(),
           2 * (size_t)total_image.height() * total_image.width() * total_image.dim() *
               total_image.dim() * sizeof(float));

    return retval;
}
