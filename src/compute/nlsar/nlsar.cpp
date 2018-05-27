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
#include "map_filter_tiles.h"

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
                     std::vector<std::string> enabled_log_levels)
{
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

    std::pair<int, int> tile_dims = nlsar::tile_size(context, height, width, dimension, search_window_size, patch_sizes, scale_sizes);

    LOG(INFO) << "tile height: " << tile_dims.first;
    LOG(INFO) << "tile width: " << tile_dims.second;

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
    insar_data total_image{ampl_master, ampl_slave, phase,
                           ref_filt, phase_filt, coh_filt,
                           height, width};

    // filtering
    start = std::chrono::system_clock::now();
    auto tm = map_filter_tiles(nlsar::filter_sub_image,
                               total_image, // same image can be used as input and output
                               total_image,
                               context,
                               nlsar_cl_wrappers,
                               tile_dims,
                               overlap,
                               search_window_size,
                               patch_sizes,
                               scale_sizes,
                               dimension,
                               nlsar_stats);

    timings::print(tm);
    end = std::chrono::system_clock::now();
    duration = end-start;
    std::cout << "filtering ran for " << duration.count() << " secs" << std::endl;

    memcpy(ref_filt,   total_image.ref_filt(), total_image.height*total_image.width*sizeof(float));
    memcpy(phase_filt, total_image.phase_filt(), total_image.height*total_image.width*sizeof(float));
    memcpy(coh_filt,   total_image.coh_filt(), total_image.height*total_image.width*sizeof(float));

    return 0;
}
