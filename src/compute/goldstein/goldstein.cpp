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

#include <CL/cl.h>
#include <chrono>
#include <vector>
#include <string>
#include <omp.h>

#include "cl_wrappers.h"
#include "data.h"
#include "tile_iterator.h"
#include "tile_size.h"
#include "goldstein_filter_sub_image.h"
#include "sub_images.h"
#include "clcfg.h"
#include "logging.h"
#include "timings.h"
#include "map_filter_tiles.h"

int despeckcl::goldstein(float* ampl_master,
                         float* ampl_slave,
                         float* phase,
                         float* ref_filt,
                         float* phase_filt,
                         float* coh_filt,
                         const unsigned int height,
                         const unsigned int width,
                         const unsigned int patch_size,
                         const unsigned int overlap,
                         const float alpha,
                         std::vector<std::string> enabled_log_levels)
{
    omp_set_num_threads(1);
    logging_setup(enabled_log_levels);

    LOG(INFO) << "filter parameters";
    LOG(INFO) << "patch_size: " << patch_size;
    LOG(INFO) << "alpha: "      << alpha;
    LOG(INFO) << "overlap: "    << overlap;

    LOG(INFO) << "data dimensions";
    LOG(INFO) << "height: " << height;
    LOG(INFO) << "width: "  << width;

    auto cl_devs = get_platform_devs(0);

    // get the maximum possible tile_size, but make sure that it is not larger (only by patch_size - 2*overlap)
    // than the height or width of the image
    const int sub_image_size = std::min(goldstein::round_up(std::max(height + 2*overlap,
                                                                     width  + 2*overlap), patch_size - 2*overlap),
                                        goldstein::tile_size(cl_devs, patch_size, overlap));
    std::pair<int, int> tile_dims{sub_image_size, sub_image_size};

    goldstein::kernel_params kp {16};

    // prepare data
    insar_data total_image{ampl_master, ampl_slave, phase,
                           ref_filt, phase_filt, coh_filt,
                           (int) height, (int) width};

    // filtering
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    auto tm = map_filter_tiles(goldstein::filter_sub_image,
                               total_image, // same image can be used as input and output
                               total_image,
                               kp,
                               tile_dims,
                               overlap,
                               patch_size,
                               overlap,
                               alpha);

    timings::print(tm);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end-start;
    VLOG(0) << "filtering ran for " << duration.count() << " secs" << std::endl;

    memcpy(ref_filt,   total_image.ref_filt(), total_image.height()*total_image.width()*sizeof(float));
    memcpy(phase_filt, total_image.phase_filt(), total_image.height()*total_image.width()*sizeof(float));
    memcpy(coh_filt,   total_image.coh_filt(), total_image.height()*total_image.width()*sizeof(float));

    return 0;
}
