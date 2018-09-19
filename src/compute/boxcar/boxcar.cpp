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
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include "logging.h"

#include "timings.h"
#include "boxcar_sub_image.h"
#include "data.h"
#include "tile_iterator.h"
#include "tile.h"
#include "map_filter_tiles.h"

int
despeckcl::boxcar(float* ampl_master,
                  float* ampl_slave,
                  float* phase,
                  float* ref_filt,
                  float* phase_filt,
                  float* coh_filt,
                  const int height,
                  const int width,
                  const int window_width,
                  std::vector<std::string> enabled_log_levels)
{
    logging_setup(enabled_log_levels);

    insar_data total_image{ampl_master, ampl_slave, phase,
                           ref_filt, phase_filt, coh_filt,
                           height, width};

    LOG(INFO) << "filter parameters";
    LOG(INFO) << "window width: " << window_width;

    LOG(INFO) << "data dimensions";
    LOG(INFO) << "height: " << height;
    LOG(INFO) << "width: " << width;

    const int overlap = (window_width - 1) / 2;

    auto cl_devs = get_platform_devs(0);

    // filtering
    LOG(INFO) << "starting filtering";
    //
    // the sub image size needs to be picked so that all buffers fit in the GPUs memory
    std::pair<int, int> tile_dims {512, 512};

    boxcar::kernel_params kp{window_width, 16};

    // filtering
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    auto tm = map_filter_tiles(boxcar::boxcar_sub_image,
                               total_image, // same image can be used as input and output
                               total_image,
                               kp,
                               tile_dims,
                               overlap);

    timings::print(tm);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end-start;
    VLOG(0) << "filtering ran for " << duration.count() << " secs" << std::endl;

    memcpy(ref_filt,   total_image.ref_filt(), total_image.height()*total_image.width()*sizeof(float));
    memcpy(phase_filt, total_image.phase_filt(), total_image.height()*total_image.width()*sizeof(float));
    memcpy(coh_filt,   total_image.coh_filt(), total_image.height()*total_image.width()*sizeof(float));

    return 0;
}
