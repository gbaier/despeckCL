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

#include "boxcar_sub_image.h"
#include "insar_data.h"
#include "tile_iterator.h"
#include "tile.h"


void despeckcl::boxcar(float* ampl_master,
                       float* ampl_slave,
                       float* dphase,
                       float* ampl_filt,
                       float* dphase_filt,
                       float* coh_filt,
                       const int height,
                       const int width,
                       const int window_width,
                       std::vector<std::string> enabled_log_levels)
{
    logging_setup(enabled_log_levels);

    insar_data_shared total_image{ampl_master, ampl_slave, dphase,
                                  ampl_filt, dphase_filt, coh_filt,
                                  height, width};

    LOG(INFO) << "filter parameters";
    LOG(INFO) << "window width: " << window_width;

    LOG(INFO) << "data dimensions";
    LOG(INFO) << "height: " << height;
    LOG(INFO) << "width: " << width;

    const int overlap = (window_width - 1) / 2;

    // legacy opencl setup
    cl::Context context = opencl_setup();

    // filtering
    LOG(INFO) << "starting filtering";
    // the sub image size needs to be picked so that all buffers fit in the GPUs memory
    const int sub_image_size = 150;

    // new build kernel interface
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds = end-start;
    start = std::chrono::system_clock::now();
    VLOG(0) << "Building kernel";
    boxcar_wrapper boxcar_routine{16, context, window_width};
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    VLOG(0) << "Time it took to build the kernels: " << elapsed_seconds.count() << "secs";


    // timing
    double boxcar_timing = 0.0;
    

    LOG(INFO) << "starting filtering";
#pragma omp parallel shared(total_image)
{
#pragma omp master
    for( auto imgtile : tile_iterator(total_image, sub_image_size, overlap, overlap) ) {
#pragma omp task firstprivate(imgtile)
        {
        boxcar_sub_image(context, boxcar_routine, // opencl stuff
                         imgtile.get()); // data
        imgtile.write(total_image);
        }
    }
#pragma omp taskwait
}
    LOG(INFO) << "filtering done";

    VLOG(0) << "elapsed time for boxcar: " << boxcar_timing << " secs";

    return;
}
