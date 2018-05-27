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
#include <string.h> // for memcpy
#include <vector>
#include <string>

#include "insar_data.h"
#include "tile_iterator.h"
#include "nlinsar_sub_image.h"
#include "sub_images.h"
#include "insarsim_simu.h"
#include "logging.h"
#include "timings.h"
#include "map_filter_tiles.h"

#include "clcfg.h"
#include "cl_wrappers.h"

int despeckcl::nlinsar(float* ampl_master,
                       float* ampl_slave,
                       float* phase,
                       float* ref_filt,
                       float* phase_filt,
                       float* coh_filt,
                       const int height,
                       const int width,
                       const int search_window_size,
                       const int patch_size,
                       const int niter,
                       const int lmin,
                       std::vector<std::string> enabled_log_levels)
{
    timings::map tm_tot;
    logging_setup(enabled_log_levels);

    insar_data total_image{ampl_master, ampl_slave, phase,
                           ref_filt, phase_filt, coh_filt,
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
    std::pair<int, int> tile_dims {80, 80};

    
    // new build kernel interface
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> duration = end-start;
    start = std::chrono::system_clock::now();
    VLOG(0) << "Building kernels";
    nlinsar::cl_wrappers nl_routines_base(context, search_window_size, patch_size, 16);
    end = std::chrono::system_clock::now();
    duration = end-start;
    VLOG(0) << "Time it took to build all kernels: " << duration.count() << "secs";
    
    // filtering
    start = std::chrono::system_clock::now();
    for(int n = 0; n<niter; n++) {
        LOG(INFO) << "Iteration " << n + 1 << " of " << niter;
        // deep copy is necessary
        insar_data total_image_temp {total_image.ampl_master(),
                                     total_image.ampl_slave(),
                                     total_image.phase(),
                                     total_image.ref_filt(),
                                     total_image.phase_filt(),
                                     total_image.coh_filt(),
                                     total_image.height,
                                     total_image.width};

        auto tm = map_filter_tiles(nlinsar::nlinsar_sub_image,
                                   total_image,
                                   total_image_temp,
                                   context,
                                   nl_routines_base,
                                   tile_dims,
                                   overlap,
                                   search_window_size, patch_size, lmin, h_para, T_para); // filter parameters

        total_image = std::move(total_image_temp);
        tm_tot = timings::join(tm, tm_tot);
    }
    timings::print(tm_tot);
    end = std::chrono::system_clock::now();
    duration = end-start;
    VLOG(0) << "filtering ran for " << duration.count() << " secs" << std::endl;

    memcpy(ref_filt,   total_image.ref_filt(), total_image.height*total_image.width*sizeof(float));
    memcpy(phase_filt, total_image.phase_filt(), total_image.height*total_image.width*sizeof(float));
    memcpy(coh_filt,   total_image.coh_filt(), total_image.height*total_image.width*sizeof(float));

    return 0;
}
