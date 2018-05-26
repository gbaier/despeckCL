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

#include "boxcar_sub_image.h"

#include <iostream>

#include "timings.h"

timings::map boxcar_sub_image(cl::Context context,
                              boxcar_wrapper boxcar_routine,
                              insar_data& sub_insar_data)
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> duration;
    timings::map tm;
    start = std::chrono::system_clock::now();

    // overlapped dimension, large enough to include the complete padded data to compute the similarities
    const int height_overlap = sub_insar_data.height;
    const int width_overlap  = sub_insar_data.width;
    const int n_elem_overlap = height_overlap*width_overlap;

    //***************************************************************************
    //
    // global buffers used by the kernels to exchange data
    //
    //***************************************************************************

    cl::Buffer device_raw_a1                  {context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, n_elem_overlap*sizeof(float), sub_insar_data.a1.get(), NULL};
    cl::Buffer device_raw_a2                  {context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, n_elem_overlap*sizeof(float), sub_insar_data.a2.get(), NULL};
    cl::Buffer device_raw_dp                  {context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, n_elem_overlap*sizeof(float), sub_insar_data.dp.get(), NULL};

    cl::Buffer device_ref_filt                {context, CL_MEM_READ_WRITE, n_elem_overlap*sizeof(float), NULL, NULL};
    cl::Buffer device_phi_filt                {context, CL_MEM_READ_WRITE, n_elem_overlap*sizeof(float), NULL, NULL};
    cl::Buffer device_coh_filt                {context, CL_MEM_READ_WRITE, n_elem_overlap*sizeof(float), NULL, NULL};

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::CommandQueue cmd_queue{context, devices[0]};

    end = std::chrono::system_clock::now();
    duration = end-start;
    tm["setup"] = duration.count();

    //***************************************************************************
    //
    // executing routines and kernels
    //
    //***************************************************************************
    tm["boxcar"] = boxcar_routine.timed_run(cmd_queue,
                                            device_raw_a1, device_raw_a2, device_raw_dp,
                                            device_ref_filt, device_phi_filt, device_coh_filt,
                                            height_overlap, width_overlap);
    
    //***************************************************************************
    //
    // copying back result and clean up
    //
    //***************************************************************************

    start = std::chrono::system_clock::now();
    cmd_queue.enqueueReadBuffer(device_ref_filt, CL_TRUE, 0, n_elem_overlap*sizeof(float), sub_insar_data.ref_filt.get(), NULL, NULL);
    cmd_queue.enqueueReadBuffer(device_phi_filt, CL_TRUE, 0, n_elem_overlap*sizeof(float), sub_insar_data.phi_filt.get(), NULL, NULL);
    cmd_queue.enqueueReadBuffer(device_coh_filt, CL_TRUE, 0, n_elem_overlap*sizeof(float), sub_insar_data.coh_filt.get(), NULL, NULL);
    end = std::chrono::system_clock::now();
    duration = end-start;
    tm["copy_sub_result"] = duration.count();

    return tm;
}
