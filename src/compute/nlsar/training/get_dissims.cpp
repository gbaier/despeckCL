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

#include "get_dissims.h"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "covmat_create.h"
#include "covmat_rescale.h"
#include "covmat_spatial_avg.h"
#include "patches.h"
#include "combinations.h"

#include "clcfg.h"

std::vector<float> nlsar::training::get_dissims(cl::Context context,
                                                nlsar::cl_wrappers nlsar_cl_wrappers,
                                                const covmat_data& training_data,
                                                const int patch_size,
                                                const int scale_size)
{
    const int nlooks = 1;

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);

    cl::CommandQueue cmd_queue{context, devices[0]};

    // overlapped dimension, large enough to include the complete padded data to compute the similarities;
    // also includes overlap for spatial averaging
    const int height_overlap_avg = training_data.height();
    const int width_overlap_avg  = training_data.width();
    const int n_elem_overlap_avg = height_overlap_avg * width_overlap_avg;

    // overlapped dimension, large enough to include the complete padded data to compute the similarities;
    const int height_overlap = height_overlap_avg - scale_size + 1;
    const int width_overlap  = width_overlap_avg  - scale_size + 1;
    const int n_elem_overlap = height_overlap * width_overlap;

    LOG(DEBUG) << "patch_size: " << patch_size;
    LOG(DEBUG) << "scale_size: " << scale_size;
    LOG(DEBUG) << "height_overlap_avg: " << height_overlap_avg;
    LOG(DEBUG) << "width_overlap_avg: " << width_overlap_avg;
    LOG(DEBUG) << "height_overlap: " << height_overlap;
    LOG(DEBUG) << "width_overlap: " << width_overlap;

    //***************************************************************************
    //
    // global buffers used by the kernels to exchange data
    //
    //***************************************************************************

    cl::Buffer device_covmat{context,
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             2 * training_data.dim() * training_data.dim() *
                                 n_elem_overlap_avg * sizeof(float),
                             training_data.data(),
                             NULL};

    cl::Buffer device_covmat_spatial_avg{context,
                                         CL_MEM_READ_WRITE,
                                         2 * training_data.dim() *
                                             training_data.dim() *
                                             n_elem_overlap * sizeof(float),
                                         NULL,
                                         NULL};

    LOG(DEBUG) << "covmat_rescale";
    nlsar_cl_wrappers.covmat_rescale_routine.timed_run(cmd_queue,
                                     device_covmat,
                                     training_data.dim(),
                                     nlooks,
                                     height_overlap_avg,
                                     width_overlap_avg);

    LOG(DEBUG) << "covmat_spatial_avg";
    nlsar_cl_wrappers.covmat_spatial_avg_routine.timed_run(cmd_queue,
                                         device_covmat,
                                         device_covmat_spatial_avg,
                                         training_data.dim(),
                                         height_overlap,
                                         width_overlap,
                                         scale_size,
                                         scale_size);

    std::vector<float> covmat_spatial_avg (2 * training_data.dim() * training_data.dim() * n_elem_overlap);
    cmd_queue.enqueueReadBuffer(device_covmat_spatial_avg, CL_TRUE, 0, covmat_spatial_avg.size() * sizeof(float), covmat_spatial_avg.data(), NULL, NULL);

    LOG(DEBUG) << "setting up training data";
    covmat_data covmat_spatial_avg_c{covmat_spatial_avg.data(),
                                     covmat_spatial_avg.data(),
                                     height_overlap,
                                     width_overlap,
                                     training_data.dim()};

    LOG(DEBUG) << "get all patches inside training data";
    std::vector<covmat_data> all_patches = training::get_all_patches(covmat_spatial_avg_c, patch_size);

    LOG(DEBUG) << "computing all patch dissimilarity combinations";
    return get_all_dissim_combs(std::move(all_patches));
}
