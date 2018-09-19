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

#include "nlsar_filter_sub_image.h"

#include "parameters.h"
#include "best_params.h"
#include "best_weights_copy.h"
#include "best_alpha_copy.h"
#include "tile_size.h"

#include <iostream>

#include "timings.h"
#include "routines.h"
#include "conversion.h"


timings::map nlsar::filter_sub_image(const cl::Context& context,
                                     cl_wrappers& nl_routines,
                                     insar_data& sub_insar_data,
                                     const int search_window_size,
                                     const std::vector<int> patch_sizes,
                                     const std::vector<int> scale_sizes,
                                     std::map<params, stats> &dissim_stats)
{

    const buffer_sizes buf_sizes{sub_insar_data.height(),
                                 sub_insar_data.width(),
                                 sub_insar_data.dim(),
                                 search_window_size,
                                 patch_sizes,
                                 scale_sizes};

    cl::CommandQueue cmd_queue{context};

    cl::Buffer covmat_ori =
        nlsar::data_to_covmat(sub_insar_data,
                              context,
                              cmd_queue,
                              nl_routines.covmat_create_routine,
                              buf_sizes);
    cl::Buffer covmat_filt{
        context, CL_MEM_READ_WRITE, buf_sizes.io_covmat(), NULL, NULL};

    timings::map tm;
    try {
        tm = nlsar::filter_sub_image_gpu(context,
                nl_routines,
                covmat_ori,
                covmat_filt,
                sub_insar_data.height(),
                sub_insar_data.width(),
                sub_insar_data.dim(),
                search_window_size,
                patch_sizes,
                scale_sizes,
                dissim_stats);
    } catch (cl::Error &error) {
        LOG(ERROR) << error.what() << "(" << error.err() << ")";
        LOG(ERROR) << "ERR while filtering sub image";
        std::terminate();
    }

    covmat_to_data(covmat_filt,
                   sub_insar_data,
                   context,
                   cmd_queue,
                   nl_routines.covmat_decompose_routine,
                   buf_sizes);

    return tm;
}


timings::map nlsar::filter_sub_image(const cl::Context& context,
                                     cl_wrappers& nl_routines,
                                     covmat_data& sub_covmat_data,
                                     const int search_window_size,
                                     const std::vector<int> patch_sizes,
                                     const std::vector<int> scale_sizes,
                                     std::map<params, stats> &dissim_stats)
{
  const buffer_sizes buf_sizes{sub_covmat_data.height(),
                               sub_covmat_data.width(),
                               sub_covmat_data.dim(),
                               search_window_size,
                               patch_sizes,
                               scale_sizes};

  cl::CommandQueue cmd_queue{context};

  cl::Buffer covmat_ori{context,
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        buf_sizes.io_covmat(),
                        sub_covmat_data.covmat_raw(),
                        NULL};
  cl::Buffer covmat_filt{
      context, CL_MEM_READ_WRITE, buf_sizes.io_covmat(), NULL, NULL};

  timings::map tm;
  try {
      tm = nlsar::filter_sub_image_gpu(context,
              nl_routines,
              covmat_ori,
              covmat_filt,
              sub_covmat_data.height(),
              sub_covmat_data.width(),
              sub_covmat_data.dim(),
              search_window_size,
              patch_sizes,
              scale_sizes,
              dissim_stats);
  } catch (cl::Error &error) {
      LOG(ERROR) << error.what() << "(" << error.err() << ")";
      LOG(ERROR) << "ERR while filtering sub image";
      std::terminate();
  }

  cmd_queue.enqueueReadBuffer(covmat_filt,
          CL_TRUE,
          0,
          buf_sizes.io_covmat(),
          sub_covmat_data.covmat_filt(),
          NULL,
          NULL);

  return tm;
}

timings::map
nlsar::filter_sub_image_gpu(const cl::Context& context,
                            cl_wrappers& nl_routines,
                            cl::Buffer& covmat_ori,
                            cl::Buffer& covmat_filt,
                            const int height,
                            const int width,
                            const int dimensions,
                            const int search_window_size,
                            const std::vector<int> patch_sizes,
                            const std::vector<int> scale_sizes,
                            std::map<params, stats>& dissim_stats)
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> duration;
    timings::map tm;
    std::map<params, int> params2idx;
    start = std::chrono::system_clock::now();

    const buffer_sizes buf_sizes{height,
                                 width,
                                 dimensions,
                                 search_window_size,
                                 patch_sizes,
                                 scale_sizes};

    const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
    const int scale_size_max = *std::max_element(scale_sizes.begin(), scale_sizes.end());

    const int psh = (patch_size_max - 1)/2;
    const int wsh = (search_window_size - 1)/2;
    const int overlap = wsh+psh;

    const int nlooks = 1;

    // overlapped dimension, large enough to include the complete padded data to compute the similarities;
    // also includes overlap for spatial averaging
    const int height_overlap_avg = height;
    const int width_overlap_avg  = width;
    const int n_elem_overlap_avg = height_overlap_avg * width_overlap_avg;

    // overlapped dimension, large enough to include the complete padded data to compute the similarities;
    const int height_overlap = height_overlap_avg - scale_size_max + 1;
    const int width_overlap  = width_overlap_avg  - scale_size_max + 1;

    // original dimension of the unpadded data
    const int height_ori = height_overlap - 2*overlap;
    const int width_ori  = width_overlap - 2*overlap;
    const int n_elem_ori = height_ori * width_ori;

    //***************************************************************************
    //
    // command queue
    //
    //***************************************************************************

    cl::CommandQueue cmd_queue{context};
    cl::CommandQueue cmd_copy_queue{context};

    //***************************************************************************
    //
    // global buffers used by the kernels to exchange data
    //
    //***************************************************************************

    LOG(DEBUG) << "allocating buffers on device";

    cl::Buffer covmat_rescaled {context, CL_MEM_READ_WRITE, buf_sizes.io_covmat(), NULL, NULL};

    std::map<params, cl::Buffer> device_lut_dissims2relidx;
    std::map<params, cl::Buffer> device_lut_chi2cdf_inv;

    std::map<params, std::vector<float>> enls_nobias;
    std::map<params, std::vector<float>> alphas;

    cl::Buffer device_best_idxs   {context, CL_MEM_READ_WRITE, buf_sizes.best_idxs(), NULL, NULL};
    cl::Buffer device_all_weights {context, CL_MEM_READ_WRITE, buf_sizes.weights_all(), NULL, NULL};

    for(auto& paramsstats : dissim_stats) {
        params parameter = paramsstats.first;
        enls_nobias [parameter] = std::vector<float> (n_elem_ori);
        alphas      [parameter] = std::vector<float> (n_elem_ori);
    }

    cl::Buffer device_best_weights {context, CL_MEM_READ_WRITE, buf_sizes.weights(), NULL, NULL};
    cl::Buffer device_best_alphas  {context, CL_MEM_READ_WRITE, buf_sizes.alphas(), NULL, NULL};


    end = std::chrono::system_clock::now();
    duration = end-start;
    tm["setup"] = duration.count();

    //***************************************************************************
    //
    // executing routines and kernels
    //
    //***************************************************************************

    LOG(DEBUG) << "covmat_rescale";

    cmd_queue.enqueueCopyBuffer(covmat_ori, covmat_rescaled, 0, 0, 2*dimensions * dimensions * n_elem_overlap_avg * sizeof(float), NULL, NULL);
    tm["covmat_rescale"] = nl_routines.covmat_rescale_routine.timed_run(cmd_queue,
                                                                        covmat_rescaled,
                                                                        dimensions,
                                                                        nlooks,
                                                                        height_overlap_avg,
                                                                        width_overlap_avg);

    int idx = 0;
    for(int scale_size : scale_sizes) {
        cl::Buffer device_pixel_similarities {context, CL_MEM_READ_WRITE,
                                              buf_sizes.pixel_similarities(),
                                              NULL, NULL};
        timings::map tm_pixel_similarities = routines::get_pixel_similarities(context,
                                                                              covmat_rescaled,
                                                                              device_pixel_similarities,
                                                                              height_overlap,
                                                                              width_overlap,
                                                                              dimensions,
                                                                              nlooks,
                                                                              search_window_size,
                                                                              scale_size,
                                                                              scale_size_max,
                                                                              nl_routines);

        tm = timings::join(tm, tm_pixel_similarities);

        for(int patch_size : patch_sizes) {
            params parameter{patch_size, scale_size};
            params2idx[parameter] = idx;
            idx++;

            const stats para_stats = dissim_stats[parameter];
            const int lut_size     = para_stats.lut_size;

            cl::Buffer device_weights {context, CL_MEM_READ_WRITE, buf_sizes.weights(), NULL, NULL};

            cl::Buffer device_lut_dissims2relidx = cl::Buffer {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                                lut_size * sizeof(float), (void*) para_stats.quantilles.data(), NULL};
            cl::Buffer device_lut_chi2cdf_inv    = cl::Buffer {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                                lut_size * sizeof(float), (void*) para_stats.chi2cdf_inv.data(), NULL};

            timings::map tm_weights = routines::get_weights(context,
                                                            device_pixel_similarities,
                                                            device_weights,
                                                            height_overlap,
                                                            width_overlap,
                                                            search_window_size,
                                                            patch_size,
                                                            patch_size_max,
                                                            dissim_stats.find(parameter)->second,
                                                            device_lut_dissims2relidx,
                                                            device_lut_chi2cdf_inv,
                                                            nl_routines);
            tm = timings::join(tm, tm_weights);
            cmd_queue.enqueueCopyBuffer(device_weights, device_all_weights,
                                        0, (idx-1)*buf_sizes.weights(),
                                        buf_sizes.weights(),
                                        NULL, NULL);

            cl::Buffer device_alphas      {context, CL_MEM_READ_WRITE, buf_sizes.alphas(), NULL, NULL};
            cl::Buffer device_enls_nobias {context, CL_MEM_READ_WRITE, buf_sizes.equivalent_number_of_looks(), NULL, NULL};
            timings::map tm_enls_nobias_and_alphas = routines::get_enls_nobias_and_alphas (context,
                                                                                           device_weights,
                                                                                           covmat_ori,
                                                                                           device_enls_nobias,
                                                                                           device_alphas,
                                                                                           height_ori,
                                                                                           width_ori,
                                                                                           search_window_size,
                                                                                           patch_size_max,
                                                                                           scale_size_max,
                                                                                           nlooks,
                                                                                           dimensions,
                                                                                           nl_routines);

            tm = timings::join(tm, tm_enls_nobias_and_alphas);

            start = std::chrono::system_clock::now();
            cmd_copy_queue.enqueueReadBuffer(device_enls_nobias,  CL_TRUE, 0, buf_sizes.equivalent_number_of_looks(), enls_nobias[parameter].data(), NULL, NULL);
            cmd_copy_queue.finish(); // without the finish the program gets stuck, at least on and AMD RX480
            cmd_copy_queue.enqueueReadBuffer(device_alphas,       CL_TRUE, 0, buf_sizes.alphas(), alphas[parameter].data(), NULL, NULL);
            end = std::chrono::system_clock::now();
            duration = end-start;

            tm["copy_dev2host"] = duration.count();
        }
    }
    try {
        cmd_copy_queue.finish();
    } catch (cl::Error &error) {
        LOG(ERROR) << error.what() << "(" << error.err() << ")";
        LOG(ERROR) << "ERR while calling finish() member function of command copy queue";
        std::terminate();
    }

    LOG(DEBUG) << "get best params";
    std::vector<params> best_parameters = get_best_params(enls_nobias);

    std::vector<int> best_idxs (best_parameters.size());

    std::transform(best_parameters.begin(), best_parameters.end(), best_idxs.begin(), [&params2idx] (params p) { return params2idx[p]; });

    LOG(DEBUG) << "copy best weights/alphas";
    start = std::chrono::system_clock::now();
    std::vector<float> best_alphas  = best_alpha_copy  (alphas,  best_parameters, height_ori, width_ori);

    cmd_queue.enqueueWriteBuffer(device_best_idxs,    CL_TRUE, 0, buf_sizes.best_idxs(), best_idxs.data());
    cmd_queue.enqueueWriteBuffer(device_best_alphas,  CL_TRUE, 0, buf_sizes.alphas(), best_alphas.data());

    end = std::chrono::system_clock::now();
    duration = end-start;
    tm["copy_best_weights/alphas"] = duration.count();

    LOG(DEBUG) << "copy_best_weights";
    tm["copy_best_weights"] = nl_routines.copy_best_weights_routine.timed_run(cmd_queue,
                                                                              device_all_weights,
                                                                              device_best_idxs,
                                                                              device_best_weights,
                                                                              height_ori,
                                                                              width_ori,
                                                                              search_window_size);

    LOG(DEBUG) << "weighted_means";
    tm["weighted_means"] = nl_routines.weighted_means_routine.timed_run(cmd_queue,
                                                                        covmat_ori,
                                                                        covmat_filt,
                                                                        device_best_weights,
                                                                        device_best_alphas,
                                                                        height_ori,
                                                                        width_ori,
                                                                        search_window_size,
                                                                        patch_size_max,
                                                                        scale_size_max);

    try {
        cmd_queue.finish();
    } catch (cl::Error &error) {
        LOG(ERROR) << error.what() << "(" << error.err() << ")";
        LOG(ERROR) << "ERR while calling finish() member function of command queue";
        std::terminate();
    }

    return tm;
}
