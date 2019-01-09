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

#include "routines.h"

timings::map nlsar::routines::get_pixel_similarities (cl::Context context,
                                                      cl::Buffer& covmat_rescaled,
                                                      cl::Buffer& device_pixel_similarities,
                                                      const int height_overlap,
                                                      const int width_overlap,
                                                      const int dimension,
                                                      const int nlooks,
                                                      const int search_window_size,
                                                      const int scale_size,
                                                      const int scale_size_max,
                                                      cl_wrappers& nl_routines)
{
    timings::map tm;

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::CommandQueue cmd_queue{context, devices[0]};

    const int n_elem_overlap = height_overlap * width_overlap;

    cl::Buffer covmat_spatial_avg        {context, CL_MEM_READ_WRITE, 2*dimension * dimension * n_elem_overlap * sizeof(float), NULL, NULL};

    LOG(DEBUG) << "covmat_spatial_avg";
    tm["covmat_spatial_avg"] = nl_routines.covmat_spatial_avg_routine.timed_run(cmd_queue,
                                                                                covmat_rescaled,
                                                                                covmat_spatial_avg,
                                                                                dimension,
                                                                                height_overlap,
                                                                                width_overlap,
                                                                                scale_size,
                                                                                scale_size_max);

    if (dimension == 2) {
      LOG(DEBUG) << "covmat_pixel_similarities_2x2";
      tm["covmat_pixel_similarities"] =
          nl_routines.compute_pixel_similarities_2x2_routine.timed_run(
              cmd_queue,
              covmat_spatial_avg,
              device_pixel_similarities,
              height_overlap,
              width_overlap,
              dimension,
              nlooks,
              search_window_size);
    } else if (dimension == 3) {
      LOG(DEBUG) << "covmat_pixel_similarities_3x3";
      tm["covmat_pixel_similarities"] =
          nl_routines.compute_pixel_similarities_3x3_routine.timed_run(
              cmd_queue,
              covmat_spatial_avg,
              device_pixel_similarities,
              height_overlap,
              width_overlap,
              dimension,
              nlooks,
              search_window_size);
    } else {
      throw std::runtime_error(
          "currently only 2x2 or 3x3 covariance matrices are supported");
    }

    return tm;
}

timings::map nlsar::routines::get_weights (cl::Context context,
                                           cl::Buffer& pixel_similarities,
                                           cl::Buffer& weights,
                                           const int height_overlap,
                                           const int width_overlap,
                                           const int search_window_size,
                                           const int patch_size,
                                           const int patch_size_max,
                                           stats& parameter_stats,
                                           cl::Buffer& lut_dissims2relidx,
                                           cl::Buffer& lut_chi2cdf_inv,
                                           cl_wrappers& nl_routines)
{
    timings::map tm;

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::CommandQueue cmd_queue{context, devices[0]};

    const int wsh = (search_window_size - 1)/2;

    const int height_pix_symm = height_overlap - wsh;
    const int width_pix_symm  = width_overlap;

    const int height_pat_symm = height_pix_symm - patch_size_max + 1;
    const int width_pat_symm  = width_pix_symm  - patch_size_max + 1;

    // original dimension of the unpadded data
    const int height_ori = height_pat_symm - wsh;
    const int width_ori  = width_pat_symm - 2*wsh;

    cl::Buffer patch_similarities {context, CL_MEM_READ_WRITE, (search_window_size*wsh + wsh) * height_pat_symm * width_pat_symm * sizeof(float), NULL, NULL};
    cl::Buffer weights_symm       {context, CL_MEM_READ_WRITE, (search_window_size*wsh + wsh) * height_pat_symm * width_pat_symm * sizeof(float), NULL, NULL};

    LOG(DEBUG) << "covmat_patch_similarities";
    tm["covmat_patch_similarities"] = nl_routines.compute_patch_similarities_routine.timed_run(cmd_queue,
                                                                                               pixel_similarities,
                                                                                               patch_similarities,
                                                                                               height_pix_symm,
                                                                                               width_pix_symm,
                                                                                               search_window_size,
                                                                                               patch_size,
                                                                                               patch_size_max);

    LOG(DEBUG) << "compute_weights";
    tm["compute_weights"] = nl_routines.compute_weights_routine.timed_run(cmd_queue,
                                                                          patch_similarities,
                                                                          weights_symm,
                                                                          height_pat_symm,
                                                                          width_pat_symm,
                                                                          search_window_size,
                                                                          patch_size,
                                                                          lut_dissims2relidx,
                                                                          lut_chi2cdf_inv,
                                                                          parameter_stats.lut_size,
                                                                          parameter_stats.dissims_min,
                                                                          parameter_stats.dissims_max);

    LOG(DEBUG) << "copy_symm_weights";
    tm["copy_symm_weights"] = nl_routines.copy_symm_weights_routine.timed_run(cmd_queue,
                                                                              weights_symm,
                                                                              weights,
                                                                              height_ori,
                                                                              width_ori,
                                                                              search_window_size);

    // set weight for self similarity
    const cl_float self_weight = 1.0f;
    cmd_queue.enqueueFillBuffer(weights,
                                self_weight,
                                height_ori * width_ori * (search_window_size * wsh + wsh) * sizeof(float), //offset
                                height_ori * width_ori * sizeof(float),
                                NULL, NULL);

    return tm;
}

timings::map nlsar::routines::get_enls_nobias_and_alphas (cl::Context context,
                                                          cl::Buffer& device_weights,
                                                          cl::Buffer& device_covmat_ori,
                                                          cl::Buffer& device_enls_nobias,
                                                          cl::Buffer& device_alphas,
                                                          const int height_ori,
                                                          const int width_ori,
                                                          const int search_window_size,
                                                          const int patch_size_max,
                                                          const int scale_size_max,
                                                          const int nlooks,
                                                          const int dimension,
                                                          cl_wrappers& nl_routines)
{
    timings::map tm;

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::CommandQueue cmd_queue{context, devices[0]};

    const int n_elem_ori = height_ori * width_ori;

    cl::Buffer device_enl                {context, CL_MEM_READ_WRITE,             n_elem_ori * sizeof(float), NULL, NULL};
    cl::Buffer device_intensities_nl     {context, CL_MEM_READ_WRITE, dimension * n_elem_ori * sizeof(float), NULL, NULL};
    cl::Buffer device_weighted_variances {context, CL_MEM_READ_WRITE, dimension * n_elem_ori * sizeof(float), NULL, NULL};
    cl::Buffer device_wsums              {context, CL_MEM_READ_WRITE,             n_elem_ori * sizeof(float), NULL, NULL};

    LOG(DEBUG) << "compute_nl_statistics";
    tm["compute_nl_statistics"] = nl_routines.compute_nl_statistics_routine.timed_run(cmd_queue, 
                                                                                      device_covmat_ori,
                                                                                      device_weights,
                                                                                      device_intensities_nl,
                                                                                      device_weighted_variances,
                                                                                      device_wsums,
                                                                                      device_enl,
                                                                                      height_ori,
                                                                                      width_ori,
                                                                                      search_window_size,
                                                                                      patch_size_max,
                                                                                      scale_size_max);

    LOG(DEBUG) << "compute_alphas";
    tm["compute_alphas"] = nl_routines.compute_alphas_routine.timed_run(cmd_queue, 
                                                                        device_intensities_nl,
                                                                        device_weighted_variances,
                                                                        device_alphas,
                                                                        height_ori,
                                                                        width_ori,
                                                                        dimension,
                                                                        nlooks);

    LOG(DEBUG) << "compute_enls_nobias";
    tm["compte_enls_nobias"] = nl_routines.compute_enls_nobias_routine.timed_run(cmd_queue, 
                                                                                 device_enl,
                                                                                 device_alphas,
                                                                                 device_wsums,
                                                                                 device_enls_nobias,
                                                                                 height_ori,
                                                                                 width_ori);

    return tm;
}
