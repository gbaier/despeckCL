#include "nlinsar_sub_image.h"

#include <iostream>

#include "../compute_env.h"

// opencl kernel wrappers
#include "precompute_similarities_1st_pass.h"
#include "precompute_similarities_2nd_pass.h"
#include "precompute_patch_similarities.h"
#include "compute_weights.h"
#include "compute_number_of_looks.h"
#include "transpose.h"
#include "precompute_filter_values.h"
#include "compute_insar.h"

#include "smoothing.h"

int nlinsar_sub_image(cl::Context context,
                      nlinsar::routines nl_routines,
                      insar_data& sub_insar_data,
                      const int search_window_size,
                      const int patch_size,
                      const int lmin,
                      const float h_para,
                      const float T_para)
{
    const int psh = (patch_size - 1)/2;
    const int wsh = (search_window_size - 1)/2;
    const int overlap = wsh+psh;

    // overlapped dimension, large enough to include the complete padded data to compute the similarities
    const int height_overlap = sub_insar_data.height;
    const int width_overlap  = sub_insar_data.width;
    const int n_elem_overlap = height_overlap*width_overlap;

    // dimension of the precomputed filtering values 'a' and 'x' in the final filtering/estimation process 
    const int height_sws = height_overlap - patch_size + 1;
    const int width_sws  = width_overlap  - patch_size + 1;
    const int n_elem_sws = height_sws*width_sws;

    // dimension of the precomputed patch similarity values
    const int height_sim = height_overlap - search_window_size + 1;
    const int width_sim  = width_overlap  - search_window_size + 1;
    const int n_elem_sim = search_window_size * search_window_size * height_sim * width_sim;

    // original dimension of the unpadded data
    const int height_ori = height_overlap - 2*overlap;
    const int width_ori  = width_overlap - 2*overlap;
    const int n_elem_ori = search_window_size * search_window_size * height_ori * width_ori;

    //***************************************************************************
    //
    // global buffers used by the kernels to exchange data
    //
    //***************************************************************************

    cl::Buffer device_raw_a1                  {context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, n_elem_overlap*sizeof(float), sub_insar_data.a1, NULL};
    cl::Buffer device_raw_a2                  {context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, n_elem_overlap*sizeof(float), sub_insar_data.a2, NULL};
    cl::Buffer device_raw_dp                  {context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, n_elem_overlap*sizeof(float), sub_insar_data.dp, NULL};

    cl::Buffer device_amp_filt                {context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_elem_overlap*sizeof(float), sub_insar_data.amp_filt, NULL};
    cl::Buffer device_phi_filt                {context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_elem_overlap*sizeof(float), sub_insar_data.phi_filt, NULL};
    cl::Buffer device_coh_filt                {context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_elem_overlap*sizeof(float), sub_insar_data.coh_filt, NULL};

    cl::Buffer device_filter_values_a         {context, CL_MEM_READ_WRITE, n_elem_sws*sizeof(float), NULL, NULL};
    cl::Buffer device_filter_values_x_real    {context, CL_MEM_READ_WRITE, n_elem_sws*sizeof(float), NULL, NULL};
    cl::Buffer device_filter_values_x_imag    {context, CL_MEM_READ_WRITE, n_elem_sws*sizeof(float), NULL, NULL};

    cl::Buffer device_pixel_similarities      {context, CL_MEM_READ_WRITE, n_elem_sim*sizeof(float), NULL, NULL};
    cl::Buffer device_pixel_kullback_leiblers {context, CL_MEM_READ_WRITE, n_elem_sim*sizeof(float), NULL, NULL};

    cl::Buffer device_patch_similarities      {context, CL_MEM_READ_WRITE, n_elem_ori*sizeof(float), NULL, NULL};
    cl::Buffer device_patch_kullback_leiblers {context, CL_MEM_READ_WRITE, n_elem_ori*sizeof(float), NULL, NULL};
    cl::Buffer device_weights                 {context, CL_MEM_READ_WRITE, n_elem_ori*sizeof(float), NULL, NULL};

    cl::Buffer device_number_of_looks         {context, CL_MEM_READ_WRITE, height_ori*width_ori*sizeof(float), NULL, NULL};

    // smoothing for a guaranteed minimum number of looks is done on the CPU,
    // since sorting is extremely slow on the GPU for this problem size.

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::CommandQueue cmd_queue{context, devices[0]};

    //***************************************************************************
    //
    // executing routines and kernels
    //
    //***************************************************************************
    nl_routines.precompute_similarities_1st_pass_routine->timed_run(cmd_queue,
                                                                    device_raw_a1, device_raw_a2, device_raw_dp,
                                                                    device_amp_filt, device_phi_filt, device_coh_filt,
                                                                    height_overlap, width_overlap,
                                                                    search_window_size,
                                                                    device_pixel_similarities, device_pixel_kullback_leiblers);

    nl_routines.precompute_similarities_2nd_pass_routine->timed_run(cmd_queue,
                                                                    height_overlap, width_overlap,
                                                                    search_window_size,
                                                                    device_pixel_similarities, device_pixel_kullback_leiblers);

    nl_routines.precompute_patch_similarities_routine->timed_run(cmd_queue,
                                                                 device_pixel_similarities, device_pixel_kullback_leiblers,
                                                                 height_sim, width_sim,
                                                                 search_window_size, patch_size,
                                                                 device_patch_similarities, device_patch_kullback_leiblers);

    nl_routines.compute_weights_routine->timed_run(cmd_queue,
                                                   device_patch_similarities, device_patch_kullback_leiblers,
                                                   device_weights,
                                                   search_window_size * search_window_size * height_ori * width_ori,
                                                   h_para, T_para);

    nl_routines.compute_number_of_looks_routine->timed_run(cmd_queue,
                                                           device_weights,
                                                           device_number_of_looks,
                                                           height_ori, width_ori,
                                                           search_window_size);

    nl_routines.transpose_routine->timed_run(cmd_queue,
                                             device_weights,
                                             search_window_size * search_window_size,
                                             height_ori * width_ori);

    nl_routines.smoothing_routine->timed_run(cmd_queue,
                                             device_weights,
                                             device_number_of_looks,
                                             sub_insar_data.a1,
                                             height_ori, width_ori,
                                             search_window_size, patch_size, lmin);

    nl_routines.precompute_filter_values_routine->timed_run(cmd_queue,
                                                            device_raw_a1, device_raw_a2, device_raw_dp,
                                                            device_filter_values_a, device_filter_values_x_real, device_filter_values_x_imag,
                                                            height_overlap, width_overlap, patch_size);

    nl_routines.compute_insar_routine->timed_run(cmd_queue,
                                                 device_filter_values_a, device_filter_values_x_real, device_filter_values_x_imag, 
                                                 device_amp_filt, device_phi_filt, device_coh_filt,
                                                 height_overlap, width_overlap,
                                                 device_weights,
                                                 search_window_size, patch_size);

    
    //***************************************************************************
    //
    // copying back result and clean up
    //
    //***************************************************************************
    const int n_elem = (height_overlap) * (width_overlap);
    cmd_queue.enqueueReadBuffer(device_amp_filt, CL_TRUE, 0, n_elem*sizeof(float), sub_insar_data.amp_filt, NULL, NULL);
    cmd_queue.enqueueReadBuffer(device_phi_filt, CL_TRUE, 0, n_elem*sizeof(float), sub_insar_data.phi_filt, NULL, NULL);
    cmd_queue.enqueueReadBuffer(device_coh_filt, CL_TRUE, 0, n_elem*sizeof(float), sub_insar_data.coh_filt, NULL, NULL);

    return 0;
}
