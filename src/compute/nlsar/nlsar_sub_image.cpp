#include "nlsar_sub_image.h"

#include "nlsar.h"

#include <iostream>

#include "../compute_env.h"

// opencl kernel wrappers
#include "compute_pixel_similarities_2x2.h"
#include "covmat_create.h"
#include "covmat_rescale.h"
#include "covmat_spatial_avg.h"
#include "weighted_means.h"

int nlsar_sub_image(cl::Context context,
                    nlsar_routines nl_routines,
                    insar_data& sub_insar_data,
                    const int search_window_size,
                    const int patch_size,
                    const int dimension,
                    stats dissim_stats)
{
    const int psh = (patch_size - 1)/2;
    const int wsh = (search_window_size - 1)/2;
    const int overlap = wsh+psh;

    const int window_width = 3;
    const int nlooks = 1;

    // overlapped dimension, large enough to include the complete padded data to compute the similarities;
    // also includes overlap for spatial averaging
    const int height_overlap_avg = sub_insar_data.height;
    const int width_overlap_avg  = sub_insar_data.width;
    const int n_elem_overlap_avg = height_overlap_avg * width_overlap_avg;

    // overlapped dimension, large enough to include the complete padded data to compute the similarities;
    const int height_overlap = height_overlap_avg - window_width + 1;
    const int width_overlap  = width_overlap_avg  - window_width + 1;
    const int n_elem_overlap = height_overlap * width_overlap;

    // dimension of the precomputed patch similarity values
    const int height_sim = height_overlap - search_window_size + 1;
    const int width_sim  = width_overlap  - search_window_size + 1;
    const int n_elem_sim = height_sim * width_sim;

    // original dimension of the unpadded data
    const int height_ori = height_overlap - 2*overlap;
    const int width_ori  = width_overlap - 2*overlap;
    const int n_elem_ori = height_ori * width_ori;

    //***************************************************************************
    //
    // global buffers used by the kernels to exchange data
    //
    //***************************************************************************

    LOG(DEBUG) << "allocating buffers on device";
    cl::Buffer device_ampl_master {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_elem_overlap_avg * sizeof(float), sub_insar_data.a1, NULL};
    cl::Buffer device_ampl_slave  {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_elem_overlap_avg * sizeof(float), sub_insar_data.a2, NULL};
    cl::Buffer device_dphase      {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_elem_overlap_avg * sizeof(float), sub_insar_data.dp, NULL};

    cl::Buffer covmat_ori                 {context, CL_MEM_READ_WRITE, 2*dimension * dimension * n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer covmat_rescaled            {context, CL_MEM_READ_WRITE, 2*dimension * dimension * n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer covmat_spatial_avg         {context, CL_MEM_READ_WRITE, 2*dimension * dimension * n_elem_overlap     * sizeof(float), NULL, NULL};


    cl::Buffer device_pixel_similarities  {context, CL_MEM_READ_WRITE, search_window_size * search_window_size * n_elem_sim * sizeof(float), NULL, NULL};
    cl::Buffer device_patch_similarities  {context, CL_MEM_READ_WRITE, search_window_size * search_window_size * n_elem_ori * sizeof(float), NULL, NULL};
    cl::Buffer device_weights             {context, CL_MEM_READ_WRITE, search_window_size * search_window_size * n_elem_ori * sizeof(float), NULL, NULL};

    std::vector<float> patch_similarities (search_window_size * search_window_size * n_elem_ori); 
    std::vector<float> weights            (search_window_size * search_window_size * n_elem_ori); 

    cl::Buffer device_ampl_filt   {context, CL_MEM_READ_WRITE,                             n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer device_dphase_filt {context, CL_MEM_READ_WRITE,                             n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer device_coh_filt    {context, CL_MEM_READ_WRITE,                             n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer covmat_filt        {context, CL_MEM_READ_WRITE, 2 * dimension * dimension * n_elem_overlap_avg * sizeof(float), NULL, NULL};

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
    LOG(DEBUG) << "covmat_create";
    nl_routines.covmat_create_routine->timed_run(cmd_queue,
                                                 device_ampl_master,
                                                 device_ampl_slave,
                                                 device_dphase,
                                                 covmat_ori,
                                                 height_overlap_avg,
                                                 width_overlap_avg);

    cmd_queue.enqueueCopyBuffer(covmat_ori, covmat_rescaled, 0, 0, dimension * dimension * n_elem_overlap * sizeof(float), NULL, NULL);

    LOG(DEBUG) << "covmat_rescale";
    nl_routines.covmat_rescale_routine->timed_run(cmd_queue,
                                                  covmat_rescaled,
                                                  dimension,
                                                  nlooks,
                                                  height_overlap_avg,
                                                  width_overlap_avg);

    LOG(DEBUG) << "covmat_spatial_avg";
    nl_routines.covmat_spatial_avg_routine->timed_run(cmd_queue,
                                                      covmat_rescaled,
                                                      covmat_spatial_avg,
                                                      dimension,
                                                      height_overlap,
                                                      width_overlap);

    LOG(DEBUG) << "covmat_pixel_similarities";
    nl_routines.compute_pixel_similarities_2x2_routine->timed_run(cmd_queue,
                                                                  covmat_spatial_avg,
                                                                  device_pixel_similarities,
                                                                  height_overlap,
                                                                  width_overlap,
                                                                  search_window_size);

    LOG(DEBUG) << "covmat_patch_similarities";
    nl_routines.compute_patch_similarities_routine->timed_run(cmd_queue,
                                                              device_pixel_similarities,
                                                              device_patch_similarities,
                                                              height_sim,
                                                              width_sim,
                                                              search_window_size,
                                                              patch_size);

    cmd_queue.enqueueReadBuffer(device_patch_similarities, CL_TRUE, 0,
                                n_elem_ori * search_window_size * search_window_size * sizeof(float), patch_similarities.data(), NULL, NULL);

    std::transform(patch_similarities.begin(), patch_similarities.end(), weights.begin(), [&dissim_stats] (float dissim) {return dissim_stats.weight(dissim);});

    cmd_queue.enqueueWriteBuffer(device_weights, CL_TRUE, 0,
                                n_elem_ori * search_window_size * search_window_size * sizeof(float), weights.data());


    LOG(DEBUG) << "weighted_means";
    nl_routines.weighted_means_routine->timed_run(cmd_queue,
                                                  covmat_ori,
                                                  covmat_filt,
                                                  device_weights,
                                                  height_ori,
                                                  width_ori,
                                                  search_window_size,
                                                  patch_size,
                                                  window_width);

    nl_routines.covmat_decompose_routine->timed_run(cmd_queue,
                                                    covmat_filt,
                                                    device_ampl_filt,
                                                    device_dphase_filt,
                                                    device_coh_filt,
                                                    height_overlap_avg,
                                                    width_overlap_avg);
    
    //***************************************************************************
    //
    // copying back result and clean up
    //
    //***************************************************************************
    LOG(DEBUG) << "copying sub result";
    cmd_queue.enqueueReadBuffer(device_ampl_filt,   CL_TRUE, 0, n_elem_overlap_avg*sizeof(float), sub_insar_data.amp_filt, NULL, NULL);
    cmd_queue.enqueueReadBuffer(device_dphase_filt, CL_TRUE, 0, n_elem_overlap_avg*sizeof(float), sub_insar_data.phi_filt, NULL, NULL);
    cmd_queue.enqueueReadBuffer(device_coh_filt,    CL_TRUE, 0, n_elem_overlap_avg*sizeof(float), sub_insar_data.coh_filt, NULL, NULL);

    return 0;
}
