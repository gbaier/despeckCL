#include "nlsar_filter_sub_image.h"

#include "nlsar.h"
#include "best_params.h"
#include "best_weights_copy.h"
#include "best_alpha_copy.h"

#include <iostream>

#include "timings.h"
#include "routines.h"

timings::map nlsar::filter_sub_image(cl::Context context,
                                     cl_wrappers nl_routines,
                                     insar_data& sub_insar_data,
                                     const int search_window_size,
                                     const int dimension,
                                     std::map<params, stats> &dissim_stats)
{
    timings::map tm;

    std::vector<params> parameters;
    for(auto keyval : dissim_stats) {
        parameters.push_back(keyval.first);
    }

    std::vector<int> patch_sizes;
    for(auto param : parameters) {
        patch_sizes.push_back(param.patch_size);
    }
    const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());

    std::vector<int> scale_sizes;
    for(auto param : parameters) {
        scale_sizes.push_back(param.scale_size);
    }
    const int scale_size_max = *std::max_element(scale_sizes.begin(), scale_sizes.end());

    const int psh = (patch_size_max - 1)/2;
    const int wsh = (search_window_size - 1)/2;
    const int overlap = wsh+psh;

    const int nlooks = 1;
    

    // overlapped dimension, large enough to include the complete padded data to compute the similarities;
    // also includes overlap for spatial averaging
    const int height_overlap_avg = sub_insar_data.height;
    const int width_overlap_avg  = sub_insar_data.width;
    const int n_elem_overlap_avg = height_overlap_avg * width_overlap_avg;

    // overlapped dimension, large enough to include the complete padded data to compute the similarities;
    const int height_overlap = height_overlap_avg - scale_size_max + 1;
    const int width_overlap  = width_overlap_avg  - scale_size_max + 1;
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

    cl::Buffer covmat_ori      {context, CL_MEM_READ_WRITE, 2*dimension * dimension * n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer covmat_rescaled {context, CL_MEM_READ_WRITE, 2*dimension * dimension * n_elem_overlap_avg * sizeof(float), NULL, NULL};

    std::map<params, cl::Buffer> device_lut_dissims2relidx;
    std::map<params, cl::Buffer> device_lut_chi2cdf_inv;

    std::map<params, std::vector<float>> weights;
    std::map<params, std::vector<float>> enls_nobias;
    std::map<params, std::vector<float>> alphas;

    for(params parameter : parameters) {
        const stats* para_stats = &dissim_stats.find(parameter)->second;
        const int lut_size = para_stats->lut_size;
        device_lut_dissims2relidx [parameter] = cl::Buffer {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                            lut_size * sizeof(float), (void*) para_stats->dissims2relidx.data(), NULL};
        device_lut_chi2cdf_inv    [parameter] = cl::Buffer {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, \
                                                            lut_size * sizeof(float), (void*) para_stats->chi2cdf_inv.data(), NULL};

        weights     [parameter] = std::vector<float> (search_window_size * search_window_size * n_elem_ori);
        enls_nobias [parameter] = std::vector<float> (n_elem_ori);
        alphas      [parameter] = std::vector<float> (n_elem_ori);
    }

    cl::Buffer device_best_weights {context, CL_MEM_READ_WRITE, search_window_size * search_window_size * n_elem_ori * sizeof(float), NULL, NULL};
    cl::Buffer device_best_alphas  {context, CL_MEM_READ_WRITE,                                           n_elem_ori * sizeof(float), NULL, NULL};

    cl::Buffer device_ampl_filt   {context, CL_MEM_READ_WRITE,                             n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer device_dphase_filt {context, CL_MEM_READ_WRITE,                             n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer device_coh_filt    {context, CL_MEM_READ_WRITE,                             n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer covmat_filt        {context, CL_MEM_READ_WRITE, 2 * dimension * dimension * n_elem_overlap_avg * sizeof(float), NULL, NULL};

    //***************************************************************************
    //
    // command queue
    //
    //***************************************************************************

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::CommandQueue cmd_queue{context, devices[0]};

    //***************************************************************************
    //
    // executing routines and kernels
    //
    //***************************************************************************
    LOG(DEBUG) << "covmat_create";
    tm["covmat_create"] = nl_routines.covmat_create_routine.timed_run(cmd_queue,
                                                                      device_ampl_master,
                                                                      device_ampl_slave,
                                                                      device_dphase,
                                                                      covmat_ori,
                                                                      height_overlap_avg,
                                                                      width_overlap_avg);

    cmd_queue.enqueueCopyBuffer(covmat_ori, covmat_rescaled, 0, 0, 2*dimension * dimension * n_elem_overlap_avg * sizeof(float), NULL, NULL);

    LOG(DEBUG) << "covmat_rescale";
    tm["covmat_rescale"] = nl_routines.covmat_rescale_routine.timed_run(cmd_queue,
                                                                        covmat_rescaled,
                                                                        dimension,
                                                                        nlooks,
                                                                        height_overlap_avg,
                                                                        width_overlap_avg);

    for(int scale_size : scale_sizes) {
        cl::Buffer device_pixel_similarities {context, CL_MEM_READ_WRITE, search_window_size * search_window_size * n_elem_sim * sizeof(float), NULL, NULL};
        timings::map tm_pixel_similarities = routines::get_pixel_similarities(context,
                                                                              covmat_rescaled,
                                                                              device_pixel_similarities,
                                                                              height_overlap,
                                                                              width_overlap,
                                                                              dimension,
                                                                              search_window_size,
                                                                              scale_size,
                                                                              scale_size_max,
                                                                              nl_routines);

        tm = timings::join(tm, tm_pixel_similarities);

        for(int patch_size : patch_sizes) {
            params parameter{patch_size, scale_size};

            cl::Buffer device_weights {context, CL_MEM_READ_WRITE, search_window_size * search_window_size * n_elem_ori * sizeof(float), NULL, NULL};
            timings::map tm_weights = routines::get_weights(context,
                                                            device_pixel_similarities,
                                                            device_weights,
                                                            height_sim,
                                                            width_sim,
                                                            search_window_size,
                                                            patch_size,
                                                            patch_size_max,
                                                            dissim_stats.find(parameter)->second,
                                                            device_lut_dissims2relidx[parameter],
                                                            device_lut_chi2cdf_inv[parameter],
                                                            nl_routines);
            tm = timings::join(tm, tm_weights);


            cmd_queue.enqueueReadBuffer(device_weights, CL_TRUE, 0, n_elem_ori*search_window_size*search_window_size*sizeof(float), weights[parameter].data(), NULL, NULL);

            std::pair<cl::Buffer, cl::Buffer> device_enls_alphas = routines::get_enls_nobias_and_alphas (context,
                                                                                                         device_weights,
                                                                                                         covmat_ori,
                                                                                                         height_ori,
                                                                                                         width_ori,
                                                                                                         search_window_size,
                                                                                                         patch_size_max,
                                                                                                         scale_size_max,
                                                                                                         nlooks,
                                                                                                         dimension,
                                                                                                         nl_routines);

            cmd_queue.enqueueReadBuffer(device_enls_alphas.first,  CL_TRUE, 0, n_elem_ori * sizeof(float), enls_nobias[parameter].data(), NULL, NULL);
            cmd_queue.enqueueReadBuffer(device_enls_alphas.second, CL_TRUE, 0, n_elem_ori * sizeof(float),      alphas[parameter].data(), NULL, NULL);
        }
    }

    std::vector<params> best_parameters = best_params(enls_nobias, height_ori, width_ori);
    std::vector<float> best_weights = best_weights_copy(weights, best_parameters, height_ori, width_ori, search_window_size);
    std::vector<float> best_alphas  = best_alpha_copy  (alphas,  best_parameters, height_ori, width_ori);

    cmd_queue.enqueueWriteBuffer(device_best_weights, CL_TRUE, 0, search_window_size * search_window_size * n_elem_ori * sizeof(float), best_weights.data());
    cmd_queue.enqueueWriteBuffer(device_best_alphas,  CL_TRUE, 0,                                           n_elem_ori * sizeof(float),  best_alphas.data());

    LOG(DEBUG) << "weighted_means";
    nl_routines.weighted_means_routine.timed_run(cmd_queue,
                                                  covmat_ori,
                                                  covmat_filt,
                                                  device_best_weights,
                                                  device_best_alphas,
                                                  height_ori,
                                                  width_ori,
                                                  search_window_size,
                                                  patch_size_max,
                                                  scale_size_max);

    nl_routines.covmat_decompose_routine.timed_run(cmd_queue,
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

    return tm;
}
