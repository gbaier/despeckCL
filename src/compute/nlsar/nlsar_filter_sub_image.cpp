#include "nlsar_filter_sub_image.h"

#include "parameters.h"
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
                                     const std::vector<int> patch_sizes,
                                     const std::vector<int> scale_sizes,
                                     const int dimensions,
                                     std::map<params, stats> &dissim_stats)
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> duration;
    timings::map tm;
    std::map<params, int> params2idx;
    start = std::chrono::system_clock::now();

    const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
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

    cl::Buffer covmat_ori      {context, CL_MEM_READ_WRITE, 2*dimensions * dimensions * n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer covmat_rescaled {context, CL_MEM_READ_WRITE, 2*dimensions * dimensions * n_elem_overlap_avg * sizeof(float), NULL, NULL};

    std::map<params, cl::Buffer> device_lut_dissims2relidx;
    std::map<params, cl::Buffer> device_lut_chi2cdf_inv;

    std::map<params, std::vector<float>> enls_nobias;
    std::map<params, std::vector<float>> alphas;

    cl::Buffer device_best_idxs   {context, CL_MEM_READ_WRITE,                                                               n_elem_ori * sizeof(float), NULL, NULL};
    cl::Buffer device_all_weights {context, CL_MEM_READ_WRITE, dissim_stats.size() * search_window_size * search_window_size * n_elem_ori * sizeof(float), NULL, NULL};

    for(auto& paramsstats : dissim_stats) {
        params parameter = paramsstats.first;
        const stats para_stats = paramsstats.second;
        const int lut_size = para_stats.lut_size;
        try {
            device_lut_dissims2relidx [parameter] = cl::Buffer {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                                lut_size * sizeof(float), (void*) para_stats.quantilles.data(), NULL};
            device_lut_chi2cdf_inv    [parameter] = cl::Buffer {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, \
                                                                lut_size * sizeof(float), (void*) para_stats.chi2cdf_inv.data(), NULL};
        } catch (cl::Error error) {
            LOG(ERROR) << "ERR copying LUT to device";
            LOG(ERROR) << error.what() << "(" << error.err() << ")";
            std::terminate();
        }
        enls_nobias [parameter] = std::vector<float> (n_elem_ori);
        alphas      [parameter] = std::vector<float> (n_elem_ori);
    }

    cl::Buffer device_best_weights {context, CL_MEM_READ_WRITE, search_window_size * search_window_size * n_elem_ori * sizeof(float), NULL, NULL};
    cl::Buffer device_best_alphas  {context, CL_MEM_READ_WRITE,                                           n_elem_ori * sizeof(float), NULL, NULL};

    cl::Buffer device_ampl_filt   {context, CL_MEM_READ_WRITE,                               n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer device_dphase_filt {context, CL_MEM_READ_WRITE,                               n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer device_coh_filt    {context, CL_MEM_READ_WRITE,                               n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer covmat_filt        {context, CL_MEM_READ_WRITE, 2 * dimensions * dimensions * n_elem_overlap_avg * sizeof(float), NULL, NULL};

    //***************************************************************************
    //
    // command queue
    //
    //***************************************************************************

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::CommandQueue cmd_queue{context, devices[0]};
    cl::CommandQueue cmd_copy_queue{context, devices[0]};

    end = std::chrono::system_clock::now();
    duration = end-start;
    tm["setup"] = duration.count();

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

    cmd_queue.enqueueCopyBuffer(covmat_ori, covmat_rescaled, 0, 0, 2*dimensions * dimensions * n_elem_overlap_avg * sizeof(float), NULL, NULL);

    LOG(DEBUG) << "covmat_rescale";
    tm["covmat_rescale"] = nl_routines.covmat_rescale_routine.timed_run(cmd_queue,
                                                                        covmat_rescaled,
                                                                        dimensions,
                                                                        nlooks,
                                                                        height_overlap_avg,
                                                                        width_overlap_avg);

    int idx = 0;
    for(int scale_size : scale_sizes) {
        cl::Buffer device_pixel_similarities {context, CL_MEM_READ_WRITE,
                                              (wsh+search_window_size*wsh) * (height_overlap-wsh) * width_overlap * sizeof(float),
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

            cl::Buffer device_weights {context, CL_MEM_READ_WRITE, search_window_size * search_window_size * n_elem_ori * sizeof(float), NULL, NULL};

            timings::map tm_weights = routines::get_weights(context,
                                                            device_pixel_similarities,
                                                            device_weights,
                                                            height_overlap,
                                                            width_overlap,
                                                            search_window_size,
                                                            patch_size,
                                                            patch_size_max,
                                                            dissim_stats.find(parameter)->second,
                                                            device_lut_dissims2relidx[parameter],
                                                            device_lut_chi2cdf_inv[parameter],
                                                            nl_routines);
            tm = timings::join(tm, tm_weights);
            cmd_queue.enqueueCopyBuffer(device_weights, device_all_weights,
                                        0, (idx-1)*n_elem_ori*search_window_size*search_window_size*sizeof(float),
                                        n_elem_ori*search_window_size*search_window_size*sizeof(float),
                                        NULL, NULL);

            cl::Buffer device_alphas      {context, CL_MEM_READ_WRITE, n_elem_ori * sizeof(float), NULL, NULL};
            cl::Buffer device_enls_nobias {context, CL_MEM_READ_WRITE, n_elem_ori * sizeof(float), NULL, NULL};
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
            cmd_copy_queue.enqueueReadBuffer(device_enls_nobias,  CL_FALSE, 0, n_elem_ori * sizeof(float), enls_nobias[parameter].data(), NULL, NULL);
            cmd_copy_queue.enqueueReadBuffer(device_alphas,       CL_FALSE, 0, n_elem_ori * sizeof(float),      alphas[parameter].data(), NULL, NULL);
            end = std::chrono::system_clock::now();
            duration = end-start;

            tm["copy_dev2host"] = duration.count();
        }
    }
    cmd_copy_queue.finish();

    LOG(DEBUG) << "get best params";
    std::vector<params> best_parameters;
    best_parameters.reserve(height_ori*width_ori);
    tm["get_best_params"] = get_best_params{}.timed_run(enls_nobias,
                                                        &best_parameters,
                                                        height_ori,
                                                        width_ori);

    std::vector<int> best_idxs (best_parameters.size());

    std::transform(best_parameters.begin(), best_parameters.end(), best_idxs.begin(), [&params2idx] (params p) { return params2idx[p]; });

    LOG(DEBUG) << "copy best weights/alphas";
    start = std::chrono::system_clock::now();
    std::vector<float> best_alphas  = best_alpha_copy  (alphas,  best_parameters, height_ori, width_ori);

    cmd_queue.enqueueWriteBuffer(device_best_idxs,    CL_TRUE, 0, n_elem_ori * sizeof(int),   best_idxs.data());
    cmd_queue.enqueueWriteBuffer(device_best_alphas,  CL_TRUE, 0, n_elem_ori * sizeof(float), best_alphas.data());

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

    LOG(DEBUG) << "covmat_decompose";
    tm["covmat_decompose"] = nl_routines.covmat_decompose_routine.timed_run(cmd_queue,
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
    start = std::chrono::system_clock::now();
    cmd_queue.enqueueReadBuffer(device_ampl_filt,   CL_TRUE, 0, n_elem_overlap_avg*sizeof(float), sub_insar_data.amp_filt, NULL, NULL);
    cmd_queue.enqueueReadBuffer(device_dphase_filt, CL_TRUE, 0, n_elem_overlap_avg*sizeof(float), sub_insar_data.phi_filt, NULL, NULL);
    cmd_queue.enqueueReadBuffer(device_coh_filt,    CL_TRUE, 0, n_elem_overlap_avg*sizeof(float), sub_insar_data.coh_filt, NULL, NULL);

    end = std::chrono::system_clock::now();
    duration = end-start;
    tm["copy_sub_result"] = duration.count();

    return tm;
}
