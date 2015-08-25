#include "get_dissims.h"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "covmat_create.h"
#include "covmat_rescale.h"
#include "covmat_spatial_avg.h"
#include "compute_pixel_similarities_2x2.h"
#include "compute_patch_similarities.h"

#include "clcfg.h"

std::vector<float> get_dissims(const insar_data& sub_insar_data,
                               const int patch_size,
                               const int window_width = 3)
{
    const int dimension = 2;
    const int nlooks = 1;

    cl::Context context = opencl_setup();

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);

    cl::CommandQueue cmd_queue{context, devices[0]};

    const int search_window_size = 5;
    const int psh = (patch_size - 1)/2;
    const int wsh = (search_window_size - 1)/2;
    const int overlap = wsh+psh;

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

    LOG(DEBUG) << "height_overlap_avg: " << height_overlap_avg;
    LOG(DEBUG) << "width_overlap_avg: " << width_overlap_avg;
    LOG(DEBUG) << "height_overlap: " << height_overlap;
    LOG(DEBUG) << "width_overlap: " << width_overlap;
    LOG(DEBUG) << "height_sim: " << height_sim;
    LOG(DEBUG) << "width_sim: " << width_sim;
    LOG(DEBUG) << "height_ori: " << height_ori;
    LOG(DEBUG) << "width_ori: " << width_ori;

    std::vector<float> patch_similarities (n_elem_ori * search_window_size * search_window_size);

    //***************************************************************************
    //
    // global buffers used by the kernels to exchange data
    //
    //***************************************************************************

    cl::Buffer device_ampl_master {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_elem_overlap_avg * sizeof(float), sub_insar_data.a1, NULL};
    cl::Buffer device_ampl_slave  {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_elem_overlap_avg * sizeof(float), sub_insar_data.a2, NULL};
    cl::Buffer device_dphase      {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_elem_overlap_avg * sizeof(float), sub_insar_data.dp, NULL};

    cl::Buffer device_covmat              {context, CL_MEM_READ_WRITE, 2 * dimension * dimension * n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer device_covmat_spatial_avg  {context, CL_MEM_READ_WRITE, 2 * dimension * dimension * n_elem_overlap     * sizeof(float), NULL, NULL};

    cl::Buffer device_pixel_similarities  {context, CL_MEM_READ_WRITE, search_window_size * search_window_size * n_elem_sim * sizeof(float), NULL, NULL};
    cl::Buffer device_patch_similarities  {context, CL_MEM_READ_WRITE, search_window_size * search_window_size * n_elem_ori * sizeof(float), NULL, NULL};

    covmat_create                  covmat_create_routine                  (16, context);
    covmat_rescale                 covmat_rescale_routine                 (16, context);
    covmat_spatial_avg             covmat_spatial_avg_routine             (16, context, window_width);
    compute_pixel_similarities_2x2 compute_pixel_similarities_2x2_routine (16, context);
    compute_patch_similarities     compute_patch_similarities_routine     (16, context, patch_size);

    covmat_create_routine.timed_run(cmd_queue,
                                    device_ampl_master,
                                    device_ampl_slave,
                                    device_dphase,
                                    device_covmat,
                                    height_overlap_avg,
                                    width_overlap_avg);

    covmat_rescale_routine.timed_run(cmd_queue,
                                     device_covmat,
                                     dimension,
                                     nlooks,
                                     height_overlap_avg,
                                     width_overlap_avg);

    covmat_spatial_avg_routine.timed_run(cmd_queue,
                                         device_covmat,
                                         device_covmat_spatial_avg,
                                         dimension,
                                         height_overlap,
                                         width_overlap);

    compute_pixel_similarities_2x2_routine.timed_run(cmd_queue,
                                                     device_covmat_spatial_avg,
                                                     device_pixel_similarities,
                                                     height_overlap,
                                                     width_overlap,
                                                     search_window_size);

    compute_patch_similarities_routine.timed_run(cmd_queue,
                                                 device_pixel_similarities,
                                                 device_patch_similarities,
                                                 height_sim,
                                                 width_sim,
                                                 search_window_size,
                                                 patch_size);

    cmd_queue.enqueueReadBuffer(device_patch_similarities, CL_TRUE, 0, patch_similarities.size() * sizeof(float), patch_similarities.data(), NULL, NULL);

    return patch_similarities;
}
