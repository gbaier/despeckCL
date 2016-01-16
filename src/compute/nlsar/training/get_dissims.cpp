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
                                                const insar_data& sub_insar_data,
                                                const int patch_size,
                                                const int scale_size)
{
    const int dimension = 2;
    const int nlooks = 1;

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);

    cl::CommandQueue cmd_queue{context, devices[0]};

    // overlapped dimension, large enough to include the complete padded data to compute the similarities;
    // also includes overlap for spatial averaging
    const int height_overlap_avg = sub_insar_data.height;
    const int width_overlap_avg  = sub_insar_data.width;
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

    cl::Buffer device_ampl_master {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_elem_overlap_avg * sizeof(float), sub_insar_data.a1, NULL};
    cl::Buffer device_ampl_slave  {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_elem_overlap_avg * sizeof(float), sub_insar_data.a2, NULL};
    cl::Buffer device_dphase      {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_elem_overlap_avg * sizeof(float), sub_insar_data.dp, NULL};

    cl::Buffer device_covmat              {context, CL_MEM_READ_WRITE, 2 * dimension * dimension * n_elem_overlap_avg * sizeof(float), NULL, NULL};
    cl::Buffer device_covmat_spatial_avg  {context, CL_MEM_READ_WRITE, 2 * dimension * dimension * n_elem_overlap     * sizeof(float), NULL, NULL};

    LOG(DEBUG) << "covmat_create";
    nlsar_cl_wrappers.covmat_create_routine.timed_run(cmd_queue,
                                    device_ampl_master,
                                    device_ampl_slave,
                                    device_dphase,
                                    device_covmat,
                                    height_overlap_avg,
                                    width_overlap_avg);

    LOG(DEBUG) << "covmat_rescale";
    nlsar_cl_wrappers.covmat_rescale_routine.timed_run(cmd_queue,
                                     device_covmat,
                                     dimension,
                                     nlooks,
                                     height_overlap_avg,
                                     width_overlap_avg);

    LOG(DEBUG) << "covmat_spatial_avg";
    nlsar_cl_wrappers.covmat_spatial_avg_routine.timed_run(cmd_queue,
                                         device_covmat,
                                         device_covmat_spatial_avg,
                                         dimension,
                                         height_overlap,
                                         width_overlap,
                                         scale_size,
                                         scale_size);

    std::vector<float> covmat_spatial_avg (2 * dimension * dimension * n_elem_overlap);
    cmd_queue.enqueueReadBuffer(device_covmat_spatial_avg, CL_TRUE, 0, covmat_spatial_avg.size() * sizeof(float), covmat_spatial_avg.data(), NULL, NULL);

    LOG(DEBUG) << "setting up training data";
    training::data covmat_spatial_avg_c {covmat_spatial_avg.data(),
                                         (uint32_t) height_overlap,
                                         (uint32_t) width_overlap,
                                         dimension};

    LOG(DEBUG) << "get all patches inside training data";
    std::vector<training::data> all_patches = covmat_spatial_avg_c.get_all_patches(patch_size);

    LOG(DEBUG) << "computing all patch dissimilarity combinations";
    return get_all_dissim_combs(all_patches);
}
