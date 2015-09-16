#include "routines.h"

cl::Buffer nlsar::routines::get_weights (cl::Buffer& pixel_similarities,
                                         cl::Context context,
                                         const int height_sim,
                                         const int width_sim,
                                         const int search_window_size,
                                         const int patch_size,
                                         const int patch_size_max,
                                         stats& parameter_stats,
                                         cl::Buffer& lut_dissims2relidx,
                                         cl::Buffer& lut_chi2cdf_inv,
                                         cl_wrappers& nl_routines)
{
    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::CommandQueue cmd_queue{context, devices[0]};

    const int wsh = (search_window_size - 1)/2;

    // original dimension of the unpadded data
    const int height_ori = height_sim - patch_size + 1;
    const int width_ori  = width_sim  - patch_size + 1;
    const int n_elem_ori = height_ori * width_ori;

    cl::Buffer patch_similarities {context, CL_MEM_READ_WRITE, search_window_size * search_window_size * n_elem_ori * sizeof(float), NULL, NULL};
    cl::Buffer weights            {context, CL_MEM_READ_WRITE, search_window_size * search_window_size * n_elem_ori * sizeof(float), NULL, NULL};

    nl_routines.compute_patch_similarities_routine.timed_run(cmd_queue,
                                                             pixel_similarities,
                                                             patch_similarities,
                                                             height_sim,
                                                             width_sim,
                                                             search_window_size,
                                                             patch_size,
                                                             patch_size_max);

    nl_routines.compute_weights_routine.timed_run(cmd_queue,
                                                  patch_similarities,
                                                  weights,
                                                  height_ori,
                                                  width_ori,
                                                  search_window_size,
                                                  patch_size,
                                                  lut_dissims2relidx,
                                                  lut_chi2cdf_inv,
                                                  parameter_stats.lut_size,
                                                  parameter_stats.dissims_min,
                                                  parameter_stats.dissims_max);

    // set weight for self similarity
    const cl_int self_weight = 1;
    cmd_queue.enqueueFillBuffer(weights,
                                self_weight,
                                height_ori * width_ori * (search_window_size * wsh + wsh) * sizeof(float), //offset
                                height_ori * width_ori * sizeof(float),
                                NULL, NULL);

    return weights;
}
