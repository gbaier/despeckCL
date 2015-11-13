#include "goldstein_filter_sub_image.h"
#include <stdlib.h>

/* No need to explicitely include the OpenCL headers */
#include <clFFT.h>

#include "goldstein_patch_ft.h"

timings::map goldstein::filter_sub_image(cl::Context context,
                                         cl_wrappers gs_routines,
                                         insar_data& sub_insar_data,
                                         const int patch_size,
                                         const int overlap,
                                         const float alpha)
{
    timings::map tm;

    const int height = sub_insar_data.height;
    const int width  = sub_insar_data.width;

    const int n_patches_width  = width  / (patch_size-2*overlap);
    const int n_patches_height = height / (patch_size-2*overlap);

    const int height_tiles = patch_size * n_patches_height;
    const int width_tiles  = patch_size * n_patches_width;

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::CommandQueue cmd_queue{context, devices[0]};

    cl::Buffer dev_ampl_master {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), sub_insar_data.a1, NULL};
    cl::Buffer dev_ampl_slave  {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), sub_insar_data.a2, NULL};
    cl::Buffer dev_dphase      {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), sub_insar_data.dp, NULL};

    // io buffers
    cl::Buffer dev_interf_real {context, CL_MEM_READ_WRITE, height * width * sizeof(float), NULL, NULL};
    cl::Buffer dev_interf_imag {context, CL_MEM_READ_WRITE, height * width * sizeof(float), NULL, NULL};

    cl::Buffer dev_interf_tiles_real_in {context, CL_MEM_READ_WRITE, height_tiles * width_tiles * sizeof(float), NULL, NULL};
    cl::Buffer dev_interf_tiles_imag_in {context, CL_MEM_READ_WRITE, height_tiles * width_tiles * sizeof(float), NULL, NULL};

    cl::Buffer dev_interf_tiles_real_out {context, CL_MEM_READ_WRITE, height_tiles * width_tiles * sizeof(float), NULL, NULL};
    cl::Buffer dev_interf_tiles_imag_out {context, CL_MEM_READ_WRITE, height_tiles * width_tiles * sizeof(float), NULL, NULL};

    /****************************************************************************
     *
     * clFFT setup
     *
     ******************************************************************/

    clfftPlanHandle plan_handle;
    clfftDim dim = CLFFT_2D;
    size_t cl_lengths[2] = {patch_size, patch_size};
    size_t in_strides [2] = {1, width};
    size_t out_strides[2] = {1, width};


    /* Setup clFFT. */
    clfftSetupData fft_setup;
    clfftInitSetupData(&fft_setup);
    clfftSetup(&fft_setup);

    /* Create a default plan for a complex FFT. */
    clfftCreateDefaultPlan(&plan_handle, context(), dim, cl_lengths);

    /* Set plan parameters. */
    clfftSetPlanPrecision(plan_handle, CLFFT_SINGLE);
    clfftSetLayout(plan_handle, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR); // separate arrays for real and complex data
    clfftSetResultLocation(plan_handle, CLFFT_OUTOFPLACE);
    clfftSetPlanInStride  (plan_handle, dim, in_strides);
    clfftSetPlanOutStride (plan_handle, dim, out_strides);
    clfftSetPlanBatchSize (plan_handle, width/patch_size);
    clfftSetPlanDistance  (plan_handle, patch_size, patch_size);

    /* Bake the plan. */
    clfftBakePlan(plan_handle, 1, &cmd_queue(), NULL, NULL);


    /****************************************************************************
     *
     * Processing
     *
     ******************************************************************/

    gs_routines.raw_interferogram_routine.run(cmd_queue,
                                              dev_ampl_master,
                                              dev_ampl_slave,
                                              dev_dphase,
                                              dev_interf_real,
                                              dev_interf_imag,
                                              height,
                                              width);

    gs_routines.patches_unpack_routine.run(cmd_queue,
                                           dev_interf_real,
                                           dev_interf_imag,
                                           dev_interf_tiles_real_in,
                                           dev_interf_tiles_imag_in,
                                           height_tiles,
                                           width_tiles,
                                           patch_size,
                                           overlap);

    goldstein_patch_ft(cmd_queue,
                       plan_handle,
                       dev_interf_tiles_real_in,
                       dev_interf_tiles_imag_in,
                       height_tiles,
                       width_tiles,
                       patch_size,
                       CLFFT_FORWARD);

    gs_routines.weighted_multiply_routine.run(cmd_queue,
                                              dev_interf_tiles_real_in,
                                              dev_interf_tiles_imag_in,
                                              dev_interf_tiles_real_out,
                                              dev_interf_tiles_imag_out,
                                              height_tiles,
                                              width_tiles,
                                              alpha);

    goldstein_patch_ft(cmd_queue,
                       plan_handle,
                       dev_interf_tiles_real_out,
                       dev_interf_tiles_imag_out,
                       height_tiles,
                       width_tiles,
                       patch_size,
                       CLFFT_BACKWARD);

    gs_routines.patches_pack_routine.run(cmd_queue,
                                         dev_interf_tiles_real_out,
                                         dev_interf_tiles_imag_out,
                                         dev_interf_real,
                                         dev_interf_imag,
                                         height_tiles,
                                         width_tiles,
                                         patch_size,
                                         overlap);

    gs_routines.slc2real_routine.run(cmd_queue,
                                     dev_interf_real,
                                     dev_interf_imag,
                                     dev_ampl_master,
                                     dev_dphase,
                                     height,
                                     width);

    /* Release the plan. */
    clfftDestroyPlan( &plan_handle );

    /* Release clFFT library. */
    clfftTeardown( );

    cmd_queue.enqueueReadBuffer(dev_ampl_master, CL_TRUE, 0, height * width * sizeof(float), sub_insar_data.amp_filt, NULL, NULL);
    cmd_queue.enqueueReadBuffer(dev_dphase,      CL_TRUE, 0, height * width * sizeof(float), sub_insar_data.phi_filt, NULL, NULL);
    cmd_queue.enqueueReadBuffer(dev_dphase,      CL_TRUE, 0, height * width * sizeof(float), sub_insar_data.coh_filt, NULL, NULL);

    return tm;
}
