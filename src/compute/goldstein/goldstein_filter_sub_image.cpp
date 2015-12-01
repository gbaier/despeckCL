#include "goldstein_filter_sub_image.h"
#include <stdlib.h>

/* No need to explicitely include the OpenCL headers */
#include <clFFT.h>

#include "goldstein_patch_ft.h"

timings::map goldstein::filter_sub_image(cl::Context context,
                                         cl_wrappers gs_routines,
                                         insar_data& sub_insar_data,
                                         const unsigned int patch_size,
                                         const unsigned int overlap,
                                         const float alpha)
{
    timings::map tm;

    const unsigned int height = sub_insar_data.height;
    const unsigned int width  = sub_insar_data.width;

    const unsigned int n_patches_width  = width  / (patch_size-2*overlap);
    const unsigned int n_patches_height = height / (patch_size-2*overlap);

    const unsigned int height_tiles = patch_size * n_patches_height;
    const unsigned int width_tiles  = patch_size * n_patches_width;

    LOG(DEBUG) << "sub_image";
    LOG(DEBUG) << "height:           " << height;
    LOG(DEBUG) << "width:            " << width;
    LOG(DEBUG) << "n_patches_height: " << n_patches_height;
    LOG(DEBUG) << "n_patches_width:  " << n_patches_width;
    LOG(DEBUG) << "height_tiles:     " << height_tiles;
    LOG(DEBUG) << "width_tiles:      " << width_tiles;
    LOG(DEBUG) << "patch_size:       " << patch_size;
    LOG(DEBUG) << "overlap:          " << overlap;
    LOG(DEBUG) << "alpha:            " << alpha;

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::CommandQueue cmd_queue{context, devices[0]};

    cl::Buffer dev_ampl_master {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), sub_insar_data.a1, NULL};
    cl::Buffer dev_ampl_slave  {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), sub_insar_data.a2, NULL};
    cl::Buffer dev_dphase      {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height * width * sizeof(float), sub_insar_data.dp, NULL};

    // io buffers
    cl::Buffer dev_interf_real {context, CL_MEM_READ_WRITE, height * width * sizeof(float), NULL, NULL};
    cl::Buffer dev_interf_imag {context, CL_MEM_READ_WRITE, height * width * sizeof(float), NULL, NULL};

    cl::Buffer dev_interf_tiles_real {context, CL_MEM_READ_WRITE, height_tiles * width_tiles * sizeof(float), NULL, NULL};
    cl::Buffer dev_interf_tiles_imag {context, CL_MEM_READ_WRITE, height_tiles * width_tiles * sizeof(float), NULL, NULL};

    /****************************************************************************
     *
     * clFFT setup
     *
     ******************************************************************/

    clfftPlanHandle plan_handle;
    clfftDim dim = CLFFT_2D;
    size_t cl_lengths[2] = {patch_size, patch_size};
    size_t in_strides [2] = {1, width_tiles};
    size_t out_strides[2] = {1, width_tiles};


    /* Setup clFFT. */
    clfftSetupData fft_setup;
    clfftInitSetupData(&fft_setup);
    clfftSetup(&fft_setup);

    /* Create a default plan for a complex FFT. */
    clfftCreateDefaultPlan(&plan_handle, context(), dim, cl_lengths);

    /* Set plan parameters. */
    clfftSetPlanPrecision(plan_handle, CLFFT_SINGLE);
    clfftSetLayout(plan_handle, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR); // separate arrays for real and complex data
    clfftSetResultLocation(plan_handle, CLFFT_INPLACE);
    clfftSetPlanInStride  (plan_handle, dim, in_strides);
    clfftSetPlanOutStride (plan_handle, dim, out_strides);
    clfftSetPlanBatchSize (plan_handle, width_tiles/patch_size);
    clfftSetPlanDistance  (plan_handle, patch_size, patch_size);

    /* Bake the plan. */
    clfftBakePlan(plan_handle, 1, &cmd_queue(), NULL, NULL);


    /****************************************************************************
     *
     * Processing
     *
     ******************************************************************/

    LOG(DEBUG) << "raw_interferogram";
    gs_routines.raw_interferogram_routine.run(cmd_queue,
                                              dev_ampl_master,
                                              dev_ampl_slave,
                                              dev_dphase,
                                              dev_interf_real,
                                              dev_interf_imag,
                                              height,
                                              width);

    LOG(DEBUG) << "patches_unpack";
    gs_routines.patches_unpack_routine.run(cmd_queue,
                                           dev_interf_real,
                                           dev_interf_imag,
                                           dev_interf_tiles_real,
                                           dev_interf_tiles_imag,
                                           height_tiles,
                                           width_tiles,
                                           patch_size,
                                           overlap);

    LOG(DEBUG) << "patch_ft forward";
    goldstein_patch_ft(cmd_queue,
                       plan_handle,
                       dev_interf_tiles_real,
                       dev_interf_tiles_imag,
                       height_tiles,
                       width_tiles,
                       patch_size,
                       CLFFT_FORWARD);

    LOG(DEBUG) << "weighted_multiply";
    gs_routines.weighted_multiply_routine.run(cmd_queue,
                                              dev_interf_tiles_real,
                                              dev_interf_tiles_imag,
                                              height_tiles,
                                              width_tiles,
                                              alpha);

    LOG(DEBUG) << "patch_ft backward";
    goldstein_patch_ft(cmd_queue,
                       plan_handle,
                       dev_interf_tiles_real,
                       dev_interf_tiles_imag,
                       height_tiles,
                       width_tiles,
                       patch_size,
                       CLFFT_BACKWARD);

    LOG(DEBUG) << "patches_pack";
    gs_routines.patches_pack_routine.run(cmd_queue,
                                         dev_interf_tiles_real,
                                         dev_interf_tiles_imag,
                                         dev_interf_real,
                                         dev_interf_imag,
                                         height_tiles,
                                         width_tiles,
                                         patch_size,
                                         overlap);

    LOG(DEBUG) << "slc2real";
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

    return tm;
}
