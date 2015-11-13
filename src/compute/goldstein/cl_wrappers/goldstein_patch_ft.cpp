#include "goldstein_patch_ft.h"

void goldstein_patch_ft(cl::CommandQueue &cmd_queue,
                        clfftPlanHandle &plan_handle,
                        cl::Buffer interf_patches_real_in,
                        cl::Buffer interf_patches_imag_in,
                        cl::Buffer interf_patches_real_out,
                        cl::Buffer interf_patches_imag_out,
                        const int height,
                        const int width,
                        const int patch_size)
{
    for(int h=0; h < height/patch_size; h++) {
        cl_mem buffers_in  [2] = {0, 0};
        cl_mem buffers_out [2] = {0, 0};
        size_t offset = patch_size*width*sizeof(float);
        cl_buffer_region rng {h*offset, offset};
        cl::Buffer dev_real_in_sub = interf_patches_real_in.createSubBuffer(CL_MEM_READ_WRITE,
                                                                 CL_BUFFER_CREATE_TYPE_REGION,
                                                                 &rng,
                                                                 NULL);
        cl::Buffer dev_imag_in_sub = interf_patches_imag_in.createSubBuffer(CL_MEM_READ_WRITE,
                                                                 CL_BUFFER_CREATE_TYPE_REGION,
                                                                 &rng,
                                                                 NULL);
        cl::Buffer dev_real_out_sub = interf_patches_real_out.createSubBuffer(CL_MEM_READ_WRITE,
                                                                   CL_BUFFER_CREATE_TYPE_REGION,
                                                                   &rng,
                                                                   NULL);
        cl::Buffer dev_imag_out_sub = interf_patches_imag_out.createSubBuffer(CL_MEM_READ_WRITE,
                                                                   CL_BUFFER_CREATE_TYPE_REGION,
                                                                   &rng,
                                                                   NULL);
        buffers_in[0] = dev_real_in_sub();
        buffers_in[1] = dev_imag_in_sub();
        buffers_out[0] = dev_real_out_sub();
        buffers_out[1] = dev_imag_out_sub();
        clfftEnqueueTransform(plan_handle, CLFFT_FORWARD, 1, &cmd_queue(), 0, NULL, NULL, buffers_in, buffers_out, NULL);
    }
}
