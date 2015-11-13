#include "goldstein_patch_ft.h"

void goldstein_patch_ft(cl::CommandQueue &cmd_queue,
                        clfftPlanHandle &plan_handle,
                        cl::Buffer interf_patches_real,
                        cl::Buffer interf_patches_imag,
                        const int height,
                        const int width,
                        const int patch_size,
                        clfftDirection dir)
{
    const size_t offset = patch_size*width*sizeof(float);
    for(int h=0; h < height/patch_size; h++) {
        cl_mem buffers  [2] = {0, 0};
        cl_buffer_region rng {h*offset, offset};
        cl::Buffer dev_real_sub = interf_patches_real.createSubBuffer(CL_MEM_READ_WRITE,
                                                                 CL_BUFFER_CREATE_TYPE_REGION,
                                                                 &rng,
                                                                 NULL);
        cl::Buffer dev_imag_sub = interf_patches_imag.createSubBuffer(CL_MEM_READ_WRITE,
                                                                 CL_BUFFER_CREATE_TYPE_REGION,
                                                                 &rng,
                                                                 NULL);
        buffers[0] = dev_real_sub();
        buffers[1] = dev_imag_sub();
        clfftEnqueueTransform(plan_handle, dir, 1, &cmd_queue(), 0, NULL, NULL, buffers, NULL, NULL);
    }
}
