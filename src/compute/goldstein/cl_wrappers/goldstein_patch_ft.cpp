/* Copyright 2015, 2016 Gerald Baier
 *
 * This file is part of despeckCL.
 *
 * despeckCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * despeckCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with despeckCL. If not, see <http://www.gnu.org/licenses/>.
 */

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
