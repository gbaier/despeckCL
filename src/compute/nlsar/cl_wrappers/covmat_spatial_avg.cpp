#include "covmat_spatial_avg.h"

const int nlsar::covmat_spatial_avg::get_output_block_size(const int scale_size)
{
    return block_size - scale_size + 1;
}

void nlsar::covmat_spatial_avg::run(cl::CommandQueue cmd_queue,
                                    cl::Buffer covmat_in,
                                    cl::Buffer covmat_out,
                                    const int dimension,
                                    const int height_overlap,
                                    const int width_overlap,
                                    const int scale_size,
                                    const int scale_size_max)
{
    const int output_block_size = get_output_block_size(scale_size);

    kernel.setArg(0, covmat_in);
    kernel.setArg(1, covmat_out);
    kernel.setArg(2, dimension);
    kernel.setArg(3, height_overlap);
    kernel.setArg(4, width_overlap);
    kernel.setArg(5, scale_size);
    kernel.setArg(6, scale_size_max);
    kernel.setArg(7, cl::Local(block_size*block_size*sizeof(float)));

    cl::NDRange global_size {(size_t) block_size*( (height_overlap - 1)/output_block_size + 1), \
                             (size_t) block_size*( (width_overlap  - 1)/output_block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
