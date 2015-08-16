#include "compute_weights.h"

#include <stdio.h>

#include "compute_weights.h"

void compute_weights::run(cl::CommandQueue cmd_queue,
                          cl::Buffer patch_similarities,
                          cl::Buffer patch_kullback_leiblers,
                          cl::Buffer weights,
                          const int n_elems,
                          const float h,
                          const float T)
{
    kernel.setArg(0, patch_similarities);
    kernel.setArg(1, patch_kullback_leiblers);
    kernel.setArg(2, weights);
    kernel.setArg(3, n_elems);
    kernel.setArg(4, h);
    kernel.setArg(5, T);

    cl::NDRange global_size {block_size*((n_elems - 1) / block_size + 1)};
    cl::NDRange local_size  {block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);

    cmd_queue.finish();
}
