#include "raw_interferogram.h"

void goldstein::raw_interferogram::run(cl::CommandQueue cmd_queue,
                                       cl::Buffer ampl_master,
                                       cl::Buffer ampl_slave,
                                       cl::Buffer dphase,
                                       cl::Buffer interf_real,
                                       cl::Buffer interf_imag,
                                       const int height,
                                       const int width)
{
    kernel.setArg( 0, ampl_master);
    kernel.setArg( 1, ampl_slave);
    kernel.setArg( 2, dphase);
    kernel.setArg( 3, interf_real);
    kernel.setArg( 4, interf_imag);
    kernel.setArg( 5, height);
    kernel.setArg( 6, width);

    cl::NDRange global_size {(size_t) block_size*( (width  - 1)/block_size + 1),
                             (size_t) block_size*( (height - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
