#include "compute_insar.h"

#include <stdio.h>
#include <sstream>

nlinsar::compute_insar::compute_insar(const size_t block_size,
                                      cl::Context context,
                                      const int search_window_size) : kernel_env<compute_insar>(block_size,
                                                                                                context,
                                                                                                return_build_options(block_size, search_window_size)),
                                                                      search_window_size(search_window_size) {}

nlinsar::compute_insar::compute_insar(const compute_insar& other) : kernel_env<compute_insar>(other),
                                                                    search_window_size(other.search_window_size) {}

std::string nlinsar::compute_insar::return_build_options(const int block_size, const int search_window_size)
{
    std::ostringstream out;
    out << " -D BLOCK_SIZE=" << block_size << " -D SEARCH_WINDOW_SIZE=" << search_window_size;
    return return_default_build_opts() + out.str();
}

void nlinsar::compute_insar::run(cl::CommandQueue cmd_queue,
                                 cl::Buffer device_filter_values_a,
                                 cl::Buffer device_filter_values_x_real,
                                 cl::Buffer device_filter_values_x_imag,
                                 cl::Buffer device_amp_filt,
                                 cl::Buffer device_phi_filt,
                                 cl::Buffer device_coh_filt,
                                 const int height_overlap,
                                 const int width_overlap,
                                 cl::Buffer device_weights,
                                 const int search_window_size,
                                 const int patch_size)
{
    const int height_ori = height_overlap - search_window_size - patch_size + 2;
    const int width_ori  = width_overlap  - search_window_size - patch_size + 2;

    kernel.setArg(0,  device_filter_values_a);
    kernel.setArg(1,  device_filter_values_x_real);
    kernel.setArg(2,  device_filter_values_x_imag);

    kernel.setArg(3,  device_amp_filt);
    kernel.setArg(4,  device_phi_filt);
    kernel.setArg(5,  device_coh_filt);
    
    kernel.setArg(6,  device_weights);
    kernel.setArg(7,  height_ori);
    kernel.setArg(8,  width_ori);
    
    kernel.setArg(9,  search_window_size);
    kernel.setArg(10, patch_size);

    cl::NDRange global_size {(size_t) block_size*((height_ori-1)/block_size+1), (size_t) block_size*((width_ori-1)/block_size+1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);

    cmd_queue.finish();
}
