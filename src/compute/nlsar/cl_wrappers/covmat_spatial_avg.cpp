#include "covmat_spatial_avg.h"

#include <cmath>
#include <numeric>

int nlsar::covmat_spatial_avg::get_output_block_size(const int scale_size)
{
    return block_size - scale_size + 1;
}

std::vector<float> nlsar::covmat_spatial_avg::gen_gauss(const int scale_size)
{
    constexpr const float pi = std::atan(1)*4;
    std::vector<float> gauss;
    gauss.reserve(scale_size*scale_size);
    const int ssh = (scale_size - 1)/2;

    for(int x = -ssh; x <= ssh; x++) {
        for(int y = -ssh; y <= ssh; y++) {
            gauss.push_back(std::exp(-pi*(x*x + y*y)/std::pow(ssh+0.5f, 2.0f)));
        }
    }
    const float K = std::accumulate(gauss.begin(), gauss.end(), 0.0f);
    std::for_each(gauss.begin(), gauss.end(), [K] (float& el) { el = el/K; });

    return gauss;
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
    std::vector<float> gauss {gen_gauss(scale_size)};

    cl::Buffer dev_gauss {context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scale_size * scale_size * sizeof(float), gauss.data(), NULL};

    kernel.setArg(0, covmat_in);
    kernel.setArg(1, covmat_out);
    kernel.setArg(2, dimension);
    kernel.setArg(3, height_overlap);
    kernel.setArg(4, width_overlap);
    kernel.setArg(5, scale_size);
    kernel.setArg(6, scale_size_max);
    kernel.setArg(7, cl::Local(block_size*block_size*sizeof(float)));
    kernel.setArg(8, dev_gauss);

    cl::NDRange global_size {(size_t) block_size*( (height_overlap - 1)/output_block_size + 1), \
                             (size_t) block_size*( (width_overlap  - 1)/output_block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
