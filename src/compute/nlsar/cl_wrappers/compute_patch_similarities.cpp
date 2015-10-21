#include "compute_patch_similarities.h"

nlsar::compute_patch_similarities::compute_patch_similarities(cl::Context context,
                                                              const size_t block_size_x,
                                                              const size_t block_size_y,
                                                              const int steps_row,
                                                              const int steps_col) : kernel_env<compute_patch_similarities>(block_size_x,
                                                                                                                            context,
                                                                                                                            this->return_build_options(block_size_x,
                                                                                                                                                       block_size_y,
                                                                                                                                                       steps_row,
                                                                                                                                                       steps_col)),
                                                                                     block_size_x(block_size_x),
                                                                                     block_size_y(block_size_y),
                                                                                     steps_row(steps_row),
                                                                                     steps_col(steps_col) {}


nlsar::compute_patch_similarities::compute_patch_similarities(const compute_patch_similarities& other) : kernel_env<compute_patch_similarities>(other),
                                                                                                         block_size_x(block_size_x),
                                                                                                         block_size_y(block_size_y),
                                                                                                         steps_row(steps_row),
                                                                                                         steps_col(steps_col) {}

std::string nlsar::compute_patch_similarities::return_build_options(const int block_size_x,
                                                                    const int block_size_y,
                                                                    const int steps_row,
                                                                    const int steps_col)
{
    std::ostringstream out;
    out << " -D BLOCK_SIZE_X=" << block_size_x << " -D BLOCK_SIZE_Y=" << block_size_y << " -D STEPS_ROW=" << steps_row << " -D STEPS_COLS" << steps_col;
    return return_default_build_opts() + out.str();
}

void nlsar::compute_patch_similarities::run(cl::CommandQueue cmd_queue,
                                            cl::Buffer pixel_similarities,
                                            cl::Buffer patch_similarities,
                                            const int height_sim,
                                            const int width_sim,
                                            const int search_window_size,
                                            const int patch_size,
                                            const int patch_size_max)
{
    const int output_block_size = get_output_block_size(patch_size);

    const int offset = (patch_size_max - patch_size) / 2;

    kernel.setArg(0, pixel_similarities);
    kernel.setArg(1, patch_similarities);
    kernel.setArg(2, height_sim);
    kernel.setArg(3, width_sim);
    kernel.setArg(4, patch_size);
    kernel.setArg(5, patch_size_max);
    kernel.setArg(6, cl::Local(block_size*block_size*sizeof(float)));
    kernel.setArg(7, cl::Local(block_size*output_block_size*sizeof(float)));

    cl::NDRange global_size {(size_t) block_size*( (height_sim - 2*offset - 1)/output_block_size + 1), \
                             (size_t) block_size*( (width_sim  - 2*offset - 1)/output_block_size + 1), \
                             (size_t) search_window_size*search_window_size };
    cl::NDRange local_size  {block_size, block_size, 1};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
}
