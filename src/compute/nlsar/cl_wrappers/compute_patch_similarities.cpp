#include "compute_patch_similarities.h"

nlsar::compute_patch_similarities::compute_patch_similarities(cl::Context context,
                                                              const size_t block_size_x,
                                                              const size_t block_size_y,
                                                              const int steps_row,
                                                              const int steps_col) : context(context),
                                                                                     block_size_x(block_size_x),
                                                                                     block_size_y(block_size_y),
                                                                                     steps_row(steps_row),
                                                                                     steps_col(steps_col)
{
    std::string build_opts = return_build_options(block_size_x, block_size_y, steps_row, steps_col);

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);

    LOG(ERROR) << "building prog";
    cl::Program program{context, kernel_source_row_pass};
    try {
        program.build(devices, build_opts.c_str());
    } catch (cl::Error error) {
        LOG(ERROR) << error.what() << "(" << error.err() << ")";
        std::string build_log;
        program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &build_log);
        LOG(ERROR) << build_log;
        std::terminate();
    }

    this->program_row_pass = build_program(build_opts, kernel_source_row_pass);
    this->kernel_row_pass  = build_kernel(this->program_row_pass, "patch_similarities_row_pass");

    this->program_col_pass = build_program(build_opts, kernel_source_col_pass);
    this->kernel_col_pass  = build_kernel(this->program_col_pass, "patch_similarities_col_pass");
}


nlsar::compute_patch_similarities::compute_patch_similarities(const compute_patch_similarities& other) : context(other.context),
                                                                                                         block_size_x(other.block_size_x),
                                                                                                         block_size_y(other.block_size_y),
                                                                                                         steps_row(other.steps_row),
                                                                                                         steps_col(other.steps_col)
{
    this->program_row_pass = other.program_row_pass;
    this->program_col_pass = other.program_col_pass;
    this->kernel_row_pass  = build_kernel(this->program_row_pass, routine_name);
    this->kernel_col_pass  = build_kernel(this->program_col_pass, routine_name);
}

std::string nlsar::compute_patch_similarities::return_build_options(const int block_size_x,
                                                                    const int block_size_y,
                                                                    const int steps_row,
                                                                    const int steps_col)
{
    std::ostringstream out;
    out << " -D BLOCK_SIZE_X=" << block_size_x << " -D BLOCK_SIZE_Y=" << block_size_y << " -D STEPS_ROW=" << steps_row << " -D STEPS_COL=" << steps_col;
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
    const int width_ori  = width_sim  - patch_size_max + 1;
    const int height_ori  = height_sim  - patch_size_max + 1;
    const int offset = (patch_size_max - patch_size) / 2;

    cl::NDRange local_size  {block_size_x, block_size_y, 1};

    cl::Buffer intermed_row_conv {context, CL_MEM_READ_WRITE, search_window_size * search_window_size * (height_sim - 2*offset) * width_ori * sizeof(float), NULL, NULL};

    kernel_row_pass.setArg(0, pixel_similarities);
    kernel_row_pass.setArg(1, intermed_row_conv);
    kernel_row_pass.setArg(2, height_sim);
    kernel_row_pass.setArg(3, width_sim);
    kernel_row_pass.setArg(4, patch_size);
    kernel_row_pass.setArg(5, patch_size_max);
    // +2 is for the left and right halo.
    // This code only works if the half of the patch size is smaller than block_size_x
    kernel_row_pass.setArg(6, cl::Local((block_size_x*(steps_row+2)) * block_size_y * sizeof(float)));

    cl::NDRange global_size_row {(size_t) block_size_x*( (width_ori - 1)/(block_size_x*steps_row) + 1), \
                                 (size_t) block_size_y*( (height_sim  - 2*offset - 1)/(block_size_y) + 1), \
                                 (size_t) search_window_size*search_window_size };

    cmd_queue.enqueueNDRangeKernel(kernel_row_pass, cl::NullRange, global_size_row, local_size, NULL, NULL);



    kernel_col_pass.setArg(0, intermed_row_conv);
    kernel_col_pass.setArg(1, patch_similarities);
    kernel_col_pass.setArg(2, height_sim);
    kernel_col_pass.setArg(3, width_sim);
    kernel_col_pass.setArg(4, patch_size);
    kernel_col_pass.setArg(5, patch_size_max);
    // +4 is for the top and bottom halo.
    // +4 instead of +2 as for the row filter since the block_size in y direction is smaller
    kernel_col_pass.setArg(6, cl::Local((block_size_y*(steps_col+4)) * block_size_x * sizeof(float)));

    cl::NDRange global_size_col {(size_t) block_size_x*( (width_ori - 1)/(block_size_x) + 1), \
                                 (size_t) block_size_y*( (height_ori  - 1)/(block_size_y*steps_col) + 1), \
                                 (size_t) search_window_size*search_window_size };

    cmd_queue.enqueueNDRangeKernel(kernel_col_pass, cl::NullRange, global_size_col, local_size, NULL, NULL);
}
