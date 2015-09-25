#include "transpose.h"

#include <sstream>

nlinsar::transpose::transpose(const size_t block_size,
                     cl::Context context,
                     const size_t thread_size_row,
                     const size_t thread_size_col) : thread_size_row(thread_size_row),
                                                     thread_size_col(thread_size_col)
{
    kernel_env::block_size = block_size;
    kernel_env::context = context;
    build_program(return_build_options());
    build_kernel();
}

nlinsar::transpose::transpose(const transpose& other) : thread_size_row(other.thread_size_row),
                                               thread_size_col(other.thread_size_col)
{
    kernel_env::block_size = other.block_size;
    kernel_env::context = other.context;
    program = other.program;
    build_kernel();
}

std::string nlinsar::transpose::return_build_options(void)
{
    std::ostringstream out;
    out << " -D THREAD_SIZE_ROW=" << thread_size_row << " -D THREAD_SIZE_COL=" << thread_size_col;
    return kernel_env::return_build_options() + out.str();
}

void nlinsar::transpose::run(cl::CommandQueue cmd_queue,
                             cl::Buffer matrix,
                             const int height,
                             const int width)
{
    cl::Buffer matrix_trans{context, CL_MEM_READ_WRITE, height * width * sizeof(float), NULL, NULL};
    
    kernel.setArg(0, matrix);
    kernel.setArg(1, matrix_trans);
    kernel.setArg(2, height);
    kernel.setArg(3, width);

    cl::NDRange global_size {(size_t) block_size * ( (height - 1)/block_size + 1), (size_t) block_size*((width - 1)/block_size + 1)};
    cl::NDRange local_size  {thread_size_row, thread_size_col};

    cmd_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, NULL);
    cmd_queue.enqueueCopyBuffer(matrix_trans, matrix,
                                0, 0,
                                height*width*sizeof(float),
                                NULL, NULL);
}
