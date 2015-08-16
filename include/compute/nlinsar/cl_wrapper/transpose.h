#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include "clcfg.h"

class transpose : public kernel_env<transpose>
{
    public:

        static constexpr const char* routine_name {"transpose"};

        const size_t thread_size_row;
        const size_t thread_size_col;

        transpose(const size_t block_size,
                  cl::Context context,
                  const size_t thread_size_row,
                  const size_t thread_size_col);

        transpose(const transpose& other);

        std::string return_build_options(void);

        void run(cl::CommandQueue cmd_queue,
                 cl::Buffer matrix,
                 const int height,
                 const int width);
};

#endif
