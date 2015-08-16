include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)

set(KERNEL_PATH kernel)

configure_file (
    "${despeckCL_SOURCE_DIR}/include/clcfg/clcfg.h.in"
    "${despeckCL_SOURCE_DIR}/include/clcfg/clcfg.h"
)

add_library(clcfg clcfg.cpp)
