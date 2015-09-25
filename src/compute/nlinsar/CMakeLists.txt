add_subdirectory(cl_kernels)
add_subdirectory(cl_wrapper)

include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/data)
include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/include/utils)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlinsar)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlinsar/cl_wrapper)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/common/cl_wrappers)

add_library(nlinsar_lib OBJECT
            nlinsar.cpp
            insarsim_simu.cpp
            nlinsar_sub_image.cpp
            ${despeckCL_SOURCE_DIR}/src/clcfg/clcfg.cpp
            smoothing)
